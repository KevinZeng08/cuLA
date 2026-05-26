## KDA Backward chunk_wy_dqkg_fused 开发经验总结

> **作者**: 博渔 (boyu.zbw)
> **时间跨度**: 2026-04-14 ~ 2026-05-13（约一个月）
> **目标**: 在 Blackwell (SM100) 上实现高性能 KDA backward wy_dqkg fused CuTeDSL kernel，超越 FLA Triton baseline

---

### 一、项目概览

KDA（Kimi Delta Attention，参见 [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) 及 [Kimi Linear 技术报告](https://yzhang.site/assets/pubs/techreport/2025/kda.pdf)）的 backward pass 中，`chunk_wy_dqkg_fused` 是计算量最大、最复杂的 kernel。它将 dq、dk、dv2、dg、db、dA 共 6 个梯度输出的计算融合在一个 persistent kernel 中，涉及 **9 个 GEMM（全 SS 模式）**、大量 CUDA Core element-wise 运算以及复杂的多 warp 数据流管道。

对应的 Triton reference 实现位于 `fla.ops.kda.chunk_bwd.chunk_kda_bwd_kernel_wy_dqkg_fused`。

#### 核心挑战

- **计算密度极高**: 单 kernel 内含 V-loop 5 个 SS GEMM + Post-V 2 个 GEMM + Phase 3 dA 后处理 2 个 GEMM = **9 个 MMA**，加上大量 element-wise 后处理（gate exp、causal mask、reduce 等）
- **TMEM 容量紧张**: 需要同时保持 dA、dq、dk、dw 4 个 fp32 accumulator + 若干临时 buffer
- **SMEM 容量紧张**: 多个 tensor 的 double buffer + 各种 swizzle 布局 operand，occ=1 用满 227KB SMEM
- **寄存器压力巨大**: WG0/WG1 各需 208 个寄存器用于 element-wise 运算
- **同步复杂度高**: 12 个 warp 间需要 18+ 条 barrier/pipeline 协调数据流

#### 最终成果

- Blackwell (SM100) 上基于 CuTeDSL 的 3-WG Warp Specialization 实现
- 核心代码量: `chunk_wy_dqkg_sm100.py`（3232 行）+ `ptx_umma_ext.py`（958 行）+ `intrinsics_sm100.py`（401 行）
- 相比 FLA Triton baseline 取得显著加速

---

### 二、开发流程

#### Phase 1: CuTeDSL 实现 (04-14 ~ 05-04)

在同一个 flashla 分支上，并行开始 CuTeDSL Python DSL 实现。经历了 v1 → v2 → v3 → **[newpipe]** 的架构演进。

| 日期 | 里程碑 | 说明 |
|------|--------|------|
| 04-16 | `add cutedsl wy dqkg` | 初始 CuTeDSL 版本 |
| 04-20 | **设计文档 + 基础设施** | 完成 SMEM 复用策略（`chunk_wy_dqkg_design_doc.md`，762 行）、TMEM 布局设计、TMA/MMA atom 选型；封装 inline PTX 文件 |
| 04-22 | v1 代码结构 | 3-WG warp specialization 框架 + mma.ws intrinsics；同期完成 bwd_intra CuTeDSL |
| 04-22 | 渐进式开发计划 | 完成 `chunk_wy_dqkg_dev_plan.md`，定义 Phase A~F 的 output-by-output 验证策略 |
| 04-26 | **v1 dv2 正确性通过** | 第一个输出验证通过（fixed len） |
| 04-27 | v1 varlen 通过 | 变长序列也正确 |
| 04-29 | v2 dq 通过 | 发现 pipeline 设计需要重构 |
| 04-30 | v3 a_stage 探索 | A loop pipeline stage 实验 |
| **05-01** | **[newpipe] 架构确立** | 推翻 v1/v2 的 pipeline 设计，改为 2-WG epilogue 协作 |
| 05-01 | newpipe dv2 + dq 通过 | 新架构快速恢复核心输出 |
| 05-02 | newpipe db + dk 通过 | |
| 05-04 | **newpipe dg + dA 通过** | **全部 6 个输出正确，功能开发完成** |

**关键决策 — [newpipe] 重写**: v2 阶段（dq 正确性刚通过）发现原始 pipeline 设计（1 WG epilogue）下，单个 WG 需要同时持有 dq/dk/dw/dg 等多个 fp32 accumulator 做 element-wise 处理，寄存器严重溢出到 LMEM。果断推翻重写为 `[newpipe]` 架构（WG0+WG1 两个 WG 在 head dim 维度切分 epilogue 工作），虽然损失约 1 周工作量，但新架构 4 天内恢复了全部输出的正确性，并且 WG0/WG1 各只需 208 个寄存器。

#### Phase 2: 迁移到 cula_inner (05-05)

在 commit `47b2c8d`（"init sm100 bwd wy dqkg"）将 flashla 的实现 adapt 到 cula_inner 仓库（5106 行新增），包括：

| 文件 | 行数 | 说明 |
|------|------|------|
| `cula/ops/chunk_wy_dqkg_sm100.py` | 3257 | 核心 kernel |
| `cula/ops/ptx_umma_ext.py` | 958 | tcgen05.mma.ws 指令封装 |
| `cula/ops/intrinsics_sm100.py` | 359 | SM100 helper intrinsics |
| `benchmarks/bench_kda_bwd_wy_dqkg_sm100.py` | 439 | benchmark 脚本 |
| `benchmarks/utils.py` | 93 | 公共测试工具 |

#### Phase 3: 性能优化与功能完善 (05-05 ~ 05-13)

在 cula_inner 上持续优化 SM100 kernel，**累计 24 commits**：

| 优化项 | 收益 | 说明 |
|--------|------|------|
| dgk/kdk 计算顺序调整 | **~1%** | 优化指令调度，减少依赖链 |
| A loop stage 1→2 (double buffer) | **12%** | 最大单项收益，掩盖 A 矩阵 TMA load 延迟 |
| WarpGroup sync tuning | **2.7%** | 精细调整 WG 间同步时序 |
| dg store 移至 aux warp (warp 10/11) | **2.8%** | 将 dg TMA store/reduce_add 从 CUDA warp 解耦到 WG2 的 aux warp |
| TMA store for non-tail chunk | — | 非尾部 chunk 使用 TMA S2G 替代逐元素 scatter |
| TMEM offloading (dq → TMEM) | **6.8%** | dq scaled 结果暂存 TMEM[368,432)，降低寄存器峰值 |
| 移除 TS ws 模式 MMA | — | 部分 GEMM 的 ws 模式反而更慢，回退为标准模式 |
| GVA (Grouped Value Attention) 支持 | — | 功能扩展：HV != H 的场景 |
| db atomic add 修复 | — | 多 chunk 下 db reduce 使用 `cute.arch.atomic_add` |
| deterministic db reduce | — | 保证跨运行数值确定性 |
| 切换到 UMMA pipeline | — | 使用 CuTe 标准 `PipelineTmaUmma` 替代手动 mbarrier 管理 |
| h/dh/v 分配到不同 consumer | — | MMA 消费 h 后 CUDA warp 也需要 h 做 dgk，分配不同 consumer pipeline 避免死锁 |
| TMA store desc prefetch | — | 减少 TMA store descriptor 的首次使用延迟 |

---

### 三、核心技术亮点

#### 1. 3-WG Warp Specialization（12 warps / 384 threads）

将 CTA 的 12 个 warp 划分为三个 WarpGroup，实现高寄存器压力的 CUDA Core 计算路径与 IO/Tensor Core 路径**彻底解耦**：

```
┌─────────────────────────────────────────────────────────────┐
│  WG0 (warp 0-3)     │  WG1 (warp 4-7)     │  WG2 (warp 8-11)  │
│  CUDA Core epilogue │  CUDA Core epilogue │  warp 8: MMA dispatch │
│  208 regs/thread    │  208 regs/thread    │  warp 9: TMA Load     │
│  head dim 低半      │  head dim 高半      │  warp 10-11: Aux I/O  │
│  + TMA Store        │  + TMA Store        │  88 regs/thread       │
└─────────────────────────────────────────────────────────────┘
```

**设计动机**: wy_dqkg kernel 的 CUDA Core element-wise 运算（gate exp2、causal mask、dgk/db reduce、dg 计算等）需要同时持有多个 fp32 中间结果，消耗大量寄存器。如果与 MMA 放在同一 WG，寄存器溢出到 LMEM 导致严重性能退化。3-WG 分离后：
- **WG0/WG1**: 各 4 个 warp（128 线程），在 head dim (BK=128) 维度切分，每个 WG 负责 64 列的 element-wise 运算（gate exp、dq/dk scale、kg 计算、dg 计算等），以及 epilogue 的 TMEM→Register drain + TMA store
- **WG2**: 4 个 warp 分工明确——warp 8 负责 MMA dispatch（`elect_one` 发射 `tcgen05.mma.ws`），warp 9 负责 TMA G2S 加载，warp 10-11 负责辅助 IO（dg TMA store/reduce_add、gn 读取等）

#### 2. tcgen05.mma.ws 指令使能与 SM100 intrinsics

**问题**: chunk_size=64 (BT=64) 意味着 MMA 的 M 维度只有 64。SM100 的 `tcgen05.mma` 指令要求 TMEM 有 128 lanes（M=128 才能填满），BT=64 时常规模式下有一半 TMEM lane 浪费。

**解决方案**: 在 CuTeDSL 中通过 NVVM intrinsics 使能 `tcgen05.mma.ws`（warp-specialized MMA）指令。`.ws` 模式下使用 `cta_group::1`（而非 `cta_group::2`），只占用一半 TMEM lanes（64 列），在 M=64 时恰好用满。

**实现** (`ptx_umma_ext.py`, 958 行):
- 手工构建 SMEM descriptor（`Tcgen05SmemDescriptor`：start_address + SBO + LBO + layout_type）
- 封装 `tcgen05mma_ws_ss_f16` 内联 PTX：支持 K,K / K,MN / MN,MN 三种 transpose 模式
- 定义 instruction descriptor 常量（`IDESC_F16_M64_N128_K_MN` 等）编码 M/N/TransposeA/TransposeB/dtype 信息

该 kernel 实际使用了 **4 种 instruction descriptor**（`ptx_umma_ext.py` 中定义了 6 种供复用）：
- M64×N128 K,K（V-loop: dq、dw、dk）/ MN,MN（Post-V: dkgb）
- M64×N64 K,K（V-loop: dA; Post-V: dA+=dw@kg^T; Phase 3: dA2=dA@A）/ MN,MN（V-loop: dvb; Phase 3: dA3=A@dA2）

#### 3. Tile Size 与 Pipeline Stage 设计

```
BT = 64   (chunk size)
BK = 128  (head_dim_k, 单 K tile)
BV = 64   (V tiling, num_v_tiles = head_dim_v / BV = 128/64 = 2)
```

**Pipeline Stage 配置**:

| Pipeline | Stage 数 | 说明 |
|----------|---------|------|
| A loop (加载 A 矩阵) | **2** (double buffer) | A[BT,BT] = 64×64 bf16 = 8KB per stage; 掩盖 TMA load 延迟，**12% 收益** |
| V loop (加载 h/dh/do/vnew/dv) | **2** (double buffer) | h[BV,BK] = 64×128 bf16 = 16KB per stage; 掩盖 V 维度迭代间的 TMA 延迟 |
| K loop | **1** | BK=128 = head_dim_k，只有单次迭代 |
| MMA stage | **1** | MMA 结果直接消费 |

A loop double buffer 是最大的单项性能收益（**12%**）：A[BT,BT] = 64×64 矩阵的 TMA 加载延迟较高，double buffer 使下一个 chunk 的 A 加载与当前 chunk 的计算完全重叠。

#### 4. SMEM 分配：occ=1，用满 227KB

Blackwell SM100 单 SM 最大可用 227KB SMEM。本 kernel 采用 `min_occupancy=1`（单 CTA 独占 SM），SharedStorage 中所有 buffer **独立分配**（无 union 复用），充分利用 SMEM 容量来为各 tensor 提供独立空间，简化 barrier 同步逻辑：

```
┌──────────┬───────┬──────────────────────────────────────────────────┐
│ Buffer   │  大小  │ 用途                                             │
├──────────┼───────┼──────────────────────────────────────────────────┤
│ buf_A    │ 16 KB │ A[BT,BT] MN-major, a_stage=2 (double buffer)    │
│ buf_k    │ 16 KB │ k[BT,BK], stage=1; 复用为 kg 存储               │
│ buf_g    │ 32 KB │ g[BT,BK] fp32, stage=1                          │
│ buf_q    │ 16 KB │ q[BT,BK], stage=1; 复用为 dA bf16 staging       │
│ buf_h    │ 32 KB │ h[BK,BV] K-major, vloop_stage=2                 │
│ buf_dh   │ 32 KB │ dh[BK,BV] K-major, vloop_stage=2                │
│ buf_do   │ 16 KB │ do[BT,BV] K-major, vloop_stage=2                │
│ buf_dv   │ 16 KB │ dv[BT,BV] K-major, vloop_stage=2                │
│ buf_vnew │ 16 KB │ vnew[BT,BV] K-major, vloop_stage=2              │
│ buf_v    │ 16 KB │ v[BT,BV], vloop_stage=2                         │
│ buf_dw   │ 16 KB │ dw[BT,BK] K-major, stage=1                      │
│ scalar   │~1.8KB │ beta(256B), db(512B), gn(512B), dgk(512B)       │
├──────────┼───────┼──────────────────────────────────────────────────┤
│ barrier  │  ~1KB │ 30+ mbarrier slots                               │
╞══════════╪═══════╪══════════════════════════════════════════════════╡
│ 合计     │~226KB │ < 227KB SMEM 上限 ✓ (occ=1)                     │
└──────────┴───────┴──────────────────────────────────────────────────┘
```

**关键 SMEM 设计点**:
- **dw 的两种 swizzle 视图共用同一片 SMEM**: dw 在 dA GEMM 中作为 SS opA（K-major），在 dkgb GEMM 中作为 SS opB（MN-major）。代码中 `sDw_k` 和 `sDw_mn` 都指向同一个 `buf_dw`，只是用不同的 swizzle layout 解释同一片内存，无需额外 SMEM 或 re-swizzle
- **buf_k 复用为 kg**: k 加载后经 CUDA Core element-wise 计算 `kg = k * gk_exp` 后回写到同一 buffer，作为后续 `dA += dw @ kg^T` 的 opB
- **buf_q 复用为 dA staging**: Phase 3 中 dA bf16 结果写入 buf_q 的 SMEM 空间，作为后续 MMA 的操作数

#### 5. TMEM 布局：512 列全利用

SM100 TMEM 在 `cta_group::1` + `.ws Layout E` 下提供 512 列。本 kernel 的 TMEM 分配方案：

```
TMEM 列     用途                       生命周期
[0, 32)     dA fp32 accumulator        V-loop 全程 → Phase 3 覆写为 dA_bf16[0,16)
[32, 96)    dq fp32 accumulator        V-loop 全程 → Post-V drain → Phase 3 复用
[96, 160)   dk fp32 accumulator        V-loop 全程 → Post-V drain → Phase 3 复用
[160, 224)  dw fp32 acc → dvb 时分     V-loop: dw → Only-once: drain 后 dvb 覆写
[224, 256)  dvb / flex 临时空间        时分复用：dvb 结果 / 其他临时
[256, 272)  A_bf16 TS opA (reserved)   预留，当前未使用
[272, 336)  dkgb fp32 accumulator      Post-V 使用
[336, 368)  dA2 fp32 (Phase 3)         Phase 3 dA@A 和 A@dA
[368, 432)  dq_scaled (TMEM offload)   Post-V: 存放 dq*gk_exp*scale，供 dg 计算使用
```

**TMEM Offloading 策略** (6.8% 收益): dq 经过 element-wise scale（`dq *= gk_exp * scale`）后，结果需要在后续 dg 计算中使用（`dg = q * dq - ...`）。如果 dq_scaled 留在寄存器中，整个 Post-V 阶段都保持 live，大量占用寄存器。改为先 store 到 TMEM[368,432)（64 列），dg 计算时再 load 回来：

```
传统路径:  dq_scaled 常驻寄存器 → 整个 Post-V 阶段 → 寄存器爆炸 → spill to LMEM
TMEM 路径: dq_scaled → TMEM[368,432) → 需要时 load back → 片上延迟，无 LMEM 访问
```

#### 6. Kernel 内部 MMA 全景

本 kernel 共发射 **9 个 GEMM**，全部为 SS (SMEM×SMEM) 模式：

```
V-loop (每 V-tile 迭代, 共 5 个):
  ❶ SS m64n128 K,K:   dq  += do @ h        [BT,BK] += [BT,BV] @ [BV,BK]
  ❷ SS m64n64  MN,MN: dvb  = A @ dv        [BT,BV]  = [BT,BT] @ [BT,BV]
  ❸ SS m64n128 K,K:   dw  += dv @ h        [BT,BK] += [BT,BV] @ [BV,BK]
  ❹ SS m64n64  K,K:   dA  += dv @ v^T      [BT,BT] += [BT,BV] @ [BT,BV]
  ❺ SS m64n128 K,K:   dk  += vnew @ dh     [BT,BK] += [BT,BV] @ [BV,BK]

Post-V (2 个):
  ❻ SS m64n128 MN,MN: dkgb = A @ dw        [BT,BK]  = [BT,BT] @ [BT,BK]
  ❼ SS m64n64  K,K:   dA  += dw @ kg^T     [BT,BT] += [BT,BK] @ [BT,BK]

Phase 3 — dA 后处理 (2 个):
  ❽ SS m64n64  K,K:   dA2  = dA_bf16 @ A   [BT,BT]
  ❾ SS m64n64  MN,MN: dA3  = A @ dA2       [BT,BT]
```

所有 MMA 均通过 `tcgen05.mma.ws` 的 `elect_one` 模式由 warp 8 单独 dispatch。A 矩阵始终从 SMEM（buf_A）读取。

---

### 四、踩过的坑和经验教训

#### 坑 1: Pipeline 架构推翻重来

**flashla v1/v2 → [newpipe] 重写**: 最初设计为 1 个 WG 做全部 epilogue（element-wise + store），开发到 v2 dq 正确性通过后，`--ptxas-options --verbose` 显示寄存器严重溢出。根因是单个 WG 需要同时持有 dq/dk/dw/dg 等多个 fp32 accumulator 的 element-wise 中间结果。

推翻为 `[newpipe]` 架构——WG0 + WG1 两个 WG 在 head dim 维度切分工作，每 WG 只负责 64 列而非 128 列，寄存器压力减半。新架构虽然损失约 1 周的开发进度，但 4 天内恢复了全部输出的正确性。

**教训**: 对于 9 个 GEMM + 大量 element-wise 的超复杂 kernel，**pipeline 架构必须在设计阶段通过寄存器预算评估确定**，而不是功能开发完再优化。

#### 坑 2: SMEM Swizzle 布局 — K-major vs MN-major 不能互用

SM100 UMMA 的 SS 模式 opA 和 opB 使用不同的 swizzle atom：
- K-major: `Swizzle<3,4,3>` + `(8,64):(64,1)` — K 维度连续
- MN-major: `Swizzle<3,4,3>` + `(64,8):(1,64)` — MN 维度连续

**同一份 SMEM 数据不能在两种 layout 间复用**，必须通过 CUDA Core 做 re-swizzle。

这个问题在 **dv** 和 **dw** 两个 tensor 上最为突出：
- **dv**: 在 `dw += dv @ h`（SS K,K opA）中需要 K-major，在 `dvb = A @ dv`（SS MN,MN opB）中需要 MN-major → CUDA Core 从 K-major buffer re-swizzle 写入 MN-major buffer
- **dw**: 在 `dA += dw @ kg^T`（SS K,K opA）中需要 K-major，在 `dkgb = A @ dw`（SS MN,MN opB）中需要 MN-major → 从 TMEM drain 时直接双写到 `buf_dw`（K-major, `dw_k_opA_smem`）和另一份 MN-major layout（`dw_mn_opB_smem`）

设计文档 §2、§6.11 对此有详细的布局推导。

#### 坑 3: WG 间 Barrier 同步与死锁

本 kernel 使用了 **18+ 条 pipeline/barrier** 协调 12 个 warp 间的数据流（参见 dev_plan §8 Barrier 总览）。涉及 `PipelineTmaAsync`、`PipelineTmaUmma`、`PipelineAsyncUmma`、`PipelineUmmaAsync` 等多种 pipeline 类型。

遇到的典型问题：
- **db atomic 竞争** (`b141030`): 多 chunk 持久化 kernel 中，不同 tile 的 db 写回到同一 GMEM 位置（同一 token 的 beta 梯度跨 K-loop 累加），需要 `cute.arch.atomic_add` 而非直接 store
- **非确定性 reduce** (`fb90057`): 初始的 db reduce 使用 `atomicAdd`，不同运行间浮点累加顺序不同导致结果微小差异，debug 时难以区分是 bug 还是精度差异。改为确定性 reduce 后可完全复现

**教训**:
1. Persistent kernel 中任何跨 tile 累加的输出（如 db 跨 K-loop）都必须使用 atomic
2. 优先使用 deterministic reduce
3. 每条 barrier 的 producer/consumer 参与线程数必须精确匹配 `mbarrier_init` 时指定的 count

#### 坑 4: 数值稳定性

- **exp2 溢出**: gate 值 g 非常负时，`exp2(gn - 2*g)` 中指数差值可能超过 fp32 范围产生 Inf/NaN。需要对中间值做 clamp
- **dg 精度**: `dg = q*dq - k*dk + m_last*dgk + kg*dkgb*beta` 是多项累加，各项量级差异大。必须使用预计算的 `kg = k * exp2(g)` 和 `gk_exp = exp2(g)` 保持精度一致性，不能分别计算再组合
- 开发后期系统性加入了 **nan/inf check** 测试用例，对所有 6 个输出逐元素检测

#### 坑 5: tcgen05.mma.ws 模式并非万能

初始版本尝试对部分 GEMM 使用 TS（TMEM×SMEM）模式（A 矩阵作为 TMEM operand），但 profile 后发现 TS 模式反而更慢（commit `6fefaad` "remove TS ws mode MMA"），最终全部改为 SS（SMEM×SMEM）模式。根因是 TS 模式需要将 A 矩阵先加载到 TMEM 再作为 operand，引入额外的 TMEM 读写开销，对于 A[BT,BT]=64×64 这种小矩阵不划算。

**教训**: TS 模式并非总优于 SS 模式，需要逐个 GEMM profile 对比。对于小尺寸 operand，SS 模式（两个 operand 都从 SMEM 读取）的额外 SMEM 带宽开销可能小于 TS 模式的 TMEM 管理开销。

---

### 五、开发方法论

#### 1. 渐进式 Output-by-Output 验证

参照 dev_plan 中的分阶段策略，严格按 Phase B → C → D → E 的顺序，**每个输出独立验证后再开发下一个**：

```
Phase B: dv2 (最简单: 1 SS MMA + scalar 乘 + store)     ← 验证门控 err_ratio < 0.008
Phase C: dq, dk (V-loop 核心 + Post-V 变换 + 2 MMA)    ← 验证门控 err_ratio < 0.008
Phase D: dg, db (scalar reduce + element-wise)           ← 验证门控 err_ratio < 0.02
Phase E: dA (Phase 3 后处理: 2 SS MMA + causal mask)    ← 验证门控 err_ratio < 0.005
```

这种策略的核心价值是**故障隔离**：当某个输出出错时，可以确定问题出在当前 Phase 新增的代码中，而非之前已验证的部分。

#### 2. 设计文档先行

在写第一行 kernel 代码之前，先完成了两份详细的设计文档：
- **`chunk_wy_dqkg_design_doc.md`**（762 行）: SMEM/TMEM 布局设计、MMA operand 约束分析（K-major vs MN-major）、TMEM 列分配 Gantt Chart、资源预算总结
- **`chunk_wy_dqkg_dev_plan.md`**: 渐进式开发计划，每个 Phase 的 MMA 调用序列、barrier 配置、验证门控阈值

设计文档确保了在实现前就完成了最关键的**资源预算验证**（TMEM 512 列够用、寄存器预算可行），避免了实现到一半才发现资源不够的风险。

#### 3. Profile-Driven 优化

所有性能优化都基于 NCU profile 数据驱动：
- **A loop double buffer**: NCU stall 分析显示 TMA load A 是最大的等待瓶颈 → stage=2
- **TMEM offloading**: NCU 显示 register spill to LMEM 集中在 Post-V element-wise → dq_scaled 存 TMEM
- **dg store → aux warp**: NCU timeline 显示 dg TMA store 与后续计算串行 → 移到 aux warp 异步执行
- **dgk/kdk 计算顺序**: 基于依赖链分析调整指令发射顺序

---

### 六、代码规模和工程量

| 组件 | 行数 | 说明 |
|------|------|------|
| `chunk_wy_dqkg_sm100.py` | 3232 | SM100 核心 kernel（cula_inner 最终版） |
| `ptx_umma_ext.py` | 958 | tcgen05.mma.ws inline PTX 封装 |
| `intrinsics_sm100.py` | 401 | SM100 helper（tcgen05 ld/st、fence、reinterpret_cast 等） |
| `chunk_wy_dqkg_fused_v2.py` | 3257 | flashla 版本 kernel |
| `chunk_wy_dqkg_fused.py` | 2425 | flashla v1 版本（后被 v2 替代） |
| `chunk_wy_dqkg_design_doc.md` | 762 | SMEM/TMEM 设计文档 |
| `chunk_wy_dqkg_dev_plan.md` | ~500 | 渐进式开发计划 |
| flashla commits | **30+** | CuTeDSL v1 ~ newpipe |
| cula_inner commits | **24** | adapt + 性能优化 + 功能完善 |

---

### 七、最终收益总结

**性能提升路径 (SM100, cula_inner 阶段)**:

```
baseline (adapt from flashla)
  → dgk/kdk 计算顺序优化    (~1%)
  → A loop double buffer     (+12%)
  → WG sync tuning           (+2.7%)
  → dg store → aux warp      (+2.8%)
  → TMEM offloading (dq)     (+6.8%)
  ─────────────────────────────
  总计: ~25%+ 累计优化
```

**技术沉淀**:
- 一套可复用的 SM100 `tcgen05.mma.ws` CuTeDSL 封装（`ptx_umma_ext.py`），定义 6 种 instruction descriptor（kernel 使用 4 种）、支持 K,K / K,MN / MN,MN 三种 transpose 模式
- 精细的 3-WG Warp Specialization 设计模式（WG0/WG1 CUDA Core + WG2 MMA/IO），可推广到其他寄存器压力高的 kernel
- TMEM Offloading 策略：利用 TMEM 空闲列暂存中间结果替代寄存器 spill，适用于任何 TMEM 列有余量的场景
- SMEM 分配策略：occ=1 独立 buffer 简化同步 + 关键 buffer 时间复用（buf_k→kg、buf_q→dA staging），适用于多 GEMM 融合 kernel
- 完整的开发方法论：**设计文档先行 → CuTeDSL 渐进式实现 → Profile-Driven 优化**