# 算子性能数据库 Q1 工作计划 (v3.1)

**目标**: 2026.3.23 完成端到端集成，DeepSeek-V3 / Qwen3-32B 仿真误差 <15%
**基准日期**: 2026.3.5（周四晚发布，3.6 起执行）
**团队**: 6 人（1 SE + 5 开发）
**周期**: 3.6-3.23（三个 Phase）
**设计文档**: `OPERATOR_PERF_DATABASE_DESIGN_zh_v1.3.1.md`（同目录）
**穿刺总结**: `reports/spike_executive_summary_zh.md`
**目标版本**: CANN 8.5（vllm 0.15.0 + torch 2.9.0），数据目录 `vllm0.15.0_torch2.9.0_cann8.5/`
**变更日志**: `CHANGELOG_20260312.md`（同目录）

---

## 目录

- [1. 项目概览](#1-项目概览)
- [2. 进展管理](#2-进展管理)
- [3. 团队与职责](#3-团队与职责)
- [4. 任务依赖总览](#4-任务依赖总览)
- [5. Phase 1：核心集成 + Mini 验证（3.6-3.13）](#5-phase-1核心集成--mini-验证36-313)
- [5.5 C10 后续计划：HCCL 数据入库与验证](#55-c10-后续计划hccl-数据入库与验证)
- [5.6 C11：DispatchFFNCombine 子内核数据采集（v3.1 新增）](#56-c11dispatchffncombine-子内核数据采集v31-新增)
- [6. Phase 2：数据扩充 + 融合 Pass + DSV3 深度匹配（3.16-3.20）](#6-phase-2数据扩充--融合-pass--dsv3-深度匹配316-320)
- [7. Phase 3：端到端精度验证（3.19-3.23）](#7-phase-3端到端精度验证319-323)
- [8. 风险与缓解](#8-风险与缓解)
- [附录 A：分支策略](#附录-a分支策略)
- [附录 B：参考索引](#附录-b参考索引)
- [附录 C：进展管理细则](#附录-c进展管理细则)

---

## 1. 项目概览

### 1.1 目标与现状

为 TensorCast 构建基于实测 Profiling 数据的算子性能估算系统（`EmpiricalPerformanceModel + DataSource` 模式，设计文档 §1.1）。db-spike 穿刺已验证架构可行性：Qwen3-32B BF16 Prefill 匹配率 87%，计算算子 100%。核心代码 production-ready，直接作为产品分支基础。

> **v3.1 更新**：最终 E2E 验证目标版本调整为 CANN 8.5（vllm 0.15.0 + torch 2.9.0）。CANN 8.3 数据保留作为参考基线。CANN 8.5 引入 DispatchFFNCombine 超级融合（35.3% DSV3），需新增 C11 任务覆盖。

### 1.2 核心假设

| 假设 | 验证方式 | 若不成立的影响 |
|------|---------|-------------|
| CommAnalytic 在 Qwen3 Prefill 上精度可接受 | Phase 1 mini 端到端验证 | 通信查询路径优先级需提前 |
| DSV3 W8A8 op_mapping 可增量完成 | C3/C4 映射验证 | 映射工作量翻倍 |
| MC2 在 compile pass 中已正确融合 | XJT验证 | 需调整 composite fallback |
| 通用 shape 线性插值 + FIA sqrt 变换可满足所有场景（不需要 per-operator 维度声明） | TCX插值精度测试 + ZZY/HDY override 标注 | 需新增 kernel_overrides |
| CANN 8.5 DispatchFFNCombine 可通过 composite 分解查询覆盖 | C11 数据采集 + Phase 2 验证 | 需新增 TC 融合 pass（估计 5+ 天） |

### 1.3 交付标准

| 指标 | 目标值 |
|-----|-------|
| 端到端耗时误差 | <15%（对比实际 vLLM Profiling） |
| 单算子误差（已匹配） | <20% |
| 时间覆盖率 | >90% |

### 1.4 最终交付物（3.23）

| 交付物 | 验收标准 |
|-------|---------|
| CLI `--performance-model profiling --compile` | Qwen3-32B + DSV3 端到端可运行 |
| 精度报告（Qwen3-32B + DSV3） | 端到端误差 <15% |
| 完整数据库（CSV + YAML） | CANN 8.5 数据，覆盖 Tier 1/2 算子（设计文档 §7.1） |
| validate.py | 可重复验证 |
| 数据采集工具链（9 个工具） | 可重复执行 |

---

## 2. 进展管理

- **飞书日报**：每人每天更新进展/阻塞/风险信号（详见[附录 C](#附录-c进展管理细则)）
- **站会**：仅讨论阻塞项和风险，Phase 1/3 每日，Phase 2 隔日
- **DIMA 看板**：任务卡片状态同步，对 MY 合作方可见
- **Review 节点**：3.13 Phase 1 Review → 3.19 Phase 2 Review → 3.23 交付 Review

---

## 3. 团队与职责

### 3.1 分工总表

| 人员 | 投入 | 职责域 | 代码 Owner |
|------|------|--------|-----------|
| **ZH** | 50% | DataSource 查询引擎：`_lookup_compute` / `_lookup_comm` / `_lookup_composite` + review 全部查询代码 PR | `perf_database/*.py` |
| **TCX** | 100% | 数据层全链路：工具链 + Microbenchmark + Attention 查询与数据 + 基础插值；协助 SE 进展管理（日报跟踪、站会记录） | `tools/perf_data_collection/`, attention 查询, 插值 |
| **ZZY** | 100% | Qwen3 op_mapping：BF16 场景验证 + Decode 扩展 + 自动化方案 spec | `op_mapping.yaml` (Qwen3), 验证报告 |
| **HDY** | 100% | DSV3 op_mapping + HCCL：W8A8 映射 + 通信数据采集 + DSV3 Profiling 分析 | `op_mapping.yaml` (DSV3), HCCL 数据 |
| **XJT** | ~~70%~~ 已交接 | 集成层：CLI + compile pass 融合（A1/A2/A3 + MC2 验证 已完成） | CLI, `compilation/` |
| **LJW** | 100%（3.17起） | 接替 XJT：融合 Pass（DispatchFFNCombine 可行性评估 + 实现） | `compilation/` |
| **HXW** | SE | spec review + 决策 + 进展管理（不 own 产品代码） | — |

> **人员变动（3.12）**：XJT 工作交接给 LJW，LJW 3.17 起全职投入。XJT 已完成 A1/A2/A3 + MC2 验证 + KvRmsNormRopeCache 确认（mlapo 已覆盖，无需独立 pass）。

### 3.2 协作关系与接口

```
XJT→LJW(集成层) ZH(查询层) TCX(数据层) ZZY(Qwen3映射) HDY(DSV3映射)
 CLI/Pass          lookup引擎   CSV工具/插值    op_mapping验证     op_mapping+HCCL
    |                |         Attn查询              |                    |
    |                |              |                |                    |
    +--- pass 产出 --+-- 查询合入 --+-- mapping 同步 -+--------------------+
```

**接口点**（需 PR review 协调的地方）：
- TCX → ZH：`_lookup_attention()` 代码合入 `profiling_data_source.py`
- TCX → ZH：InterpolatingDataSource 代码合入 `perf_database/`
- ZZY/HDY → ZH：`op_mapping.yaml` 变更影响查询逻辑时需同步
- XJT → ZH：新增 compile pass 产生的 TC op 需同步到 `op_mapping.yaml`

HXW（SE）：决策 + 进展管理（TCX协助）；不 own 产品代码，按需参与技术讨论。

### 3.3 技术方案确认

每个技术方案由负责人自行起草并验证。验证方式：对照设计文档对应章节 + 穿刺报告已有结论，在日报中简要说明方案要点和验证结果即可。有疑问或分歧时在站会提出讨论。

| 方案 | 负责人 | 验证依据 | 完成时间 |
|------|--------|---------|---------|
| 17 项简化评估 | HXW | 穿刺报告 §5 | 3.6 |
| 计算+通信融合算子确认 | HDY | DSV3 Profiling CSV 中搜索计算+通信融合类 kernel Type（含 MC2 及其他融合形式） | 3.6（1h） |
| Attention 匹配规则 | TCX | 穿刺报告 §4.1 + 设计文档 §4.8，写单元测试验证 | 3.9 |
| 通信数据表格式 | ZH | 设计文档 §4.4 + §4.7，对照 `comm_config_example.yaml` | 3.9 |
| MoE/MLA 匹配规则 | ZH | 设计文档 §4.2 composite 分解表，写单元测试验证 | 3.12 |

---

## 4. 任务依赖总览

### 4.1 依赖图

```
          db-spike 已有代码 (feat/perf-database 基础)
                    |
    +---------------+---------------+---------------+
    v               v               v               v
 A1 CLI          B1 通信查询     C1+C2 算子清单   D1 解析验证
 (XJT)        (ZH)          (张+胡,并行)     (TCX)
    |               |               |               |
    v               v               v               v
 A2 端到端       B2 Composite    C3 Qwen3验证    D2 Attention
 (XJT)        (ZH)          (ZZY)         查询实现
    |               |          C7 DSV3映射       (TCX)
    v               |          (HDY)             |
 A3 融合merge       |               |               v
 + MC2验证          v               v            D3 基础插值
 (XJT)       B1+B2 完成     C3+C7+C8完成     (TCX)
    |               |               |               |
    +-------+-------+-------+-------+-------+-------+
            v                                       v
   Phase 1 交付 + Mini 端到端验证 (3.13)
            |
    +-------+-------+-------+-------+
    v       v       v       v       v
  E1-E4   F1      G1-G2   H1-H4   E5
  数据    融合    MoE/MLA  分析    Attn插值
  (TCX)(XJT)(ZH) (张+胡)  (TCX)
    |       |       |       |       |
    +-------+-------+-------+-------+
            v
   Phase 2 交付 + DSV3 Mini 验证 (3.19)
            |
    +-------+-------+
    v       v       v
  J1 Qwen3 J2 DSV3 J3 修复
  (祝+许)  (胡+许)  (张分析+祝/唐修复)
            v
   J4 精度报告 (3.23)
```

### 4.2 关键路径

`A1 → A2 → A3 → Mini 验证 → G1 → J2 → J4`

### 4.3 Phase 时间线

```
3.5(发布)  3.6 ──────── 3.13        3.16 ──────── 3.19  3.20 ──── 3.23
           ←── Phase 1 ──→ Review    ←── Phase 2 ──→ Review       交付
                                                 ←── Phase 3 ────→
```

**Phase 1 任务排布（3.6-3.13）**：

```
      3.6       3.9       3.10      3.11      3.12      3.13
       |         |         |         |         |         |
XJT |-- A1 ---|------ A2 ---------|-- A3+MC2验证 ------|
       |         |         |         |         |         |
ZH   |         |--- B1 ------------|--- B2 ------------|
       |         |         |         |         |         |
TCX |-- D1 ---|-- D2 Attention ---|-- D3 插值 --|D4---|
       |         |         |         |         |         |
ZZY |         |= C1+C2 =|--- C3 Qwen3验证 ---|C4+C5--|
       |         |         |         |         |         |
HDY |C6+MC2查 |--- C9 --|--- C7 DSV3映射 ----|C8+C10-|
```

---

## 5. Phase 1：核心集成 + Mini 验证（3.6-3.13，6 个工作日）

**目标**：CLI 端到端可运行 + 计算/通信/Attention/Composite 四条查询路径 + op_mapping 双模型验证 + 基础插值 + **Mini 端到端首次跑通**。

---

### 任务 A：CLI 集成 + Compile Pass（XJT，70%）

**目标**：让 `--performance-model profiling --compile` 端到端可运行，并验证已有融合 pass 正确工作。

**背景**：db-spike 已有 `empirical.py`（97 行）和 CLI 改动参考。`--compile` 是正确使用 profiling 模式的前提（穿刺报告 §3.3）。MC2 pass 已有完整实现（`compilation/freezing_passes/patterns/matmul_allreduce.py`，261 行，5 种量化变体），需验证其与 profiling 数据的对齐。

**修改范围**：`tensor_cast/scripts/text_generate.py`, `tensor_cast/core/model_runner.py`, `tensor_cast/core/config_resolver.py`

**参考**：设计文档 §5.1-§5.3（CLI 接口）、§9.1（融合 Gap 状态）

| # | 检查点 | 完成日期 | 验收标准 |
|---|-------|---------|---------|
| A1 | CLI `--performance-model {analytic,profiling}` + `--perf-database` 路径参数 | 3.9 → ✅ 3.10 | analytic 行为不变；profiling 模式创建 EmpiricalPerformanceModel |
| A2 | 端到端：Qwen3-32B Prefill `--performance-model profiling --compile` | 3.11 → ✅ 3.11 | 不报错，log_stats 输出命中率 |
| A3 | 融合 Pass merge + MC2 pass 验证 + KvRmsNormRopeCache 确认 | 3.13 → ✅ 3.11 | MC2 BF16+W8A8 验证通过；KvRmsNormRopeCache 被 mlapo 覆盖，无需独立 pass |

**计算+通信融合算子验证要点**：
- 确认 `--compile` 后 dispatch trace 中出现 `tensor_cast.matmul_all_reduce`（不再是分离的 mm + all_reduce）
- HDY 3.6 确认 DSV3 Profiling 中是否有计算+通信融合类 kernel Type（含 MC2 及其他融合形式）
- **结论（已确认）**：
  - **MC2（MatMul+AllReduce 融合）**：DSV3 Profiling 中无专用 kernel Type，matmul（`QuantBatchMatmulV3`）和通信（`hcom_reduceScatter_` / `hcom_allGather_`）分开记录 → 保留 `composite: true` + `sub_kernels: [QuantBatchMatmulV3, hcom_allReduce_]` 分解查询
  - **DispatchFFNCombine（计算+通信融合）**：DSV3 Profiling 中**存在**此融合 kernel，融合了 `all_to_all×2 + GroupedMatmul×2 + SwiGlu + MoE routing`，耗时占端到端 **35.3%**，是 DSV3 最重要的单一 kernel。TC 将其分解为 `permute_tokens + grouped_matmul×2 + swiglu + unpermute_tokens + all_to_all×2`，op_mapping.yaml 已配置 `composite: true` 处理，无需新增直接映射

---

### 任务 B：DataSource 查询路径（ZH，50%）

**目标**：在已有 `_lookup_compute()` 基础上，新增通信查询和 Composite 查询两条路径。

**背景**：当前 `profiling_data_source.py` 的 `lookup()` 中，`communication` 和 `composite` 两个分支直接 return None（穿刺简化项 S-11/S-12）。

**修改范围**：`tensor_cast/performance_model/perf_database/profiling_data_source.py`

**参考**：设计文档 §4.2（查询分派）、§4.4（通信查询）、§4.7（通信 CSV 格式）

**前置依赖**：通信数据表 spec（ZH自己起草，3.9 前完成，HXW review）

| # | 检查点 | 完成日期 | 验收标准 |
|---|-------|---------|---------|
| B1 | `_lookup_comm()`：从 OpInvokeInfo 计算 message_bytes + topology_tier，查询通信 CSV | 3.11 → ✅ 3.11 | topology_tier 精确匹配实现（8fa2da3） |
| B2 | `_lookup_composite()`：matmul_all_reduce 分解 + MLA 分解框架 | 3.13 → 🔄 进行中 | MC2 compute+comm sum 初稿完成；MoE/MLA spec 起草中 |

**通信查询实现要点**（设计文档 §4.2）：
- `args[0]` → `message_bytes = tensor.nelement() * tensor.element_size()`
- `rank_group` 位置因算子而异：all_reduce=args[2], all_gather=args[3], all_to_all=args[4]
- `topology_tier = comm_grid._get_topology_idx_for_group(rank_group)`
- CSV 按 `(num_devices, topology_tier)` 精确匹配

**Composite 查询实现要点**（设计文档 §4.2）：
- `composite: true` 时分解为 sub_kernels 逐个查询并求和
- MLA 分解复用 `performance_model/__init__.py` 已有 shape 推导逻辑
- 任一子内核未命中 → 整体 return None → fallback analytic

**说明**：`_lookup_attention()` 由TCX实现（任务 D2），提交 PR 后ZH review 并合入。

---

### 任务 C：op_mapping 系统化验证（ZZY + HDY，各 100%）

**目标**：系统性验证已有 op_mapping 映射，补充 DSV3 Decode W8A8 场景映射。这是端到端精度的**核心瓶颈** — 映射错误直接导致算子 MISS。

**背景**：穿刺阶段建立了 60+ 条映射，但仅在 Qwen3 BF16 Prefill 上验证。DSV3 Decode 有 10+ 个新 kernel type 需要映射（QuantBatchMatmulV3, GroupedMatmul, DequantSwigluQuant 等）。

**参考**：
- **映射方法论**：`tutorial/OP_PLUGIN_MAPPING_TUTORIAL.md`（正向/反向映射操作手册 + 速查表）
- **映射格式**：设计文档 §4.5（op_mapping.yaml 规格）
- **映射示例**：`examples/op_mapping_example.yaml`
- **算子分级**：设计文档 §7.1-§7.2（Tier 1/2/3 + 占比数据）

**修改范围**：`perf_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/v0.13.0/op_mapping.yaml`

**验证方法论**（每条映射的验证步骤）：
1. 从 Profiling 提取 kernel Type 及其 Input Shapes / Data Types
2. 用 analytic 模式跑 TC，导出 dispatch trace，找到对应的 TC op 及其 args shapes
3. 按 `tutorial/OP_PLUGIN_MAPPING_TUTORIAL.md` §6-7 确认 TC op → kernel Type 的映射链
4. 对比 TC args shapes 与 Profiling Input Shapes，记录差异（batch 维度、FRACTAL_NZ、padding 等）
5. 确认差异可被 `profiling_data_source.py` 的通用规则处理（穿刺报告 §3.1 八类差异）

#### ZZY（Qwen3 主线）

| # | 检查点 | 完成日期 | 验收标准 |
|---|-------|---------|---------|
| C1 | Qwen3 Profiling 算子清单：Top-20 (Type, 调用次数, 耗时占比) | 3.9 → ✅ 3.10 | 表格输出 |
| C2 | TC dispatch trace 导出 | 3.9 → ✅ 3.10 | **发现**：TC compile 路径不可行用于 op_mapping 验证，改用 AI/skill 方案辅助 |
| C3 | BF16 场景逐条映射验证 | 3.11 → ✅ 3.11 | 验证报告完成（97.01% 覆盖） |
| C4 | Qwen3 Decode 场景映射补充 + 验证 | 3.12 → 🔄 进行中 | shape 不匹配问题待解决 |
| C5 | op_mapping 自动化方案 spec | 3.13 | **调整**：TC compile 路径不可行，改为 AI/skill 方案 + 教程增补 |

#### HDY（DSV3 主线 + HCCL）

| # | 检查点 | 完成日期 | 验收标准 |
|---|-------|---------|---------|
| C6 | DSV3 Profiling 算子清单：Top-20 排序表 | 3.6 → ✅ 3.10 | 表格输出 |
| 计算+通信融合确认 | 查 DSV3 Profiling 计算+通信融合类 kernel Type | 3.6 → ✅ 3.6 | **结论**：DispatchFFNCombine 占 35.3%（见 A3 结论） |
| C9 | HCCL 数据采集方案：`generate_comm_microbench.py` 实现 | 3.10 → ✅ 3.11 | 脚本重构完成（单 session + 全局预热） |
| C7 | DSV3 W8A8 op_mapping 扩展 | 3.12 → ✅ 3.11 | op_mapping 覆盖 DSV3 Top-15 |
| C8 | W8A8 量化场景映射验证 | 3.13 → ✅ 3.11 | 验证报告完成（98.02% 覆盖） |
| C10 | HCCL 集群数据采集（4 种通信算子 x 各 topology_tier） | 3.13 → 🔄 部分完成 | CSV 已产出（tier=1/2）；tier=0 需多节点环境（见 §C10 后续计划） |

**插值 override 标注**：分析 op_mapping 时顺便确认各 kernel_type 是否需要插值特殊处理（`interpolation_policy.kernel_overrides`）。预期结果：仅 FusedInferAttentionScore 需要 sqrt 变换，其余算子均适用默认线性插值。

**双人交叉验证**：ZZY review HDY的 DSV3 映射，HDY review ZZY的 Qwen3 映射。

---

### 任务 D：数据采集工具链 + Attention + 插值（TCX，100%）

**目标**：验证数据解析工具 + 实现 Attention 查询 + 实现基础插值 + 算子发现工具。

**背景**：当前 7 个工具中只有 `parse_kernel_details.py` 完整（342 行），其他 6 个是 stub。Attention (`FusedInferAttentionScore`) 是 Prefill 中延迟最高的单算子（~100us+），对端到端精度影响最大（穿刺报告 §4.1）。插值是实用性的关键瓶颈（穿刺报告 S-17）。

**修改范围**：
- `tools/perf_data_collection/parse_kernel_details.py`, `discover_operators.py`
- `tensor_cast/performance_model/perf_database/profiling_data_source.py`（`_lookup_attention()` 方法）
- `tensor_cast/performance_model/perf_database/interpolating_data_source.py`

**参考**：
- Attention：设计文档 §4.8（FusedAttention 特殊处理）、穿刺报告 §4.1
- 插值：设计文档 §4.4（InterpolatingDataSource）、AI Configurator 实现（`src/aiconfigurator/sdk/perf_database.py` 插值方法）

| # | 检查点 | 完成日期 | 验收标准 |
|---|-------|---------|---------|
| D1 | `parse_kernel_details.py` 验证 | 3.6 → ✅ | 输出 CSV 与 db-spike 已有数据一致 |
| D2 | `_lookup_attention()` 实现 | 3.10 → 🔄 进行中 | 支撑 microbench 问题中 |
| D3 | InterpolatingDataSource 基础版 | 3.12 → 🔄 进行中 | |
| D4 | `discover_operators.py` | 3.13 | |

**D2 Attention 查询实现要点**（设计文档 §4.8）：
- 从 `OpInvokeInfo.args[6]`（seq_lens）计算 `batch_size = len(seq_lens)` 和 `avg_seq_len = mean(seq_lens)`
- 从 `OpInvokeInfo.args[0]`（query tensor）提取 `num_heads`, `head_dim`
- FIA CSV 索引维度：`(batch_size, avg_seq_len, num_heads, head_dim, dtype)`
- 区分 PA（PagedAttention, decode, seq_lens 长）和 FA（FlashAttention, prefill, query_lens 长）
- 提交 PR 后由ZH review 并合入 `profiling_data_source.py`

**D3 插值实现要点**（参考 AI Configurator + 设计文档 §4.4）：
- Wrapper 模式包装 ProfilingDataSource：精确命中 → 直接返回，未命中 → 插值
- **通用插值逻辑（不需要 per-operator 维度声明）**：dtype+format 精确匹配（已在 ProfilingDataSource 实现），shape 维度做最近邻搜索 + 线性插值
- 读取 `op_mapping.yaml` 的 `interpolation_policy.kernel_overrides` 应用特殊变换（当前仅 FIA 需要 sqrt）
- 提交 PR 后由ZH review 并合入 `perf_database/`

---

### Phase 1 里程碑（3.13）

**必达交付物**：

| 交付物 | 验收标准 | 负责人 |
|-------|---------|--------|
| CLI `--performance-model profiling` | 端到端可运行 | XJT |
| `_lookup_comm()` | 单元测试通过 | ZH |
| `_lookup_composite()` | matmul_all_reduce 分解通过 | ZH |
| `_lookup_attention()` | Qwen3 Prefill FIA 命中 | TCX → ZH review |
| InterpolatingDataSource 基础版 | 线性插值可用 | TCX → ZH review |
| op_mapping 验证报告（Qwen3 BF16） | 覆盖 Top-15 | ZZY |
| op_mapping 扩展（DSV3 W8A8） | 覆盖 Top-15 | HDY |
| HCCL 数据 | 集群采集完成 | HDY |
| 融合 Pass merge + MC2 验证 | 单元测试通过 | XJT |

**Mini 端到端验证（3.13，全员）**：

用已有数据跑 Qwen3-32B Prefill 端到端，记录：

| 指标 | 记录内容 |
|------|---------|
| 命中率 | HIT / MISS / FALLBACK 各多少 |
| Fallback 算子耗时占比 | 哪些算子走了 analytic fallback，占端到端百分比 |
| 已匹配算子误差 | 与 Profiling 实测对比 |
| 端到端初始误差 | 允许远超 15%，重点暴露系统性问题 |

**Go/No-Go**：若 >50% 算子 MISS 或 fallback 占比 >30%，Phase 2 优先级需重排。

---

## 5.5 C10 后续计划：HCCL 数据入库与验证

> **背景**：C10 初次采集（2026.3.11）已产出 4 个通信算子 CSV（all_reduce / all_gather / reduce_scatter / all_to_all），覆盖 tier=1（intra_pod，16 卡）。数据分析发现若干质量问题，需在 H3 交叉验证前完成修复和补采。

### 数据质量现状

| 文件 | 行数 | 问题 |
|------|------|------|
| `hcom_allReduce_.csv` | 22（重复） | 两次 torchrun append，需去重；1MB/256MB/512MB 有异常值 |
| `hcom_allGather_.csv` | 11 | 4KB/16KB 高延迟（HCCL JIT 初始化）；1MB/4MB 偏慢 |
| `hcom_reduceScatter_.csv` | 10 | 缺 512MB；4KB/16MB 异常 |
| `hcom_allToAll_.csv` | 11 | 文件名错误（应为 `hcom_alltoallv_.csv`）；4KB/16KB 高延迟 |

**根本原因**：旧脚本每个 op 独立 torchrun，HCCL 每次重新初始化，小消息命中 JIT 编译开销。

### 脚本修复（已完成，commit f16a6ac）

| 修复项 | 说明 |
|--------|------|
| 单 session 运行 | 所有 op + message_sizes 合并为一次 torchrun，HCCL 只初始化一次 |
| 全局预热 | 每个 (op, group) 先跑一次 1KB 触发 HCCL JIT 编译，再开始正式计时 |
| WARMUP_ITERS 10→20 | 每个 message_size 的预热轮次加倍 |
| tier=2 覆盖 | 新增 `--num-devices 2`，采集 die_level（同 node 内 2 卡）数据 |
| 文件名修正 | `_OP_TO_CSV_FILENAME` 映射 `all_to_all → hcom_alltoallv_.csv` |

### 后续任务清单

| # | 任务 | 负责人 | 截止 | 验收标准 |
|---|------|--------|------|---------|
| C10-1 | 重新采集：`bash run_comm_bench.sh ./hccl_data_v2`（单 session，含 tier=2） | HDY | 3.14 | 4 个 CSV，每个 22 行（11 sizes × 2 tiers），无重复行 |
| C10-2 | 数据入库：将 CSV 放入 `data/ATLAS_800_A3_752T_128G_DIE/hccl/v8.5/`（对应 `communication_data_ref: "../../hccl/v8.5/"`） | HDY | 3.14 | ProfilingDataSource `_lookup_comm` 能命中 |
| C10-3 | 冒烟验证：`pytest tests/perf_database/ -k comm -v` | HDY | 3.14 | 通信查询单元测试通过 |
| H3 | HCCL Test 交叉验证：用 hccl_test 工具对相同 message_sizes 跑一遍，与 Python benchmark 对比 | HDY | 3.18 | 偏差 <10%；重点验证 1MB/256MB/512MB 异常点 |
| C10-4（可选）| tier=0（inter_pod）数据采集：需多节点（>16 卡）环境 | HDY | 视资源 | 有多节点资源时补采 |

### 数据入库路径

```
tensor_cast/performance_model/perf_database/data/
└── ATLAS_800_A3_752T_128G_DIE/
    ├── vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5/
    │   └── op_mapping.yaml  ← communication_data_ref: "../../hccl/v8.5/"
    └── hccl/
        └── v8.5/            ← 新建目录，放 4 个 CSV
            ├── hcom_allReduce_.csv
            ├── hcom_allGather_.csv
            ├── hcom_reduceScatter_.csv
            └── hcom_alltoallv_.csv
```

### 异常值处理策略

重新采集后若仍有异常值（单次测量抖动），处理优先级：
1. **H3 交叉验证**：用 hccl_test 确认真实值，以 hccl_test 结果为准覆盖异常行
2. **InterpolatingDataSource**：异常值会被插值平滑，对端到端精度影响有限
3. **tier=0 缺失**：当前 DSV3 TP=4 EP=8 的 all_to_all 走 tier=0，暂时 fallback analytic，等多节点资源

---

## 5.6 C11：DispatchFFNCombine 子内核数据采集（v3.1 新增）

> **背景**：CANN 8.5 引入 DispatchFFNCombine 超级融合算子，融合 `all_to_all×2 + GroupedMatmul×2 + SwiGlu + MoE routing`，占 DSV3 Decode **35.3%**。当前 op_mapping 已配置 `composite: true` 分解，但子内核 CSV 数据不全。

| # | 任务 | 负责人 | 截止 | 验收标准 |
|---|------|--------|------|---------|
| C11-1 | 确认 DispatchFFNCombine 子内核列表 + 现有 CSV 覆盖情况 | HDY | 3.13 | 子内核清单 + 缺口报告 |
| C11-2 | 缺失子内核 CSV 数据采集（GroupedMatmulSwigluQuant, MoeDistributeDispatch/CombineV2, hcom_alltoallv_） | HDY | 3.18 | CSV 入库 |
| C11-3 | DispatchFFNCombine composite 分解端到端验证 | ZH | 3.19 | 误差 <20% |

> **注**：C11-1 截止日期提前至 3.13，作为 Phase 1 E2E 验证的前置条件。若子内核 CSV 数据充分，composite 分解即可覆盖；若不充分，Phase 2 需评估 TC 侧融合 pass 可行性（LJW F1）。

---

## 6. Phase 2：数据扩充 + 融合 Pass + DSV3 深度匹配（3.16-3.20，5 个工作日）

**目标**：Microbenchmark 数据扩充 + KvRmsNormRopeCache Pass + DSV3 MoE/MLA 匹配 + Attention 插值升级 + DSV3 mini 验证。

---

### 任务 E：Microbenchmark + Attention 升级（TCX）

**参考**：设计文档 §6.1-§6.4（数据库构建三步走）、§6.2（计算算子 Microbenchmark）、§6.4（FusedAttention Microbenchmark）

| # | 检查点 | 完成日期 | 验收标准 |
|---|-------|---------|---------|
| E1 | `generate_shape_grid.py`：按 kernel_type 分派生成逻辑（GEMM: 模型 N/K + M 网格; Attention: 模型 heads + batch×seq 网格; Elementwise: 模型 hidden + num_tokens 网格）+ powers-of-2 补充 | 3.16 | Qwen3 + DSV3 shape 网格覆盖实际维度 |
| E2 | `generate_microbench.py`：读 op_mapping.yaml 的 torch_npu_reference 生成脚本 | 3.17 | 生成的脚本语法正确 |
| E3 | 集群 Microbenchmark 采集 + `build_database.py` | 3.19 | 每个 kernel_type CSV 行数 > Profiling 原始 |
| E4 | FusedAttention Microbenchmark（构造 paged KV cache 输入） | 3.20 | FIA CSV 覆盖多种 (batch_size, seq_len) 组合 |
| E5 | Attention 插值 sqrt 变换：O(n^2) 算子插值前做 sqrt 线性化 | 3.20 | 不同 seq_len 下 FIA 插值误差 <20% |

---

### 任务 F：融合 Pass（LJW，100%，3.17 起）

**目标**：评估 DispatchFFNCombine TC 侧融合可行性；若可行则实现 pass，否则依赖 composite 分解兜底。

**背景变更（v3.1）**：
- ~~KvRmsNormRopeCache pass~~：**不再需要**（mlapo op 已覆盖，穿刺验证 3f82c2b）
- DispatchFFNCombine 成为 CANN 8.5 最高优先级融合需求（35.3% DSV3）
- LJW 接替 XJT，3.17 起投入

**参考实现**：
- 模式参考：`compilation/patterns/rms_norm.py`（544 行）+ `freezing_passes/grouped_matmul_swiglu_pass.py`（204 行）
- DispatchFFNCombine 融合了 `all_to_all×2 + GroupedMatmul×2 + SwiGlu + MoE routing`
- 需获取贺博的超级融合算子设计文档（3.12 站会待办）

| # | 检查点 | 完成日期 | 验收标准 |
|---|-------|---------|---------|
| F1 | DispatchFFNCombine 可行性评估 + 设计文档 | 3.18 | 评估报告（做/不做 + 理由） |
| F2 | （条件性）DispatchFFNCombine pass 实现 | 3.20 | 单元测试通过 |

**MoeGatingTopK**：Q1 不做 pass，用 op_mapping composite 或 analytic fallback 兜底。Q2 补 pass。

---

### 任务 G：MoE/MLA 匹配（ZH）

**参考**：设计文档 §4.2（composite 查询 + MLA 分解）

**前置依赖**：MoE/MLA spec（ZH起草 3.12，HXW review）

| # | 检查点 | 完成日期 | 验收标准 |
|---|-------|---------|---------|
| G1 | MoE 算子匹配：MoeGatingTopK, MoeDistributeDispatch/CombineV2 | 3.18 | 单元测试 |
| G2 | MLA 分解查询完善：区分 Prefill/Decode 子内核 shape（设计文档 §4.2 MLA 分解表） | 3.20 | 单元测试 |

---

### 任务 H：DSV3 深度分析 + 验证工具（ZZY + HDY + TCX）

| # | 负责人 | 检查点 | 完成日期 | 验收标准 |
|---|-------|-------|---------|---------|
| H1 | HDY | DSV3 Decode Profiling 逐层耗时分析 | 3.16 | 分析报告 |
| H2 | ZZY | TC vs Profiling 算子对齐表（Qwen3 + DSV3） | 3.18 | 对齐表格（匹配/不匹配/原因） |
| H3 | HDY | HCCL Test 交叉验证 | 3.18 | Python benchmark 与 hccl_test 偏差 <10%（见 §C10 后续计划） |
| H4 | ZZY | 未覆盖算子分析 + 耗时影响评估 | 3.20 | 缺口清单 + 优先级排序 |
| H5 | TCX | `validate.py`：逐算子 + 端到端精度报告输出 | 3.20 | 精度报告可输出 |

### Phase 2 检查点（3.19 Review + 3.20 收尾）

| 交付物 | 验收标准 | 负责人 |
|-------|---------|--------|
| 扩充 CSV 数据库 | shape 覆盖 > Profiling 原始 | TCX |
| FIA Microbenchmark + sqrt 插值 | 多种 batch/seq + 误差 <20% | TCX |
| validate.py | 精度报告可输出 | TCX |
| DispatchFFNCombine 可行性评估（+ 条件性 pass） | 评估报告 / 单元测试通过 | LJW |
| DSV3 MoE/MLA 匹配 | 单元测试通过 | ZH |
| DSV3 对齐分析 | 对齐表格 + 缺口清单 | ZZY + HDY |

**DSV3 Mini 验证**（3.19）：同 Phase 1 格式，覆盖 DSV3 Decode 场景。

---

## 7. Phase 3：端到端精度验证（3.19-3.23，3 个工作日）

**目标**：端到端精度 <15%，交付精度报告。

> **说明**：Phase 3 与 Phase 2 尾部有 1 天重叠（3.19-3.20），ZH和XJT可在 3.19 Phase 2 Review 后直接启动端到端验证。

| # | 检查点 | 完成日期 | 负责人 | 验收标准 |
|---|-------|---------|--------|---------|
| J1 | Qwen3-32B 端到端验证（Prefill + Decode） | 3.20 | ZH + XJT | 误差 <15%, 覆盖 >90% |
| J2 | DSV3 端到端验证（Decode, MoE + MLA） | 3.20 | HDY + XJT | 误差 <15%, 覆盖 >90% |
| J3 | 精度问题定位 + 修复 | 3.23 | ZZY分析 + ZH/TCX修复 | 补数据/修映射/调插值 |
| J4 | 精度总报告 | 3.23 | 全员 | 交付 |

---

## 8. 风险与缓解

| # | 风险 | 影响 | 概率 | 缓解措施 |
|---|------|------|------|---------|
| R1 | 集群资源不足 | E3/C10 延迟 | 中 | 3.10 前预约；Phase 1 用现有 Profiling 数据 |
| R2 | KvRmsNormRopeCache Pass 比预期复杂 | F1 延期 | 低 | op_mapping composite 兜底；有 RmsNorm+RoPE 现成 pattern 参考 |
| R3 | DSV3 MoE/MLA 映射复杂 | G1/G2 延期 | 中 | 穿刺已验证通用逻辑；HDY全职 DSV3 分析降低不确定性 |
| R4 | Attention 匹配精度不足 | 端到端误差超标 | 中 | FIA.csv 已有 67 行；E4 补 Microbenchmark；E5 sqrt 插值 |
| R5 | 端到端精度 <15% 难达到 | Phase 3 调优期不足 | 高 | **核心缓解**：Phase 1/2 各做 mini 验证提前暴露问题 |
| R6 | op_mapping 错误致系统性 MISS | 匹配率下降 | 中 | 双人交叉 review；discover_operators 检测覆盖率 |
| R7 | 通信占比高但精度不足（Qwen3 89.8%） | Qwen3 误差超标 | 中 | Phase 1 mini 验证确认 CommAnalytic 精度 |
| R8 | ZH 50% 导致 Phase 2 DataSource 进度不足 | G1/G2 延期 | 中 | TCX承担 attention+插值减轻ZH负担；MoE/MLA spec 提前准备 |
| R9 | ~~XJT被其他项目拖住~~ | ~~A2 延期影响全队~~ | ~~中~~ | **已关闭**：A1/A2/A3 已完成，XJT→LJW 交接 |
| R10 | 单算子与整网算子耗时 gap | Microbenchmark 数据无法直接反映整网场景（3.12 TCX 提出） | 高 | Phase 1 mini 验证暴露差距；TCX 调研解决方案；必要时引入校正因子 |
| R11 | CANN 8.5 DispatchFFNCombine 覆盖不足 | DSV3 35.3% 耗时无法匹配 | 高 | C11 子内核数据采集 + composite 分解兜底；LJW F1 评估 TC 融合可行性 |
| R12 | LJW 上手周期 | 新人需 2-3 天熟悉代码 | 中 | XJT 交接 + 现有 compile pass 代码参考丰富 |

---

## 附录 A：分支策略

```
develop (稳定主线)
  |
  +-- feat/perf-database (从 db-spike 创建)
        |
        +-- LJW: feat/perf-db-compiler (接替 XJT)
        +-- ZH:   feat/perf-db-datasource
        +-- TCX: feat/perf-db-toolchain
        +-- ZZY: feat/perf-db-op-mapping
        +-- HDY: feat/perf-db-op-mapping-dsv3
```

**操作步骤**：
- Step 1（3.6，HXW）：创建 feat/perf-database，确认测试通过
- Step 2（3.6-3.9，各负责人）：拉个人分支
- Step 3（3.23，HXW）：feat/perf-database PR 回 develop

---

## 附录 B：参考索引

每个任务涉及的设计文档/教程章节速查：

| 任务 | 设计文档章节 | 其他参考 |
|------|------------|---------|
| A1-A2 CLI | §5.1-§5.3 | — |
| A3 融合 Pass | §9.1 | `compilation/freezing_passes/patterns/matmul_allreduce.py` |
| B1 通信查询 | §4.2, §4.4, §4.7 | `examples/comm_config_example.yaml` |
| B2 Composite | §4.2 (composite + MLA 分解) | `performance_model/__init__.py` (shape 推导) |
| C1-C8 op_mapping | §4.5, §7.1-§7.2 | `tutorial/OP_PLUGIN_MAPPING_TUTORIAL.md`, `examples/op_mapping_example.yaml` |
| C9-C10 HCCL | §6.3 | HCCL Test 文档 |
| D1 解析 | §6.5 | — |
| D2 Attention | §4.8 | 穿刺报告 §4.1 |
| D3 插值 | §4.4 | AI Configurator `perf_database.py` 插值方法 |
| D4 发现 | §6.6 | — |
| E1-E4 Microbench | §6.1-§6.4 | — |
| F1 KvRmsNormRopeCache | §9.1 | `compilation/patterns/rms_norm.py`, `patterns/rotary_embedding.py` |
| G1-G2 MoE/MLA | §4.2 | — |

---

## 附录 C：进展管理细则

### 飞书日报

每人每天 18:00 前更新，模板：

```
【日报】姓名 日期

完成：
- [任务 ID] 具体完成内容

进行中：
- [任务 ID] 进展描述

阻塞：
- 无 / 描述阻塞原因和需要谁帮助

风险信号：
- 无 / 描述发现的潜在问题

明日计划：
- [任务 ID] 计划做什么
```

**规则**：
- "阻塞"= 我无法继续推进，需要外部帮助
- "风险信号"= 我能继续但发现了潜在问题
- 连续 2 天同一任务无进展且无阻塞，SE 主动询问

### DIMA 看板

按 Phase 分 Swimlane，每个检查点一张卡片。必填字段：Owner、Due Date、Status（To Do / In Progress / Review / Done / Blocked）。

### 站会规则

- 严格 15 分钟，每人 2 分钟
- **只回答两个问题**：1) 有阻塞需要帮助吗？2) 发现风险信号了吗？
- 所有人都回答"无"→ 3 分钟散会

### Review 节点

| 时间 | 形式 | 内容 |
|------|------|------|
| **3.13** | Review 会 1h | Phase 1 mini 端到端结果 + Phase 2 优先级调整 |
| **3.19** | Review 会 1h | DSV3 mini 验证 + Phase 3 go/no-go |
| **3.23** | Review 会 1h | 精度报告 Review |
