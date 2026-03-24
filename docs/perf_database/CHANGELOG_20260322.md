# CHANGELOG 2026-03-22 → 2026-03-24

设计文档 v1.4 → v1.5 + 工作计划 v3.2 → v3.3

两批变更：(1) 3.22 基于 ZH 的 FIA 研究报告重设计 FIA lookup 方案；(2) 3.24 基于 TCX/HDY 的工具链更新重写 §6 数据库构建方案 + 新增理论导向 Shape Grid 策略。

---

## 设计文档变更 (v1.4 → v1.5)

### 重设计

- **§4.8**: FusedAttention 特殊处理全面重写（原 1 段 → 7 个子节 §4.8.1-§4.8.7）
  - §4.8.1: KV 维度问题 — 确认 CANN 8.3/8.5 均为预分配 pool shape（by design），non-paged prefill 是唯一例外但数据极少（2/24 行）
  - §4.8.2: avg_seq_len 必要性 — profiling FIA 数据不能用于查询（缺 context length），仅做 E4 microbench 验证 ground truth
  - §4.8.3: CSV 格式 — 结构化 CSV `(batch_size, avg_seq_len, ...)` → enriched CSV（raw 31 slots + `avg_seq_len` + `sparse_mode` 2 列）。基于 CANN 源码分析确认只需 2 个 enriched 列，其余维度（N, D, num_kv_heads, dtype）可从 raw CSV 推导
  - §4.8.4: 查询逻辑 — `_lookup_attention()` 覆盖为 6 维匹配（N, D, num_kv_heads, dtype, sparse_mode 精确 + avg_seq_len sqrt 插值），忽略 KV slots。CANN FIA 有 4 条 kernel 路径（FAI/IFA/PFA/V3），sparse_mode 2/3 跳过 ~50% 计算
  - §4.8.5: CANN 版本差异 — 8.3 TND `(T,16,512)` vs 8.5 BNSD `(B,16,1,512)` layout 归一化（统一 squeeze 到 3D）
  - §4.8.6: MLA composite FIA 子内核 — decode 路径通过 `_lookup_attention_by_params()` 查 enriched CSV
  - §4.8.7: DSV3 MLA prefill 路径修正 — 不走 FIA，走 RINGMLAPrefillBF16Kernel
  - §4.8.7: RINGMLAPrefillBF16Kernel 和 FIA 同类问题 — `seqlen` 参数（CPU tensor）控制实际计算量，CSV 不记录值，同 shape 下 duration 差 3.5 倍，需要 enriched CSV
- **§6**: 数据库构建方案全面更新
  - §6.2: 计算算子 microbench 从脚本生成模式（`generate_microbench.py`）改为 **op-replay 框架**（`op_replay/` 26 个算子脚本 + `start_microbench.py` 编排 + `msprof` 自动聚合回写）
  - §6.3: 通信 microbench 描述保持不变（详细设计见独立文档 COMM_BENCH_GUIDE.md）
  - §6.6: `discover_operators.py` 替换为 `compute_m6.py` 端到端验证
  - §6.7: 新增工具链总览表
  - 删除对已移除工具的引用（`generate_microbench.py`、`discover_operators.py`、`build_database.py`）
- **§7.3**: Shape 网格策略重新设计
  - 旧: 从 HuggingFace 模型配置提取维度 + powers-of-2 补充
  - 新: **理论导向的模型无关网格** — 固定维度覆盖常见 LLM 架构值的离散集，变化维度按算子性能曲线特征采样（GEMM 用 tiling-aware，Elementwise 用稀疏均匀，Attention 用 sqrt 空间）
  - 与 InterpolatingDataSource 的协同设计：网格保证 bracket 覆盖，插值补齐中间值

### 新增

- **附录 I**: 理论导向 Shape Grid 详细设计
  - I.1: 算子维度分类（固定 vs 变化维度）
  - I.2: 固定维度网格设计（NK_GRID, HEADS_GRID 等，含主流模型覆盖验证表）
  - I.3: 变化维度按算子类型的采样策略（GEMM/Elementwise/Attention/MOE/Comm）
  - I.4: Shape 总量预估（~70-170K/设备，参考 AI Configurator H100 ~65K）
  - I.5: 与当前 template mutation 方案的对比
  - I.6: `generate_shape_grid.py` 改造路径（`--mode theory` 新增模式）

### 修正

- **§9 Shape 对齐**: MLA composite 描述更新
  - 旧: "1:N 映射（TC → TransposeBatchMatMul + FIA）"
  - 新: "decode = TransposeBMM + FIA + TransposeBMM，prefill = MatMulV2 + RINGMLAPrefillBF16Kernel"
- **§9 Shape 对齐**: FusedAttention 描述更新
  - 旧: "通过 attention_special 模式处理"
  - 新: "通过 enriched CSV + attention_special 模式处理（v1.5 重设计）"

### 架构决策

1. **Enriched CSV 统一格式**: 不维护两套 CSV schema（raw + structured）。E4 microbench 输出 raw 31 slots + avg_seq_len 列，查询代码只需一条路径。决策原因：结构化 CSV 的维度（batch_size, num_heads, head_dim）均可从 raw CSV slot 0 Q shape 提取，唯一缺失的 avg_seq_len 通过新增列补充
2. **Profiling FIA 数据仅做验证**: 不参与查询和插值（avg_seq_len 未知 → duration 不可信），仅在 E4 microbench 数据到位后做 sanity check（同 workload 误差 <20%）
3. **`_lookup_attention()` 直接覆盖**: 删除旧结构化 CSV 路径（`required_cols = {"batch_size", "avg_seq_len", ...}` 检查），不做兼容。决策原因：两套路径增加维护成本，且结构化 CSV 格式不会再使用
4. **MLA decomposer prefill 分支修正**: FIA → RINGMLAPrefillBF16Kernel。Decomposer 硬编码 kernel type，op_mapping `sub_kernels` 列出并集做文档参考。决策原因：短期不改 `_lookup_composite` 读取逻辑，最小化代码变更
5. **FIA 和 RINGMLAPrefillBF16Kernel 是同类问题**: 两者都有"tensor 值控制计算量但 CSV 不记录值"的问题。FIA 的 `actual_seq_lengths_kv` 和 RING kernel 的 `seqlen` 本质相同——都是 CPU tensor 包含 per-request 实际 token 数。标准 compute 路径（`_inputs_match`）无法处理，需要 enriched CSV + 专属匹配逻辑。源码验证：`torch_npu.atb.npu_ring_mla(seqlen=query_lens)` → `RingMlaAtb.cpp` → ATB `ring_mla_operation.cpp` → `RINGMLAPrefillTiling(totalTaskNum=sum(qSeqLen))`
6. **op_mapping.yaml 保持单一职责**: Shape 推导逻辑放在 `generate_shape_grid.py` 工具中（读取维度范围、按算子类型分派采样策略），op_mapping.yaml 仅做 TC op → NPU kernel 的 1:1 名字映射。决策原因：职责分离——映射、采样、插值三者解耦
7. **理论导向 Shape Grid**: 从"在已知 profiling 模板附近随机扰动"改为"基于 LLM 架构空间定义模型无关网格"。固定维度（N, K, heads）要求精确匹配不插值，因此网格需覆盖所有目标模型的实际值；变化维度（M, seq_len）支持插值，网格只需覆盖关键 bracket 点。参考 AI Configurator 的实践：GEMM/Attention 模型无关、MOE 模型列表

---

## 工作计划变更 (v3.2 → v3.3)

### 任务更新

| 任务 | 变更 |
|------|------|
| E4 | 输出改为 enriched CSV 格式（raw + avg_seq_len 列）；不可降优先级（avg_seq_len 唯一来源）；deadline 3.20 → 3.24；验收标准增加 Qwen3 + DSV3 MLA head 配置覆盖 |
| G2 | Scope 变更：(1) prefill FIA → RINGMLAPrefillBF16Kernel (2) `_lookup_attention()` 覆盖为 enriched CSV (3) Q shape 归一化 CANN 8.3/8.5；deadline 3.20 → 3.23；验收增加 Qwen3 E2E FIA 命中 |

### Phase 2 优先级调整

| # | 旧描述 | 新描述 |
|---|--------|--------|
| P0-2 | FIA microbench CSV 格式 (Qwen3 ~9%) — ZZY | FIA lookup 重设计 — ZH 查询侧 + TCX 数据侧。enriched CSV + `_lookup_attention()` 覆盖 |
| P0-3 | MLA/MLAPO composite shape 修复 (DSv3 ~3-5%) — ZH | MLA composite 修正 — prefill FIA → RINGMLAPrefillBF16Kernel |

### 新增文档

| 文档 | 说明 |
|------|------|
| `reports/FIA_TODO_20260322.md` | 9 条研究结论 + 与 ZH 报告差异 + ZH/TCX/HXW 任务分配 + 注意事项 + 时间线 |
| `reports/FIA_KV_DIMENSION_ANALYSIS_v0.1.md` | ZH: FIA KV 维度分析（编译路径追溯 + profiling 验证） |
| `reports/FIA_LOOKUP_AND_MLA_RESEARCH_20260321.md` | ZH: FIA lookup + MLA 分解研究（3 种方案对比） |
