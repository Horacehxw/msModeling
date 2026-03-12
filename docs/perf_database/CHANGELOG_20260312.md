# CHANGELOG 2026-03-12

设计文档 v1.3 → v1.3.1 + 工作计划 v3 → v3.1

---

## 设计文档变更 (v1.3 → v1.3.1)

### 新增

- **§2.2**: CANN 8.5 DispatchFFNCombine 超级融合说明（35.3% DSV3 Decode）
- **§3.3**: 完整版本字符串目录命名约定（`vllm{ver}_torch{ver}_cann{ver}`）
- **§4.2**: Input 数量不匹配处理规则（QuantBatchMatmulV3 2→4, ReshapeAndCacheNdKernel 4→5, TensorMove 2→1）
- **§4.7**: HCCL v8.5 数据质量注意事项（JIT 初始化问题 + 单 session 修复方案）
- **§9.1**: DispatchFFNCombine gap 项（CANN 8.5 新增，35.3% DSV3）
- **§9.1**: TP Padding bug 已修复记录（59cf184）
- **§9.1**: MC2 已验证记录（3f82c2b，BF16 + W8A8）

### 变更

- **§1.4**: 目标后端 vllm-ascend 0.13.0 → 0.15.0（CANN 8.5，torch 2.9.0）
- **§3.3**: 数据存储路径更新（含 CANN 8.3 legacy + CANN 8.5 production target）
- **§4.2**: ProfilingDataSource 构造参数 `comm_grid` → `device_profile`（统一硬件参数访问）
- **§4.5**: op_mapping.yaml cann_version 更新为 8.5，communication_data_ref 更新
- **§9.1**: KvRmsNormRopeCache 状态 "仍开放" → "已关闭"（mlapo 已覆盖）
- **§9.1**: MC2 融合状态 "仍开放" → "已验证"
- **§9.2**: MLA 分解状态更新（composite 查询已实现）

### 架构决策

1. **ProfilingDataSource 接口统一**：所有查询路径（计算 CSV + 通信 HCCL）通过 `device_profile` 对象访问，替代分散的 `comm_grid` 参数
2. **ModelRunnerMetrics 多模型支持**：`execution_time_s: float` → `Dict[str, float]`；新增 `tps_per_model: Dict[str, float]`
3. **Input 数量不匹配**：采用 kernel_type 级过滤规则（非通用截断），因 NPU kernel 实际接收的 input 可能包含 TC 层不可见的内部参数

---

## 工作计划变更 (v3 → v3.1)

### 人员变动

- **XJT → LJW**：XJT 工作交接给 LJW，LJW 3.17 起全职投入
- **XJT 已完成交付**：A1/A2/A3 + MC2 验证 + KvRmsNormRopeCache 确认

### 目标调整

- **E2E 验证版本**：CANN 8.3 → **CANN 8.5**（vllm 0.15.0 + torch 2.9.0）
- **数据目录**：`vllm0.15.0_torch2.9.0_cann8.5/`
- **数据采集工具链**：7 个 → 9 个

### Phase 1 进展（截至 3.12）

| 状态 | 任务 |
|------|------|
| ✅ 已完成 | A1, A2, A3, B1, C1, C2, C3, C6, C7, C8, C9, D1 |
| 🔄 部分完成 | B2(draft), C10(CSV 产出，tier=0 缺失) |
| 🔄 进行中 | C4, D2, D3, D4 |

**关键发现**：
- TC compile 路径不可行用于 op_mapping 验证（ZZY），改用 AI/skill 方案
- KvRmsNormRopeCache 被 mlapo 覆盖，F1 scope 缩减
- TP Padding bug 已修复（全局 input → MoE layer-local，消除 DSV3 Decode 8x 高估）
- Qwen3 BF16 验证覆盖 97.01%，DSV3 W8A8 覆盖 98.02%

### 新增任务

- **C11**: DispatchFFNCombine 子内核数据采集（HDY, C11-1 截止 3.13, C11-2 截止 3.18, C11-3 截止 3.19）
- **F1 scope 变更**: ~~KvRmsNormRopeCache pass~~ → DispatchFFNCombine 可行性评估（LJW）
- **F2**: 条件性 DispatchFFNCombine pass 实现

### 新增风险

| # | 风险 | 概率 |
|---|------|------|
| R10 | 单算子与整网算子耗时 gap（TCX 3.12 提出） | 高 |
| R11 | CANN 8.5 DispatchFFNCombine 覆盖不足（35.3% DSV3） | 高 |
| R12 | LJW 上手周期（2-3 天） | 中 |

### 已关闭风险

- **R9**（XJT被其他项目拖住）：A1/A2/A3 已完成，XJT→LJW 交接

### 核心假设新增

- CANN 8.5 DispatchFFNCombine 可通过 composite 分解查询覆盖（验证：C11 + Phase 2）
