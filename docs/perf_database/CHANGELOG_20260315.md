# CHANGELOG 2026-03-15

设计文档 v1.3.1 → v1.4 + 工作计划 v3.1 → v3.2

基于 Phase 1 E2E 集成验证结果（4 场景，Qwen3-32B + DeepSeek-V3）更新。

---

## 设计文档变更 (v1.3.1 → v1.4)

### 新增

- **§4.2**: 通信 alpha-beta 模型插值（least-squares fit，精确匹配优先 + 插值 fallback，默认开启）
- **§4.2**: 3D→2D Flatten Batch 匹配规则 — quantize/norm 类 kernel 的 `(B,M,D)→(B*M,D)` 展平匹配，区别于通用 `_strip_batch_dim`（仅 B=1）
- **§4.9**: FRACTAL_NZ 恢复后权重转置匹配 — 对 `_MATMUL_KERNELS` 全集（含 `QuantBatchMatmulV3` 等量化变体）生效，不限于 ND 格式
- **§7.5**: M1-M5 五层评估指标体系 + 悲观规则 + 融合分组（Phase 1 E2E 建立）

### 变更

- **§4.2**: 通信查询策略从"精确匹配 `(num_devices, topology_tier)`"改为"精确匹配优先 + alpha-beta 插值 fallback"

### 架构决策

1. **通信插值默认开启**: `message_bytes` 为连续值，精确匹配命中率极低，alpha-beta 模型（`latency = α + β × message_bytes`）对线性 bandwidth-dominated 通信拟合准确
2. **Flatten Batch 为 kernel-specific 规则**: 仅对 quantize/norm 类 kernel 生效，不对 matmul 等算子生效。批次展平 `(B,M,D)→(B*M,D)` 仅在 element-wise/row-wise 算子上语义正确
3. **评估指标采用悲观规则**: 同一 op 任一 shape MISS → 整个 op 计 MISS。融合分组（DFC、MLAPO、MLA、MC2）反映 NPU 端实际 kernel 粒度

---

## 工作计划变更 (v3.1 → v3.2)

### Phase 1 完成状态

Phase 1 E2E 集成验证于 3.15 完成，GO/NO-GO: **GO**。

| 状态 | 任务 |
|------|------|
| ✅ 新完成 | B2 (composite lookup), C4 (Decode E2E 验证), D2 (attention lookup), D3 (InterpolatingDataSource), D4 (discover_operators) |
| ✅ 已完成 | A1-A3, B1, C1-C3, C6-C10, D1 |
| ⚠ 部分完成 | C5 (discover tool 存在，自动生成未实现), C11-1 (DFC 配置 composite 分解替代独立清单) |

### Phase 2 提前完成的任务

以下 Phase 2 任务在 Phase 1 中已提前完成:

| 任务 | 说明 |
|------|------|
| E1 generate_shape_grid.py | GEMM + attention shape 网格生成 |
| E2 generate_microbench.py | torch.mm + ATB kernel 脚本生成 |
| E5 Attention sqrt 插值 | InterpolatingDataSource 含 sqrt 变换 |
| G1 MoeGatingTopK | op + CSV + op_mapping 全链路 |

### Phase 2 新增任务 (E2E 发现)

| 任务 | 负责人 | 说明 |
|------|--------|------|
| P-E2E-1 RoPE dtype 宽松匹配 | 待定 | _triton_rope CSV FLOAT vs TC BF16，~10 行 |
| P-E2E-2 quantize/norm Microbenchmark 网格 | TCX | DSv3 M×D 缺失组合补充 |
| P-E2E-3 MoE routing 辅助 ops CSV | TCX | TopKV2, ReduceSum, Sigmoid 等 |

### Phase 2 优先级调整

基于 E2E 发现重新排序:
1. **P0**: DFC TC fusion pass — DSv3 ~40% 延迟 (LJW)
2. **P0**: FIA microbench CSV 格式 — Qwen3 ~9% 延迟 (ZZY)
3. **P0**: MLA/MLAPO composite shape 修复 — DSv3 ~3-5% (ZH)

### TC 主分支依赖 (新增)

| Issue | 描述 | 影响 |
|-------|------|------|
| add_rms_norm2 SP 维度 | TC 未对 seq dim 除以 TP | Qwen3 Prefill norm MISS |
| MLA output quantize shape | TC 3D per-head vs NPU 2D hidden | DSv3 W8A8 quantize MISS |

### 新增风险

| # | 风险 | 概率 | 缓解 |
|---|------|------|------|
| R13 | _triton_rope CSV dtype gap | 低 | Phase 2 dtype 宽松匹配 |
| R14 | TC 主分支 SP/MLA 修复依赖 | 中 | 已提 Issue |

### E2E 关键发现

- `_FLATTEN_BATCH_KERNELS` 规则已实现但当前不改变指标（需数据补充配合）
- `_ROPE_KERNELS` 已扩展 + normalize 支持 tc_input_count=2，但 dtype gap 阻止 HIT
- quantize `(8,16,128)` MISS 为 TC MLA output view/quantize 顺序问题，非 3D→2D 变换
- Qwen3 Prefill rms_norm/add_rms_norm2 MISS 为 SP 建模问题（M vs M/TP）
