# 算子性能数据库 Q1 工作计划

**目标**: 2026.3.23 完成端到端集成，DeepSeek-V3 / Qwen3-32B 仿真误差 <15%
**基准日期**: 2026.3.5
**设计文档**: `docs/perf_database/OPERATOR_PERF_DATABASE_DESIGN_zh_v1.2.md`
**穿刺总结**: `docs/perf_database/reports/spike_executive_summary_zh.md`

---

## 1. Spike 定位与代码评估

### 1.1 Spike 原则

Spike 的交付物是**知识和决策**。db-spike 已完成使命，验证了 DataSource 架构可行性（87% 匹配率）。后续新的技术穿刺产出 spec 和决策，**不合入产品分支**。

### 1.2 db-spike 代码质量评估

对 db-spike 的核心代码做了逐文件审查：

| 文件 | 行数 | 评估 | 说明 |
|------|------|------|------|
| `data_source.py` | 36 | Production-ready | 干净的 ABC, 无问题 |
| `profiling_data_source.py` | 465 | Production-ready | 无 debug print, 无 hack, docstring 引用设计文档, CSV 缓存, 完整错误降级 |
| `empirical.py` | 97 | Production-ready | 干净的 DataSource 委托 + fallback, 命中率统计 |
| `interpolating_data_source.py` | 20 | Phase 2 占位 | 设计正确, 一个 TODO |
| `__init__.py` | 11 | Production-ready | 干净导出 |
| **测试 (3 文件)** | **688** | **测试/实现 = 1.29:1** | 覆盖率 ~95%, 含 false-positive 防护测试 |
| `parse_kernel_details.py` | 342 | 完整可用 | CSV 解析 + 聚合 + 增量更新 |
| 其他 6 个 tools | 各 12 | Stub | `raise NotImplementedError` |

**结论：db-spike 代码可以直接作为产品分支基础**，不需要重写。17 项简化中大部分是设计决策（如 padding 容差、batch 剥离）而非代码质量问题，应在后续迭代中按优先级逐项升级。

---

## 2. 分支策略

### 2.1 方案：db-spike → feat/perf-database

```
develop (稳定主线)
  │
  └── feat/perf-database (从 db-spike 创建, 包含已验证的穿刺代码)
        │
        ├── XJT: feat/perf-db-cli
        ├── ZH:   feat/perf-db-datasource
        ├── TCX: feat/perf-db-toolchain
        ├── ZZY: feat/perf-db-op-mapping
        └── HDY: feat/perf-db-hccl

  └── db-spike (只读参考, 后续穿刺在独立分支, 不合入 feat)
```

**理由**：
- db-spike 是 develop 的超集（develop 无领先 commit），无合并风险
- 核心代码质量 8.5/10，重写无工程收益
- dev-tcx + feat/database 的 merge 内容本身应进 develop
- CSV 数据和 op_mapping.yaml 是数据资产，直接复用

### 2.2 操作步骤

**Step 1（3.5-3.6，HXW）**：
1. 从 db-spike 创建 `feat/perf-database` 分支推送到 gitcode
2. 移除穿刺实验文档（spike plan docs），保留设计文档和报告
3. 确认 `pytest tests/perf_database/ -v` 通过

**Step 2（3.6-3.7，各负责人）**：从 `feat/perf-database` 拉个人分支

**Step 3（3.23，HXW）**：`feat/perf-database` PR 回 `develop`

### 2.3 后续穿刺规则

新的技术穿刺（如 MC2 实际表现、vLLM op graph 直接抓取）在独立分支进行，产出 spec 和决策文档，**不合入 feat/perf-database**。

---

## 3. 团队分工

| 人员 | 投入 | 职责 | 代码 Owner |
|------|------|------|-----------|
| **ZH** | 全职 | ProfilingDataSource 扩展（Attention/通信/插值/MoE/MLA）+ 集成验证 | `perf_database/*.py` |
| **TCX** | 全职 | 数据采集工具链（6 个 stub → 完整实现）+ 集群数据采集 | `tools/perf_data_collection/` |
| **ZZY** | 全职 | op_mapping 系统化验证 + DSV3 Decode 映射 + 自动化方案 + Profiling 数据分析 | `op_mapping.yaml` + 验证报告 |
| **HDY** | 全职 | 通信 HCCL 数据采集 + 端到端验证 | HCCL 数据 |
| **XJT** | 全职 | CLI 集成 + 融合 Pass（MC2, KvRmsNormRopeCache） | `empirical.py`, CLI, Pass |
| **HXW** | SE | 出 spec → review PR → 新方向 spike → 进展管理 | 不 own 产品代码 |

### SE（HXW）的产出时间表

| 时间 | 产出 | 消费者 |
|------|------|--------|
| 3.5-3.6 | feat/perf-database 分支准备 | 全员 |
| 3.5-3.7 | Attention 匹配 spec（输入输出 + 匹配规则 + 测试用例） | ZH |
| 3.5-3.7 | 17 项简化清单产品化评估（逐项标注保留/改进/必须实现） | ZH |
| 3.10-3.12 | DSV3 op_mapping 草稿（基于穿刺经验 + AI 辅助生成） | ZZY验证 → ZH集成 |
| 3.14-3.16 | MoE/MLA 匹配 spec | ZH |
| 持续 | PR Review | 全员 |

---

## 4. Phase 1：核心集成（3.5 → 3.14）

**目标**：CLI 端到端可运行 + Attention/通信/插值三个 P0 查询路径 + op_mapping 完整验证 + 数据采集工具链基础。

### 任务 A：CLI 集成 + 端到端打通（XJT）

**目标**：`--performance-model profiling --compile` 端到端可运行。

db-spike 已有 `empirical.py`（97 行，production-ready）和 CLI 改动的参考。XJT在 `feat/perf-database` 上集成并确保端到端可运行。

| # | 检查点 | 预计完成 | 验证方式 |
|---|-------|---------|---------|
| A1 | CLI `--performance-model {analytic,profiling}` + `--perf-database` 路径参数 | 3.7 | analytic 行为不变；profiling 模式创建 EmpiricalPerformanceModel |
| A2 | 端到端：Qwen3-32B Prefill `--performance-model profiling --compile` | 3.10 | 不报错，log_stats 输出命中率 |
| A3 | 融合 Pass merge：SwiGlu + GroupedMatmul+SwiGlu 从 develop 合入 | 3.14 | 单元测试通过 + dispatch trace 对齐 |

---

### 任务 B：ProfilingDataSource 扩展（ZH）

**目标**：在已有的 `_lookup_compute()` 基础上，新增 Attention、通信、插值三条查询路径。

当前 `profiling_data_source.py` 的 `lookup()` 中，`attention_special` 和 `communication` 两个分支直接 return None（穿刺简化项 S-11/S-12）。需要实现这两条路径。

| # | 检查点 | 预计完成 | 验证方式 |
|---|-------|---------|---------|
| B1 | `_lookup_attention()`：从 OpInvokeInfo 提取 (seq_len, num_heads, head_dim)，匹配 FusedInferAttentionScore.csv | 3.10 | 单元测试：Qwen3 Prefill attention 命中 |
| B2 | `_lookup_comm()`：从 OpInvokeInfo 计算 message_bytes + topology_tier，查询通信 CSV | 3.12 | 单元测试：all_reduce/all_gather 返回耗时 |
| B3 | InterpolatingDataSource 实现：精确命中 → 直接返回，未命中 → 线性插值 | 3.14 | 单元测试：seq=136 精确命中；seq=200 插值 |

**SE 输入**：HXW 3.7 前交付 Attention 匹配 spec + 简化项评估。

---

### 任务 C：op_mapping 系统化验证 + DSV3 映射（ZZY，全职）

**目标**：系统性验证已有 op_mapping 60+ 条映射，补充 DSV3 Decode 场景映射，输出自动化方案 spec。

ZZY全职投入，承担 op_mapping 从验证到 DSV3 扩展的完整链条。

| # | 检查点 | 预计完成 | 验证方式 |
|---|-------|---------|---------|
| C1 | Profiling 算子清单：从 Qwen3-30B + DSV3 Decode 提取 (Type, 调用次数, 耗时占比) Top-20 排序表 | 3.7 | 表格输出 |
| C2 | TC dispatch trace 导出：用 analytic 模式跑 Qwen3-32B + DSV3 | 3.9 | dispatch trace 日志 |
| C3 | BF16 场景逐条映射验证：对照 C1/C2 验证 op_mapping 每条映射 | 3.11 | 验证报告（✅已验证/⚠️需注意/❌不匹配） |
| C4 | DSV3 Decode op_mapping 扩展：新增 W8A8 算子映射（QuantBatchMatmulV3、AscendQuantV2、DequantSwigluQuant、GroupedMatmul、TransposeBatchMatMul、MoeGatingTopK 等） | 3.13 | op_mapping 覆盖 DSV3 Top-15 算子 |
| C5 | W8A8 量化场景映射验证 | 3.14 | 验证报告 |
| C6 | op_mapping 自动化方案 spec：设计 Profiling → op_mapping 半自动生成流程 | 3.14 | spec 文档 |

**说明**：
- C4 的 DSV3 映射由HXW提供 AI 辅助生成的草稿（3.12 交付），ZZY负责验证和修正
- C2 可用 analytic 模式直接跑，不依赖XJT的 profiling CLI
- C6 是 Q2 投入方向，Q1 只需 spec

---

### 任务 D：数据采集工具链基础（TCX）

**目标**：完善 `parse_kernel_details.py` + 实现 `discover_operators.py` + `generate_shape_grid.py`。

当前 7 个工具中只有 `parse_kernel_details.py` 完整（342 行），其他 6 个是 stub。Phase 1 优先完成发现和 shape 网格工具。

| # | 检查点 | 预计完成 | 验证方式 |
|---|-------|---------|---------|
| D1 | `parse_kernel_details.py` 验证：在 Qwen3 + DSV3 数据上运行，确认输出正确 | 3.7 | 输出 CSV 与 db-spike 已有数据一致 |
| D2 | `discover_operators.py` 实现：对比 Profiling Type 列 vs op_mapping.yaml，输出覆盖率统计 | 3.10 | known 算子覆盖 >90% 调用次数 |
| D3 | `generate_shape_grid.py` 实现：从 HuggingFace 模型配置提取维度 + 2 的幂次网格 | 3.12 | Qwen3-32B + DSV3 shape 网格覆盖实际维度 |
| D4 | `generate_microbench.py` 实现：读 op_mapping.yaml 的 torch_npu_reference，生成 benchmark 脚本 | 3.14 | 生成的脚本语法正确 |

---

### 任务 E：通信 HCCL 数据采集（HDY）

| # | 检查点 | 预计完成 | 验证方式 |
|---|-------|---------|---------|
| E1 | `generate_comm_microbench.py` 实现：生成 torch.distributed benchmark 脚本 | 3.10 | 脚本可运行 |
| E2 | HCCL 数据采集（集群） | 3.14 | 4 种通信算子 × 各 topology_tier |
| E3 | HCCL Test 交叉验证 | 3.14 | Python benchmark 与 hccl_test 偏差 <10% |

---

### Phase 1 检查点（3.14）

| 交付物 | 验收标准 | 负责人 |
|-------|---------|--------|
| CLI `--performance-model profiling` | 端到端可运行 | XJT |
| `_lookup_attention()` | 单元测试通过 | ZH |
| `_lookup_comm()` | 单元测试通过 | ZH |
| InterpolatingDataSource | 插值可工作 | ZH |
| op_mapping 验证报告（BF16 + W8A8） | Qwen3 + DSV3 两场景 | ZZY |
| DSV3 op_mapping 扩展 | 覆盖 Top-15 算子 | ZZY |
| HCCL 数据 | 集群采集完成 | HDY |
| discover + shape_grid + microbench 工具 | 可运行 | TCX |
| 融合 Pass merge | 单元测试通过 | XJT |

### Phase 1 并行度

```
      3.5       3.7       3.9       3.11      3.13      3.14
       |         |         |         |         |         |
HXW |= spec ==>| review ==================================>|
       | 分支准备  | Attn spec|        | DSV3草稿 | MoE spec|
       | 简化评估  |         |         |         |         |
       |         |         |         |         |         |
XJT |--- A1 --|------ A2 ---------|         |--- A3 --|
       |         |         |         |         |         |
ZH   |         |--- B1 -----------|--- B2 --|--- B3 --|
       |         |         |         |         |         |
ZZY |--- C1 --|--- C2 --|--- C3 --|--- C4 --|C5+C6 --|
       |         |         |         |         |         |
TCX |--- D1 --|--- D2 --|--- D3 --|--- D4 --|         |
       |         |         |         |         |         |
HDY |--- E1 -----------|--- E2 ------------|-- E3 ---|
```

---

## 5. Phase 2：数据扩充 + 融合 Pass + DSV3 场景验证（3.14 → 3.20）

**目标**：Microbenchmark 数据扩充 + 融合 Pass 补齐 + DSV3 MoE/MLA 匹配 + mini 精度验证。

### 任务 F：Microbenchmark 数据采集（TCX）

| # | 检查点 | 预计完成 | 验证方式 |
|---|-------|---------|---------|
| F1 | 集群 Microbenchmark 采集 + `build_database.py` 实现 | 3.18 | 每个 kernel_type CSV 行数 > Profiling 原始 |
| F2 | FusedAttention Microbenchmark 特殊处理 | 3.20 | 构造 paged KV cache 输入可运行 |
| F3 | `validate.py` 实现 | 3.20 | 逐算子 + 端到端精度报告输出 |

### 任务 G：融合 Pass 补齐（XJT）

| # | 检查点 | 预计完成 | 验证方式 |
|---|-------|---------|---------|
| G1 | KvRmsNormRopeCache Pass | 3.17 | 单元测试 + dispatch trace |
| G2 | MC2 融合 Pass（P1 可选） | 3.20 | 如无法完成 → composite fallback |

### 任务 H：DSV3 MoE/MLA 匹配（ZH）

**SE 输入**：HXW 3.16 前交付 MoE/MLA 匹配 spec。

| # | 检查点 | 预计完成 | 验证方式 |
|---|-------|---------|---------|
| H1 | MoE 算子匹配：MoeGatingTopK, MoeDistributeDispatch/CombineV2, permute/unpermute_tokens | 3.18 | 单元测试 |
| H2 | MLA 分解查询：MLA → TransposeBatchMatMul + FIA 子查询 | 3.20 | 单元测试 |

### 任务 I：DSV3 Profiling 深度分析（ZZY，全职）

| # | 检查点 | 预计完成 | 验证方式 |
|---|-------|---------|---------|
| I1 | DSV3 Decode Profiling 逐层耗时分析：按 Transformer layer 拆解算子耗时分布 | 3.16 | 分析报告 |
| I2 | TC vs Profiling 算子对齐表：逐算子对比 TC dispatch trace 与 Profiling kernel 的 shape 差异 | 3.18 | 对齐表格，标注匹配/不匹配/原因 |
| I3 | 未覆盖算子分析：识别 Profiling 中有但 op_mapping 未覆盖的算子，评估耗时影响 | 3.20 | 缺口清单 + 优先级排序 |

### Phase 2 检查点（3.20）

| 交付物 | 验收标准 | 负责人 |
|-------|---------|--------|
| 扩充 CSV 数据库 | shape 覆盖 > Profiling 原始 | TCX |
| validate.py | 精度报告可输出 | TCX |
| 融合 Pass | KvRmsNormRopeCache 通过 | XJT |
| DSV3 MoE/MLA 匹配 | 单元测试通过 | ZH |
| DSV3 深度对齐分析 | 对齐表格 + 缺口清单 | ZZY |

---

## 6. Phase 3：端到端验证（3.20 → 3.23）

**目标**：端到端精度 <15%，交付精度报告。

### 任务 J：端到端精度验证（全员）

| # | 检查点 | 预计完成 | 负责人 | 验证方式 |
|---|-------|---------|--------|---------|
| J1 | Qwen3-32B 端到端验证（Prefill + Decode） | 3.22 | ZH + XJT | 误差 <15%, 覆盖 >90% |
| J2 | DSV3 端到端验证（Decode, MoE + MLA） | 3.22 | HDY + XJT | 误差 <15%, 覆盖 >90% |
| J3 | 精度问题定位 + 修复（如不达标） | 3.22 | ZZY分析 + ZH/TCX修复 | 补数据/修映射/调插值 |
| J4 | 精度总报告 | 3.23 | 全员 | 交付 |

**验证标准**：

| 指标 | 目标值 |
|-----|-------|
| 端到端耗时误差 | <15% |
| 单算子误差（已匹配） | <20% |
| 时间覆盖率 | >90% |

### 最终交付（3.23）

| 交付物 | 验收标准 |
|-------|---------|
| Qwen3-32B 精度报告 | 端到端误差 <15% |
| DeepSeek-V3 精度报告 | 端到端误差 <15% |
| 完整数据库（CSV + YAML） | 覆盖 Tier 1/2 算子 |
| validate.py | 可重复验证 |
| 数据采集工具链（7 个工具） | 可重复执行 |

---

## 7. 任务依赖图

```
          db-spike 已有代码 (feat/perf-database 基础)
                    │
    ┌───────────────┼───────────────┬───────────────┐
    ▼               ▼               ▼               ▼
 A1 CLI          B1 Attention    C1 算子清单     D1 解析验证
 (XJT)        (ZH)          (ZZY)       (TCX)
    │               │               │               │
    ▼               │               ▼               ▼
 A2 端到端 ◄────────┤          C2 Trace导出     D2 发现工具
 (XJT)           │          (ZZY)         (TCX)
    │               ▼               │               │
    │          B2 通信查询      C3 BF16验证      D3 Shape网格
    │          (ZH)          (ZZY)         (TCX)
    │               │               │               │
    │               ▼               ▼               ▼
    │          B3 插值         C4 DSV3映射     D4 Microbench
    │          (ZH)     ◄── (ZZY)  ◄── HXW草稿
    │               │               │
    └───────┬───────┘          C5/C6 验证+spec
            ▼
   Phase 1 交付 (3.14)
            │
    ┌───────┼───────┬───────┐
    ▼       ▼       ▼       ▼
  F1-F3   G1-G2   H1-H2   I1-I3
  数据    融合    MoE/MLA  DSV3分析
  (楚笑)  (锦涛)  (ZH)   (震宇)
    │       │       │       │
    └───────┼───────┘       │
            ▼               ▼
   Phase 2 交付 (3.20)  ◄──┘
            │
    ┌───────┼───────┐
    ▼       ▼       ▼
  J1 Qwen3 J2 DSV3 J3 修复(如需)
    │       │       ↑
    └───────┼───────┘ ZZY分析定位
            ▼
   J4 精度报告 (3.23)
```

**关键路径**：A1 → A2 → (B1 + C4) → H1 → J1 → J4

---

## 8. 风险与缓解

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|---------|
| 集群资源不足 | F1/E2 延迟 | 中 | 3.10 前预约；Phase 1 用现有 Profiling 数据 |
| MC2 融合 Pass 复杂 | G2 延期 | 中 | 降为 P1 可选，composite fallback |
| 精度 <15% 难达到 | J1/J2 调优 | 中 | Phase 2 mini 验证提前暴露；ZZY全职分析定位 |
| DSV3 MoE/MLA 映射复杂 | H1/H2 延期 | 中 | 穿刺已验证通用匹配逻辑，MoE 是增量 |
| Attention 匹配精度不足 | 端到端误差超标 | 低 | FIA.csv 已有 67 行数据，可补 Microbenchmark |

---

## 9. 新方向 Spike（HXW，时间盒 1-3 天）

后续穿刺产出 spec 和决策，不合入 feat/perf-database。

| 候选 Spike | 触发条件 | 时间盒 | 产出 |
|-----------|---------|-------|------|
| MC2 Profiling 实际表现 | Phase 2 G2 启动前 | 1 天 | MC2 是单 kernel 还是分离的？决定实现方案 |
| 直接从 vLLM 抓 op graph | Phase 3 后 | 2 天 | 绕过 TC dispatch 的可行性评估 |
| Attention 插值精度 | B3 前 | 1 天 | 不同 seq_length 下 FIA 插值误差评估 |

---

## 10. 每周同步

| 时间 | 形式 | 内容 |
|------|------|------|
| Phase 1（3.5-3.14） | 每日 15min 站会 | 昨日完成 / 今日计划 / 阻塞项 |
| Phase 2（3.14-3.20） | 隔日同步 | 进展 + 集群协调 |
| Phase 3（3.20-3.23） | 每日同步 | 精度快速响应 |
| 关键节点 | Review 会 | 3.14 / 3.20 / 3.23 |
