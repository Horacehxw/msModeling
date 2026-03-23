# 通信查询接口设计文档

**版本**：v1.0
**作者**：ZH
**日期**：2026-03-11
**关联文档**：`OPERATOR_PERF_DATABASE_DESIGN_zh_v1.2.md` §4.2/§4.7、`COMM_DATA_SPEC.md`
**关联任务**：B1（已完成）、B2（进行中）

---

## 1. 概述

通信查询是 `ProfilingDataSource` 四条查询路径之一（compute / comm / attention / composite）。当 `op_mapping.yaml` 中某算子标记 `category: communication` 时，`lookup()` 分派到 `_lookup_comm()`。

查询维度：`(message_bytes, num_devices, topology_tier)`，精确匹配通信 CSV。未命中时 fallback 到 `CommAnalyticModel`。

---

## 2. 接口定义

### 2.1 查询入口

```python
# profiling_data_source.py
def _lookup_comm(
    self, op_invoke_info: "OpInvokeInfo", mapping: dict
) -> Optional[QueryResult]:
```

**输入**：
- `op_invoke_info`：Runtime 拦截的算子调用信息，含 `func`、`args`
- `mapping`：来自 `op_mapping.yaml` 的该算子配置，至少含 `kernel_type`

**输出**：
- 命中 → `QueryResult(latency_us=float)`
- 未命中 → `None`（`self.last_miss_reason` 记录原因）

### 2.2 OpInvokeInfo args 布局

设计文档 §4.7 定义，所有 TC 通信算子 rank_group 均为最后一个 arg，rank 为倒数第二：

| Op | args[0] | args[1] | args[2] | args[3] | args[4] |
|----|---------|---------|---------|---------|---------|
| `all_reduce` | Tensor x | int rank | List rank_group | | |
| `all_gather` | Tensor x | int dim | int rank | List rank_group | |
| `reduce_scatter` | Tensor x | int dim | int rank | List rank_group | |
| `all_to_all` | Tensor x | List out_splits | List in_splits | int rank | List rank_group |

> 实现中统一用 `args[-1]` 取 rank_group，`args[-2]` 取 rank，无需按算子类型分支。

---

## 3. 查询逻辑

### 3.1 流程

```
_lookup_comm(op_invoke_info, mapping)
  │
  ├─ 1. 取 kernel_type（来自 mapping）
  │      未配置 → miss: "unmapped"
  │
  ├─ 2. 加载 CSV（_load_csv(kernel_type)）
  │      文件不存在 → miss: "csv_not_found"
  │
  ├─ 3. 检查 CSV 格式（required_cols = {message_bytes, num_devices}）
  │      缺列 → miss: "csv_format_raw"（原始 profiling CSV，非 microbenchmark 格式）
  │
  ├─ 4. 计算 message_bytes = args[0].nelement() * args[0].element_size()
  │
  ├─ 5. 取 rank_group = args[-1]，num_devices = len(rank_group)
  │
  ├─ 6. 解析 topology_tier（_resolve_topology_tier(rank_group)）
  │      comm_grid 为 None → topology_tier = None
  │
  ├─ 7. 构建匹配 mask
  │      基础：message_bytes == x AND num_devices == x
  │      若 topology_tier 非 None 且 CSV 有 topology_tier 列：追加 topology_tier == x
  │
  └─ 8. 查找匹配行
         无匹配 → miss: "shape_mismatch"
         有匹配 → QueryResult(latency_us=matched.iloc[0]["Duration(us)"])
```

### 3.2 topology_tier 解析

```python
def _resolve_topology_tier(self, group: list) -> Optional[int]:
    """从 rank_group 通过 CommGrid 推导 topology_tier。

    返回 diff_dim（第一个有差异的维度索引），与 CommAnalyticModel 镜像。
    comm_grid 为 None 时返回 None（降级为两字段匹配）。
    """
```

tier 语义（ATLAS_800_A3 三维网格 [48, 8, 2]）：

| tier | 名称 | 含义 | 典型场景 |
|------|------|------|---------|
| 0 | inter_pod | 跨 pod | DSV3 EP all_to_all，32 卡 |
| 1 | intra_pod | pod 内跨 node | Qwen3 TP=16，16 卡 |
| 2 | die_level | 同 node 内 | DSV3 TP=4，4 卡 |

### 3.3 降级策略

| 条件 | 行为 |
|------|------|
| `comm_grid` 为 None | 不过滤 topology_tier，仅用 message_bytes + num_devices 匹配 |
| CSV 无 `topology_tier` 列 | 同上（兼容旧格式 CSV） |
| topology_tier 解析失败 | 同上（不抛异常） |
| 整体未命中 | return None → EmpiricalPerformanceModel fallback 到 CommAnalyticModel |

---

## 4. CSV 格式约定

### 4.1 列定义

| 列名 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `message_bytes` | int | 是 | 单设备发送/接收字节数 |
| `num_devices` | int | 是 | 参与通信的设备数 |
| `topology_tier` | int | 推荐 | 拓扑层级（0/1/2），整数类型 |
| `Duration(us)` | float | 是 | 平均耗时（微秒） |
| `dtype` | str | 可选 | 当前不参与匹配，仅供参考 |
| `bandwidth_gbps` | float | 可选 | 实测带宽，用于验证 |

> 列名大小写敏感：`Duration(us)` 不能写成 `duration_us` 或 `Duration_us`。

### 4.2 文件命名

文件名必须与 `op_mapping.yaml` 中 `kernel_type` 字段完全一致：

| TC Op | kernel_type（文件名） |
|-------|----------------------|
| `tensor_cast.all_reduce.default` | `hcom_allReduce_.csv`（末尾有下划线） |
| `tensor_cast.all_gather.default` | `hcom_allGather_.csv` |
| `tensor_cast.reduce_scatter.default` | `hcom_reduceScatter_.csv` |
| `tensor_cast.all_to_all.default` | `hcom_alltoallv_.csv` |

文件路径：`{data_dir}/{kernel_type}.csv`，`data_dir` 由 `op_mapping.yaml` 的 `communication_data_ref` 字段指定。

### 4.3 dtype 字符串对齐问题

C9 采集脚本输出 `DT_BF16 / DT_FP16 / DT_FLOAT / DT_INT8`，而 `DTYPE_MAP` 映射为：

| torch dtype | DTYPE_MAP 输出 | C9 脚本输出 | 是否一致 |
|-------------|---------------|------------|---------|
| `torch.bfloat16` | `DT_BF16` | `DT_BF16` | ✓ |
| `torch.float16` | `DT_BF16` | `DT_FP16` | ✗ |
| `torch.int8` | `INT8` | `DT_INT8` | ✗ |
| `torch.float32` | `FLOAT` | `DT_FLOAT` | ✗ |

**当前影响**：`_lookup_comm` 不按 dtype 过滤，不影响命中。若后续加 dtype 过滤，需在 `_lookup_comm` 内做归一化，或要求 C9 脚本与 `DTYPE_MAP` 对齐。

---

## 5. op_mapping.yaml 配置

通信算子在 `operator_mappings` 中标记 `category: communication`：

```yaml
operator_mappings:
  "tensor_cast.all_reduce.default":
    kernel_type: hcom_allReduce_
    category: communication
  "tensor_cast.all_gather.default":
    kernel_type: hcom_allGather_
    category: communication
  "tensor_cast.reduce_scatter.default":
    kernel_type: hcom_reduceScatter_
    category: communication
  "tensor_cast.all_to_all.default":
    kernel_type: hcom_alltoallv_
    category: communication
```

通信数据路径通过顶层字段指定：

```yaml
communication_data_ref: "../../hccl/v8.1.RC1/"
communication_fallback: analytic
```

---

## 6. 目录结构

```
data/
└── ATLAS_800_A3_752T_128G_DIE/
    ├── vllm_ascend/v0.13.0/
    │   └── op_mapping.yaml          # communication_data_ref 指向 hccl 目录
    └── hccl/
        └── v8.1.RC1/
            ├── comm_config.yaml     # 拓扑描述
            ├── hcom_allReduce_.csv
            ├── hcom_allGather_.csv
            ├── hcom_reduceScatter_.csv
            └── hcom_alltoallv_.csv
```

通信数据与 vLLM 版本解耦，按 CANN 版本存储，可跨 vLLM 版本复用。

---

## 7. miss_reason 枚举

| miss_reason | 含义 | 处理建议 |
|-------------|------|---------|
| `unmapped` | op_mapping 中无该算子 | 补充 op_mapping.yaml |
| `csv_not_found` | kernel_type 对应 CSV 不存在 | 采集数据（C10） |
| `csv_format_raw` | CSV 是原始 profiling 格式，缺少 message_bytes/num_devices 列 | 用 microbenchmark 格式替换 |
| `invalid_args` | args[0] 不是 Tensor 或 rank_group 不是 list/tuple | 检查 TC op 注册 |
| `shape_mismatch` | CSV 中无匹配的 (message_bytes, num_devices, topology_tier) 组合 | 扩充采集矩阵 |

---

## 8. 待办与已知问题

| 项目 | 状态 | 说明 |
|------|------|------|
| B1 `_lookup_comm` topology_tier 匹配 | 已完成 | 43 个测试通过 |
| topology_tier 类型验证 | 待确认 | CSV 中必须是 int，pandas 读取默认 int64，应无问题，需跑测试验证 |
| dtype 过滤 | 暂不实现 | 当前不按 dtype 过滤；后续加时需处理 C9 脚本与 DTYPE_MAP 的不一致 |
| C10 HCCL 数据采集 | HDY 负责，3.13 截止 | CSV 到位后 _lookup_comm 可真正命中 |
