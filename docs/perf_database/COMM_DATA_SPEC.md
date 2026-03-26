# 通信算子数据表格式规范 (COMM_DATA_SPEC)

**版本**：v1.0
**作者**：ZH
**日期**：2026-03-10
**状态**：待 HXW review
**关联任务**：B1（ZH）、C9/C10（HDY）

---

## 1. 目录结构

通信数据独立于计算数据，按 CANN 版本存储（跨 vLLM 版本复用）：

```
tensor_cast/performance_model/profiling_database/data/
└── {device}/
    └── hccl/
        └── {cann_version}/
            ├── comm_config.yaml          # 拓扑描述 + 算子映射
            ├── hcom_allReduce_.csv       # AllReduce 数据
            ├── hcom_allGather_.csv       # AllGather 数据
            ├── HcomReduceScatter.csv     # ReduceScatter 数据
            └── hcom_alltoallv_.csv       # AllToAll 数据
```

**示例路径**（ATLAS_800_A3，CANN 8.1.RC1）：
```
data/ATLAS_800_A3_752T_128G_DIE/hccl/v8.1.RC1/
```

> 注：`op_mapping.yaml`（位于 `vllm_ascend/{version}/`）通过 `communication_data_ref` 字段指向此目录：
> ```yaml
> communication_data_ref: "../../hccl/v8.1.RC1/"
> ```

---

## 2. comm_config.yaml 格式

完整示例见 `docs/perf_database/examples/comm_config_example.yaml`。

**必填字段**：

```yaml
device: ATLAS_800_A3_752T_128G_DIE
cann_version: "8.1.RC1"
collection_date: "2026-03-10"

topology:
  grid_shape: [48, 8, 2]   # ATLAS_800_A3: [pod数, node/pod, die/node]
  tiers:
    0: {name: "inter_pod",  bandwidth_gbps: 196, latency_us: 5.5, type: "CLOS"}
    1: {name: "intra_pod",  bandwidth_gbps: 196, latency_us: 0.5, type: "CLOS"}
    2: {name: "die_level",  bandwidth_gbps: 224, latency_us: 0.2, type: "SIO"}

comm_operator_mappings:
  "tensor_cast.all_reduce.default":
    kernel_type: hcom_allReduce_
  "tensor_cast.all_gather.default":
    kernel_type: hcom_allGather_
  "tensor_cast.reduce_scatter.default":
    kernel_type: HcomReduceScatter
  "tensor_cast.all_to_all.default":
    kernel_type: hcom_alltoallv_
```

---

## 3. 通信 CSV 格式

### 3.1 列定义

| 列名 | 类型 | 说明 |
|------|------|------|
| `message_bytes` | int | 单设备发送/接收的数据量（字节）。AllReduce/AllGather/ReduceScatter 为 tensor 字节数；AllToAll 为单设备发送总量 |
| `num_devices` | int | 参与通信的设备数（= `len(rank_group)`） |
| `dtype` | str | 数据类型，见 §3.2 |
| `topology_tier` | int | 拓扑层级，见 §3.3 |
| `Duration(us)` | float | 平均耗时（微秒），warmup 后 N 次重复的均值 |
| `bandwidth_gbps` | float | 可选，实测带宽（GB/s），用于验证 |

**示例**（`hcom_allReduce_.csv`）：
```csv
message_bytes,num_devices,dtype,topology_tier,Duration(us),bandwidth_gbps
1048576,2,DT_BF16,2,45.3,22.1
1048576,8,DT_BF16,2,125.7,7.9
1048576,16,DT_BF16,1,198.4,5.0
1048576,16,DT_BF16,0,342.1,2.9
4194304,2,DT_BF16,2,156.2,25.6
4194304,8,DT_BF16,2,421.3,9.4
```

### 3.2 dtype 取值

| dtype 字符串 | 对应 torch dtype |
|-------------|----------------|
| `DT_BF16` | `torch.bfloat16` |
| `DT_FP16` | `torch.float16` |
| `INT8` | `torch.int8` |
| `DT_FP8` | `torch.float8_e4m3fn` |

**优先采集**：`DT_BF16`（Qwen3 BF16 场景）、`INT8`（DSV3 W8A8 场景）。

### 3.3 topology_tier 取值

ATLAS_800_A3 三维网格 `[48, 8, 2]`（pod × node/pod × die/node）：

| tier | 名称 | 含义 | 典型 num_devices |
|------|------|------|----------------|
| `2` | die_level | 同一 node 内 2 个 die 间（SIO，224 GB/s） | 2 |
| `1` | intra_pod | 同一 pod 内跨 node（CLOS，196 GB/s） | 2, 4, 8, 16 |
| `0` | inter_pod | 跨 pod（CLOS，196 GB/s，5.5µs latency） | 2, 4, 8, 16, 32, 64 |

**tier 判定规则**（与 `_get_topology_tier()` 一致）：
- rank_group 内所有 rank 在 grid 中的坐标，找到第一个有差异的维度 `diff_dim`
- `topology_tier = diff_dim`（0=inter_pod, 1=intra_pod, 2=die_level）

---

## 4. 采集矩阵

### 4.1 推荐采集范围

| 维度 | 取值 |
|------|------|
| message_bytes | 1KB, 4KB, 16KB, 64KB, 256KB, 1MB, 4MB, 16MB, 64MB, 256MB, 1GB（2x 递增） |
| num_devices | 2, 4, 8, 16, 32, 64（按 topology_tier 实际可用值） |
| dtype | DT_BF16（必须）、INT8（DSV3 场景必须）、DT_FP16（可选） |
| topology_tier | 0, 1, 2（各层级分别测） |

### 4.2 各算子优先级

| 算子 | CSV 文件 | 优先级 | 说明 |
|------|---------|--------|------|
| AllReduce | `hcom_allReduce_.csv` | P0 | Qwen3 Prefill 占比 89.8% |
| AllGather | `hcom_allGather_.csv` | P0 | DSV3 Decode 高频 |
| AllToAll | `hcom_alltoallv_.csv` | P1 | DSV3 MoE EP 场景 |
| ReduceScatter | `HcomReduceScatter.csv` | P1 | DSV3 Decode |

### 4.3 Qwen3-32B 实际通信参数（参考）

TP=16，BF16，hidden_size=7168：
- AllReduce message_bytes = `16 * 7168 * 2 bytes` = 229,376 bytes（BF16 每元素 2 字节）
- 实际 token 数影响 message_bytes，建议覆盖 128K ~ 16MB 范围

---

## 5. 采集方法

### 5.1 Python microbenchmark（主要方法）

使用 `tools/perf_data_collection/generate_comm_microbench.py` 生成脚本（HDY 任务 C9）。

核心逻辑：
```python
import torch.distributed as dist
import torch_npu

# 初始化 HCCL backend
dist.init_process_group(backend="hccl")

# 控制 topology_tier：通过 rank_group 参数
# tier=2 (die_level): rank_group=[0, 1]（同 node 内 2 个 die）
# tier=1 (intra_pod): rank_group=[0, 1, 2, 3, 4, 5, 6, 7]（同 pod 内 8 个 die）
# tier=0 (inter_pod): rank_group 跨 pod

group = dist.new_group(ranks=rank_group, backend="hccl")

# 计时：warmup 10 次 + 测量 100 次
for _ in range(warmup):
    dist.all_reduce(tensor, group=group)
torch.npu.synchronize()

start = torch.npu.Event(enable_timing=True)
end = torch.npu.Event(enable_timing=True)
start.record()
for _ in range(repeat):
    dist.all_reduce(tensor, group=group)
end.record()
torch.npu.synchronize()
duration_us = start.elapsed_time(end) * 1000 / repeat  # ms → us
```

### 5.2 HCCL Test（交叉验证）

```bash
# 位于 CANN toolkit
mpirun -n 8 ./bin/all_reduce_test -b 8K -e 2048M -f 2 -d fp16 -o sum -p 8
mpirun -n 8 ./bin/all_gather_test -b 8K -e 512M  -f 2 -d fp16 -p 8
mpirun -n 8 ./bin/all_to_all_test  -b 8K -e 256M  -f 2 -d fp16 -p 8
```

参考文档：https://www.hiascend.com/document/detail/zh/mindstudio/70RC1/mscommandtoolug/mscommandug/auxiliarydevtool_0017.html

Python benchmark 与 HCCL Test 偏差应 < 10%（H3 验收标准）。

---

## 6. 与 ProfilingDataSource 的接口

`_lookup_comm()` 查询逻辑（`profiling_data_source.py`）：

1. 从 `OpInvokeInfo.args[0]` 计算 `message_bytes`（tensor 字节数）
2. 从 `args[-1]`（rank_group）计算 `num_devices = len(rank_group)`
3. 若 `comm_grid` 可用，调用 `_get_topology_tier(comm_grid, rank_group)` 得到 `topology_tier`
4. 在 CSV 中精确匹配 `(message_bytes, num_devices, topology_tier)`
5. 未命中 → fallback 到 `CommAnalyticModel`

**CSV 必须包含的列**：`message_bytes`, `num_devices`, `topology_tier`, `Duration(us)`
**可选列**：`dtype`（当前版本不参与匹配，仅供参考）、`bandwidth_gbps`

> 注：当前 `_lookup_comm()` 不做 dtype 过滤，采集时建议每个 dtype 单独存文件或在同一文件中用 dtype 列区分。

---

## 7. 验收标准（C10）

- [ ] 4 种算子（AllReduce, AllGather, ReduceScatter, AllToAll）各有 CSV 文件
- [ ] 每个 CSV 覆盖 3 个 topology_tier（0/1/2）
- [ ] message_bytes 覆盖 1KB ~ 1GB（至少 8 个数量级点）
- [ ] DT_BF16 和 INT8 两种 dtype 均有数据
- [ ] `comm_config.yaml` 存在且格式正确
- [ ] Python benchmark 与 HCCL Test 偏差 < 10%（H3）
