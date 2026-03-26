# `fill_fia_runtime_metadata.py` 使用说明

本文档说明如何为 `FusedInferAttentionScore` 采集 runtime metadata，并通过
[`fill_fia_runtime_metadata.py`](G:\仿真开发\msmodeling\tools\perf_data_collection\fill_fia_runtime_metadata.py)
回填到
[`FusedInferAttentionScore.csv`](G:\仿真开发\msmodeling\tensor_cast\performance_model\perf_database\data\ATLAS_800_A3_752T_128G_DIE\vllm_ascend\v0.20.0\FusedInferAttentionScore.csv)。

适用场景：
- 已经通过 `parse_kernel_details.py` 生成了 FIA 初版 CSV
- 已经在 `vllm-ascend` 中埋点，拿到了 `fia_runtime_metadata.jsonl`
- 需要把真实的 `actual_seq_lengths`、`actual_seq_lengths_kv`、`block_table_valid_blocks` 回填到 CSV，供 `FusedInferAttentionScore_run.py` 直接 replay

---

## 1. 脚本作用

[`fill_fia_runtime_metadata.py`](G:\仿真开发\msmodeling\tools\perf_data_collection\fill_fia_runtime_metadata.py)
会做以下事情：

1. 读取 `FusedInferAttentionScore.csv`
2. 读取 `fia_runtime_metadata.jsonl`
3. 按 FIA signature 做精确匹配
4. 把 JSONL 中的真实 runtime 值回填到 CSV 的 `Runtime ...` 列

当前回填的主要列包括：
- `Runtime actual_seq_lengths_shape`
- `Runtime actual_seq_lengths_values`
- `Runtime actual_seq_lengths_kv_shape`
- `Runtime actual_seq_lengths_kv_values`
- `Runtime avg_seq_len`
- `Runtime block_table_valid_blocks`
- `Runtime metadata_completeness`

---

## 2. 为什么需要这个脚本

`parse_kernel_details.py` 只能从 profiling bundle 中恢复 shape、dtype、format 和部分 runtime 属性，但通常拿不到下面这些真实值：

- `actual_seq_lengths`
- `actual_seq_lengths_kv`
- `block_table` 每行实际使用的 block 数

这些值正是 FIA replay 是否能接近整网真实性能的关键。

因此流程应拆成两步：

1. 用 profiling 生成初版 FIA CSV
2. 用模型运行时埋点生成 JSONL，再回填到 CSV

---

## 3. 整体流程

完整流程如下：

1. 运行 `parse_kernel_details.py`，生成初版 `FusedInferAttentionScore.csv`
2. 在 `vllm-ascend` 的 FIA 调用点埋点
3. 跑和 profiling 同源的模型服务与 bench，生成 `fia_runtime_metadata.jsonl`
4. 运行 `fill_fia_runtime_metadata.py`
5. 再用 `FusedInferAttentionScore_run.py` 或 `start_microbench.py` 做 replay

---

## 4. 在 `vllm-ascend` 哪里埋点

最推荐的埋点位置是 FIA 的实际调用点，也就是调用：

```python
torch_npu.npu_fused_infer_attention_score(...)
```

或：

```python
torch_npu.npu_fused_infer_attention_score.out(...)
```

的前一行。

### 4.1 标准 FIA 路径

推荐位置：
[attention_v1.py](G:\仿真开发\vllm-ascend\vllm_ascend\attention\attention_v1.py)

适合采集：
- `query_shape`
- `key_shape`
- `value_shape`
- `atten_mask_shape`
- `block_table_shape`
- `actual_seq_lengths`
- `actual_seq_lengths_kv`
- `block_table_valid_blocks`
- `num_heads`
- `num_key_value_heads`
- `input_layout`
- `sparse_mode`
- `block_size`

### 4.2 MLA FIA 路径

推荐位置：
[mla_v1.py](G:\仿真开发\vllm-ascend\vllm_ascend\attention\mla_v1.py)

适合采集：
- `q_nope_shape`
- `k_nope_shape`
- `q_pe_shape`
- `k_pe_shape`
- `block_table_shape`
- `actual_seq_lengths`
- `actual_seq_lengths_kv`
- `block_table_valid_blocks`
- `num_heads`
- `num_key_value_heads`
- `input_layout`
- `sparse_mode`
- `block_size`

如果你当前 FIA CSV 的 `Runtime input_layout = TND`，优先看 `attention_v1.py`。  
如果是 `BNSD_NBSD` 或 `mla_paged_runtime`，优先看 `mla_v1.py`。

---

## 5. 埋点输出格式

推荐输出文件：

```text
fia_runtime_metadata.jsonl
```

格式要求：
- UTF-8 编码
- 每行一条 JSON
- 每条记录对应一次 FIA 调用
- 字段名固定，便于后处理

### 5.1 推荐字段

最少建议包含这些字段：

```json
{
  "op_name": "npu_fused_infer_attention_score",
  "timestamp_ns": 1774356338436082859,
  "query_shape": [128, 4, 128],
  "key_shape": [12005, 128, 128],
  "value_shape": [12005, 128, 128],
  "atten_mask_shape": [2048, 2048],
  "block_table_shape": [1, 512],
  "actual_seq_lengths": [128],
  "actual_seq_lengths_kv": [4104],
  "block_table_valid_blocks": [33],
  "num_heads": 4,
  "num_key_value_heads": 1,
  "input_layout": "TND",
  "sparse_mode": 3,
  "block_size": 128,
  "scale": 0.08838834764831845
}
```

### 5.2 字段说明

- `query_shape`
  FIA 的 query 实际 shape
- `key_shape`
  FIA 的 key 实际 shape
- `value_shape`
  FIA 的 value 实际 shape
- `atten_mask_shape`
  mask shape，缺失时可写 `null`
- `block_table_shape`
  paged attention 场景的 block table shape，非 paged 可写 `null`
- `actual_seq_lengths`
  query 侧真实长度列表
- `actual_seq_lengths_kv`
  KV 侧真实长度列表
- `block_table_valid_blocks`
  每条 request 实际使用的 block 数
- `num_heads`
  注意力头数
- `num_key_value_heads`
  KV 头数
- `input_layout`
  FIA 调用时实际 layout，如 `TND`、`BNSD_NBSD`
- `sparse_mode`
  FIA sparse mode
- `block_size`
  paged KV block size

### 5.3 `block_table_valid_blocks` 怎么算

如果运行时已经能直接拿到每条 request 的有效 block 数，直接记录最稳。

如果没有，可按下面方式推导：

```python
valid_blocks = [(seq_len + block_size - 1) // block_size for seq_len in actual_seq_lengths_kv]
```

---

## 6. 埋点代码模板

下面是一版可以直接贴进 `vllm-ascend` 的最小模板。

### 6.1 公共 helper

```python
import json
import os
import time


def _dump_fia_runtime_metadata(payload):
    dump_path = os.environ.get("MSMODELING_FIA_DUMP_PATH")
    if not dump_path:
        return

    with open(dump_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _shape(x):
    return list(x.shape) if x is not None else None


def _to_int_list(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    if hasattr(x, "tolist"):
        return [int(v) for v in x.tolist()]
    return None


def _valid_blocks(seq_lens, block_size):
    if not seq_lens or not block_size:
        return None
    return [(int(v) + int(block_size) - 1) // int(block_size) for v in seq_lens]
```

### 6.2 `attention_v1.py` 模板

在 `torch_npu.npu_fused_infer_attention_score(...)` 调用前插入：

```python
actual_seq_lengths_list = _to_int_list(actual_seq_lengths_q)
actual_seq_lengths_kv_list = _to_int_list(actual_seq_lengths_kv)

_dump_fia_runtime_metadata({
    "op_name": "npu_fused_infer_attention_score",
    "attn_variant": "standard",
    "source_file": "attention_v1.py",
    "timestamp_ns": time.time_ns(),
    "query_shape": _shape(query),
    "key_shape": _shape(key),
    "value_shape": _shape(value),
    "atten_mask_shape": _shape(attn_metadata.attn_mask),
    "block_table_shape": _shape(block_table),
    "actual_seq_lengths": actual_seq_lengths_list,
    "actual_seq_lengths_kv": actual_seq_lengths_kv_list,
    "block_table_valid_blocks": _valid_blocks(actual_seq_lengths_kv_list, block_size),
    "num_heads": int(self.num_heads),
    "num_key_value_heads": int(self.num_kv_heads),
    "input_layout": "TND",
    "sparse_mode": 3,
    "block_size": int(block_size) if block_size is not None else None,
    "scale": float(self.scale),
})
```

### 6.3 `mla_v1.py` 模板

在 `torch_npu.npu_fused_infer_attention_score(...)` 调用前插入：

```python
actual_seq_lengths_list = _to_int_list(actual_seq_lengths)
actual_seq_lengths_kv_list = _to_int_list(decode_meta.seq_lens_list)

_dump_fia_runtime_metadata({
    "op_name": "npu_fused_infer_attention_score",
    "attn_variant": "mla",
    "source_file": "mla_v1.py",
    "timestamp_ns": time.time_ns(),
    "query_shape": _shape(q_nope),
    "key_shape": _shape(k_nope),
    "value_shape": _shape(k_nope),
    "atten_mask_shape": _shape(attn_mask),
    "block_table_shape": _shape(decode_meta.block_table),
    "actual_seq_lengths": actual_seq_lengths_list,
    "actual_seq_lengths_kv": actual_seq_lengths_kv_list,
    "block_table_valid_blocks": _valid_blocks(actual_seq_lengths_kv_list, block_size),
    "num_heads": int(self.num_heads),
    "num_key_value_heads": int(self.num_kv_heads),
    "input_layout": input_layout,
    "sparse_mode": int(sparse_mode),
    "block_size": int(block_size) if block_size is not None else None,
    "scale": float(self.scale),
})
```

---

## 7. 运行时环境变量

埋点通过环境变量控制输出路径。

例如：

```bash
export MSMODELING_FIA_DUMP_PATH=/tmp/fia_runtime_metadata.jsonl
```

如果不设置这个变量，helper 会直接返回，不写任何文件。

Windows PowerShell 示例：

```powershell
$env:MSMODELING_FIA_DUMP_PATH='G:\仿真开发\FIA\fia_runtime_metadata.jsonl'
```

---

## 8. 如何保证 JSONL 和 CSV 是同源的

这是最关键的一点。

为了让 `fill_fia_runtime_metadata.py` 正确回填，JSONL 必须和 CSV 对应同一批运行条件：

- 同一个模型
- 同一套 vllm-ascend 代码
- 同一套 bench 参数
- 同一套输入输出长度
- 同一套并发和请求速率

建议操作方式：

1. 删除旧的 `fia_runtime_metadata.jsonl`
2. 设置 `MSMODELING_FIA_DUMP_PATH`
3. 拉起和 profiling 相同配置的 vLLM 服务
4. 用和 profiling 相同的 bench 参数发压
5. 结束后再运行 `fill_fia_runtime_metadata.py`

否则很容易出现：
- CSV 里是 `q=128,4,128`
- JSONL 里是 `q=64,4,128`
- 二者无法匹配

---

## 9. 回填命令

### 9.1 原地覆盖

```bash
py -3 tools/perf_data_collection/fill_fia_runtime_metadata.py \
  --csv-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/v0.20.0/FusedInferAttentionScore.csv \
  --jsonl-path G:\仿真开发\FIA\fia_runtime_metadata.jsonl
```

### 9.2 输出到新文件

```bash
py -3 tools/perf_data_collection/fill_fia_runtime_metadata.py \
  --csv-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/v0.20.0/FusedInferAttentionScore.csv \
  --jsonl-path G:\仿真开发\FIA\fia_runtime_metadata.jsonl \
  --output-path G:\仿真开发\msmodeling\FusedInferAttentionScore.backfilled.csv
```

### 9.3 可选参数

- `--metadata-tag`
  默认值是 `runtime_values_dumped`

例如：

```bash
py -3 tools/perf_data_collection/fill_fia_runtime_metadata.py \
  --csv-path ... \
  --jsonl-path ... \
  --metadata-tag runtime_values_dumped
```

---

## 10. 匹配规则

脚本按下面这些字段组成的 signature 做匹配：

- query shape
- key shape
- value shape
- block table shape
- `num_heads`
- `num_key_value_heads`
- `input_layout`
- `sparse_mode`
- `block_size`

如果 CSV 中 `block_size` 为空，而 JSONL 里只有唯一一个同 signature 的候选，脚本会自动放宽匹配并完成回填。

---

## 11. 回填后怎么检查

回填后重点检查这些列：

- `Runtime actual_seq_lengths_values`
- `Runtime actual_seq_lengths_kv_values`
- `Runtime avg_seq_len`
- `Runtime block_table_valid_blocks`
- `Runtime metadata_completeness`

正常情况下：

- `Runtime actual_seq_lengths_values` 不再为空
- `Runtime actual_seq_lengths_kv_values` 不再为空
- `Runtime avg_seq_len` 不再是默认 `4096`
- `Runtime metadata_completeness` 变为 `runtime_values_dumped`

---

## 12. 回填后如何使用

回填后的 CSV 可以直接给
[`FusedInferAttentionScore_run.py`](G:\仿真开发\msmodeling\tools\perf_data_collection\op_replay\FusedInferAttentionScore_run.py)
使用：

```bash
python tools/perf_data_collection/op_replay/FusedInferAttentionScore_run.py \
  --device ATLAS_800_A3_752T_128G_DIE \
  --vllm-ascend-version 0.20.0
```

也可以继续走：

```bash
python tools/perf_data_collection/start_microbench.py \
  --device ATLAS_800_A3_752T_128G_DIE \
  --vllm-ascend-version 0.20.0 \
  --op FusedInferAttentionScore
```

---

## 13. 常见问题

### 13.1 为什么脚本显示只回填了部分行

最常见原因：
- JSONL 和 CSV 不是同一次 bench/profiling 采出来的
- 埋点打到了 `attention_v1.py`，但 CSV 是 MLA 路径
- shape、layout、heads、block table shape 有一项对不上

### 13.2 为什么 `avg_seq_len` 还是 4096

说明该行没有被 JSONL 匹配上，或者对应记录里没有 `actual_seq_lengths_kv`。

### 13.3 为什么 `block_table_valid_blocks` 是空

可能原因：
- 该行是非 paged 场景
- 埋点里没有记录 `block_table_valid_blocks`
- `actual_seq_lengths_kv` 缺失，导致无法计算

---

## 14. 推荐实践

- 优先在 FIA 最终调用点埋点，不要只在上游 metadata 构造处埋点
- 埋点文件按单次任务清空，避免混入历史样本
- JSONL 中增加 `attn_variant` 和 `source_file` 字段，方便排查
- 同一次 profiling 和同一次 JSONL 采集尽量使用完全一致的 bench 参数
- 先用 `--output-path` 生成新文件确认效果，再决定是否覆盖原 CSV

