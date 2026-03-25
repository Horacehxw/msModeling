# FIA Runtime Metadata 埋点方案

本文档说明如何为 `FusedInferAttentionScore` 增加运行时埋点，以补齐当前 profiling bundle 无法提供、但 replay 对齐整网性能所必需的 runtime metadata。

适用目标：

- `torch_npu.npu_fused_infer_attention_score(...)`
- vLLM / vLLM-Ascend 中 FIA 调用链路
- `tools/perf_data_collection/parse_kernel_details.py`
- `tools/perf_data_collection/op_replay/FusedInferAttentionScore_run.py`

---

## 1. 背景

当前 `FusedInferAttentionScore.csv` 已经能从 profiling bundle 中恢复：

- `Input Shapes`
- `Input Data Types`
- `Input Formats`
- `Runtime input_layout`
- `Runtime sparse_mode`
- `Runtime block_size`
- `Runtime num_heads`
- `Runtime num_key_value_heads`
- `Runtime block_table_shape`
- `Runtime attn_state`
- `Runtime kv_cache_mode`

但仍然拿不到最关键的真实动态值：

- `actual_seq_lengths`
- `actual_seq_lengths_kv`
- `block_table` 的真实有效 block 分布

这导致 FIA replay 虽然可以“合法运行”，但仍然无法精确复刻整网中的真实运行状态。

---

## 2. 埋点目标

埋点的核心目标是补齐以下字段：

- `actual_seq_lengths`
- `actual_seq_lengths_kv`
- `block_table_shape`
- `block_table_valid_blocks`
- `num_heads`
- `num_key_value_heads`
- `sparse_mode`
- `input_layout`
- `block_size`
- `softmax_lse_flag`
- `pre_tokens`
- `next_tokens`

其中最重要的是：

1. `actual_seq_lengths_kv`
2. `block_table_valid_blocks`
3. `actual_seq_lengths`

---

## 3. 推荐埋点位置

### 3.1 第一优先级：FIA 实际调用点

最佳位置是调用：

```python
torch_npu.npu_fused_infer_attention_score(...)
```

之前的那一层。

优点：

- 记录的是最终传给 NPU 算子的真实值
- 不需要推断
- 最适合后续直接回填到 `FusedInferAttentionScore.csv`

缺点：

- 需要定位 vLLM / vLLM-Ascend 中实际 FIA 调用路径

### 3.2 第二优先级：attention metadata 构造处

如果 FIA 调用点不好改，可以在构造下列数据的位置埋点：

- `actual_seq_lengths`
- `actual_seq_lengths_kv`
- `block_table`
- `input_layout`
- `sparse_mode`

优点：

- 语义更直观

缺点：

- 不一定完全等于最终传给算子的值

### 3.3 第三优先级：paged KV cache / block manager

如果当前最关心的是 paged attention 的真实性，可以在：

- block table 生成处
- KV cache block 分配处

补充记录：

- `block_table_shape`
- `block_table_valid_blocks`
- `actual_seq_lengths_kv`

---

## 4. 推荐输出格式

不要把信息混在普通日志里，建议直接写 JSONL。

推荐文件名：

- `fia_runtime_metadata.jsonl`

一行表示一次 FIA 调用，格式示例：

```json
{
  "op_name": "npu_fused_infer_attention_score",
  "timestamp_ns": 0,
  "request_tag": "",
  "phase": "prefill|decode|chunked_prefill|unknown",
  "query_shape": [5, 16, 1, 512],
  "key_shape": [1135, 1, 128, 512],
  "value_shape": [1135, 1, 128, 512],
  "query_rope_shape": [5, 16, 1, 64],
  "key_rope_shape": [1135, 1, 128, 64],
  "atten_mask_shape": null,
  "actual_seq_lengths": [1, 1, 1, 1, 1],
  "actual_seq_lengths_kv": [4096, 4096, 4096, 4096, 4096],
  "block_table_shape": [5, 512],
  "block_table_valid_blocks": [32, 32, 32, 32, 32],
  "num_heads": 16,
  "num_key_value_heads": 1,
  "scale": 0.044194173,
  "input_layout": "BNSD_NBSD",
  "sparse_mode": 3,
  "block_size": 128,
  "softmax_lse_flag": true,
  "pre_tokens": 2147483647,
  "next_tokens": 2147483647
}
```

---

## 5. 字段说明

### `phase`

建议取值：

- `prefill`
- `decode`
- `chunked_prefill`
- `unknown`

作用：

- 帮助后续判断该 FIA 属于哪类推理阶段

### `query_shape` / `key_shape` / `value_shape`

作用：

- 作为与 `FusedInferAttentionScore.csv` 行匹配的主键之一

### `query_rope_shape` / `key_rope_shape`

作用：

- 用于区分 MLA rope 场景

### `actual_seq_lengths`

作用：

- 提供 query 侧真实长度信息

### `actual_seq_lengths_kv`

作用：

- 提供 KV 侧真实上下文长度
- 是最影响 replay 与整网对齐程度的字段之一

### `block_table_shape`

作用：

- 记录 block table 大小

### `block_table_valid_blocks`

作用：

- 记录每条 request 实际使用的 block 数
- 比单纯 shape 更接近真实 cache 使用情况

### `num_heads`

作用：

- query 侧 head 数

### `num_key_value_heads`

作用：

- KV 侧 head 数
- 用于区分 MHA / GQA / MQA 风格

### `input_layout`

作用：

- 记录如 `TND`、`BNSD_NBSD` 等布局信息

### `sparse_mode`

作用：

- 记录真实 sparse 路径

### `block_size`

作用：

- 表示 paged KV cache 每个 block 的 token 数

### `pre_tokens` / `next_tokens`

作用：

- 补齐当前 replay 里没有真实值的标量参数

---

## 6. 最小可用埋点集

如果只想做最小侵入实现，建议先记录这 8 个字段：

- `query_shape`
- `key_shape`
- `actual_seq_lengths`
- `actual_seq_lengths_kv`
- `block_table_shape`
- `block_table_valid_blocks`
- `input_layout`
- `sparse_mode`

这已经能显著提升 FIA replay 的真实性。

---

## 7. 代码模板

下面给一份最小可用的埋点 helper 模板。

### 7.1 Helper 函数

```python
import json
import os
import time
from pathlib import Path


def _shape_list(tensor):
    if tensor is None:
        return None
    try:
        return list(tensor.shape)
    except Exception:
        return None


def _to_int_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    try:
        return [int(x) for x in value.tolist()]
    except Exception:
        return None


def _infer_block_table_valid_blocks(block_table, block_size, actual_seq_lengths_kv):
    if block_table is None:
        return None
    if actual_seq_lengths_kv is not None and block_size:
        return [
            max(1, (int(seq_len) + int(block_size) - 1) // int(block_size))
            for seq_len in actual_seq_lengths_kv
        ]
    try:
        import torch
        valid_blocks = []
        block_table_cpu = block_table.detach().cpu()
        for row in block_table_cpu:
            valid_blocks.append(int(torch.count_nonzero(row >= 0).item()))
        return valid_blocks
    except Exception:
        return None


def dump_fia_runtime_metadata(
    *,
    query,
    key,
    value,
    query_rope=None,
    key_rope=None,
    atten_mask=None,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    block_table=None,
    num_heads=None,
    num_key_value_heads=None,
    scale=None,
    input_layout=None,
    sparse_mode=None,
    block_size=None,
    softmax_lse_flag=None,
    pre_tokens=None,
    next_tokens=None,
    phase="unknown",
    request_tag="",
):
    if os.getenv("MSMODELING_FIA_DUMP", "0") != "1":
        return

    output_path = os.getenv(
        "MSMODELING_FIA_DUMP_PATH",
        "/tmp/fia_runtime_metadata.jsonl",
    )

    actual_seq_lengths_list = _to_int_list(actual_seq_lengths)
    actual_seq_lengths_kv_list = _to_int_list(actual_seq_lengths_kv)
    block_table_valid_blocks = _infer_block_table_valid_blocks(
        block_table=block_table,
        block_size=block_size,
        actual_seq_lengths_kv=actual_seq_lengths_kv_list,
    )

    payload = {
        "op_name": "npu_fused_infer_attention_score",
        "timestamp_ns": time.time_ns(),
        "request_tag": request_tag,
        "phase": phase,
        "query_shape": _shape_list(query),
        "key_shape": _shape_list(key),
        "value_shape": _shape_list(value),
        "query_rope_shape": _shape_list(query_rope),
        "key_rope_shape": _shape_list(key_rope),
        "atten_mask_shape": _shape_list(atten_mask),
        "actual_seq_lengths": actual_seq_lengths_list,
        "actual_seq_lengths_kv": actual_seq_lengths_kv_list,
        "block_table_shape": _shape_list(block_table),
        "block_table_valid_blocks": block_table_valid_blocks,
        "num_heads": None if num_heads is None else int(num_heads),
        "num_key_value_heads": None if num_key_value_heads is None else int(num_key_value_heads),
        "scale": None if scale is None else float(scale),
        "input_layout": input_layout,
        "sparse_mode": None if sparse_mode is None else int(sparse_mode),
        "block_size": None if block_size is None else int(block_size),
        "softmax_lse_flag": None if softmax_lse_flag is None else bool(softmax_lse_flag),
        "pre_tokens": None if pre_tokens is None else int(pre_tokens),
        "next_tokens": None if next_tokens is None else int(next_tokens),
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
```

### 7.2 在 FIA 调用点接入

建议在实际调用 `torch_npu.npu_fused_infer_attention_score(...)` 前插入：

```python
dump_fia_runtime_metadata(
    query=query,
    key=key,
    value=value,
    query_rope=query_rope,
    key_rope=key_rope,
    atten_mask=atten_mask,
    actual_seq_lengths=actual_seq_lengths,
    actual_seq_lengths_kv=actual_seq_lengths_kv,
    block_table=block_table,
    num_heads=num_heads,
    num_key_value_heads=num_key_value_heads,
    scale=scale,
    input_layout=input_layout,
    sparse_mode=sparse_mode,
    block_size=block_size,
    softmax_lse_flag=softmax_lse_flag,
    pre_tokens=pre_tokens,
    next_tokens=next_tokens,
    phase="unknown",
)

output = torch_npu.npu_fused_infer_attention_score(...)
```

---

## 8. 环境变量控制

建议通过环境变量控制埋点开关：

```bash
export MSMODELING_FIA_DUMP=1
export MSMODELING_FIA_DUMP_PATH=/tmp/fia_runtime_metadata.jsonl
```

优点：

- 默认不影响正常运行
- profiling 时按需打开
- 不污染线上日志

---

## 9. 如何接回当前 perf database 流程

后续可以让 `parse_kernel_details.py` 额外读取 `fia_runtime_metadata.jsonl`，并按 shape 签名做匹配，回填到：

- `Runtime actual_seq_lengths_values`
- `Runtime actual_seq_lengths_kv_values`
- `Runtime block_table_valid_blocks`
- `Runtime num_heads`
- `Runtime num_key_value_heads`
- `Runtime sparse_mode`
- `Runtime input_layout`
- `Runtime block_size`

这样 `FusedInferAttentionScore_run.py` 就能优先使用真实 runtime metadata 做 replay。

---

## 10. 关于 bench 输入 4096、输出 1536 的作用

这些信息是有用的，但只是场景级先验，不是算子级 runtime 值。

它可以帮助：

- 判断当前 profiling 处于长 prompt + 长 decode 场景
- 为 `Runtime avg_seq_len` 提供合理默认值
- 作为日志校验信息

但它不能替代：

- `actual_seq_lengths_kv` 的真实逐 request 列表
- `block_table_valid_blocks`
- 某次具体 FIA 调用对应的真实动态上下文状态

---

## 11. 总结

当前 FIA replay 最缺的不是 shape，而是真实运行时动态长度和 block 映射。

最推荐的做法是：

1. 在 FIA 实际调用点埋点
2. 输出 JSONL 格式 runtime metadata
3. 让 `parse_kernel_details.py` 消费这些 metadata
4. 回填到 `FusedInferAttentionScore.csv`
5. 让 replay 直接使用这些真实值

只有这样，FIA microbench 才能真正接近整网实测表现。
