# FusedInferAttentionScore CSV 参数映射说明

本文档整理 `FusedInferAttentionScore.csv` 中 `Input Shapes / Input Data Types / Input Formats` 与
`torch_npu.npu_fused_infer_attention_score` 接口参数的对应关系，并补充这份 CSV 的实际 case 分类。

适用对象：
- `tensor_cast/performance_model/profiling_database/data/.../FusedInferAttentionScore.csv`
- `tools/perf_data_collection/op_replay/FusedInferAttentionScore_run.py`

## 1. 结论概览

`FusedInferAttentionScore.csv` 里的输入不是“只记录了实际出现的参数”，而是按
`torch_npu.npu_fused_infer_attention_score` 的 tensor 参数顺序展开成固定槽位。

- 共 31 个 tensor 输入槽位
- 空字符串表示该参数在该行未传入
- 非 tensor 标量参数不在这 31 个槽位中，需要脚本额外推断或显式传入

当前分析主要依据四部分交叉确认：
- `torch_npu.npu_fused_infer_attention_score` 官方文档签名
- `FusedInferAttentionScore.csv` 各行非空槽位分布
- shape/dtype/format 的语义特征
- 仓内 attention / paged attention / MLA 相关代码

## 2. CSV 31 个输入槽位与接口参数的对应关系

| Index | 参数名 | 当前 CSV 是否出现 | 典型 Shape | 说明 |
|---|---|---:|---|---|
| 0 | `query` | 是 | `3072,4,128` / `16,4,128` / `4,16,1,512` | 主输入 Q |
| 1 | `key` | 是 | `12235,128,128` / `892,1,128,512` / `41040,1,128` | 主输入 K，paged 场景下是 KV cache 形态 |
| 2 | `value` | 是 | `12235,128,128` / `892,1,128,512` / `41040,1,128` | 主输入 V |
| 3 | `pse_shift` | 否 | - | 当前库里未出现 |
| 4 | `atten_mask` | 是 | `2048,2048` | attention mask |
| 5 | `actual_seq_lengths` | 是 | `2` / `11` / `16` | 实际 Q 长度列表，对 TND 常按累计长度理解 |
| 6 | `actual_seq_lengths_kv` | 是 | `2` / `11` / `4` / `5` | 实际 KV 长度列表 |
| 7 | `dequant_scale1` | 否 | - | 未出现 |
| 8 | `quant_scale1` | 否 | - | 未出现 |
| 9 | `dequant_scale2` | 否 | - | 未出现 |
| 10 | `quant_scale2` | 否 | - | 未出现 |
| 11 | `quant_offset2` | 否 | - | 未出现 |
| 12 | `antiquant_scale` | 否 | - | 未出现 |
| 13 | `antiquant_offset` | 否 | - | 未出现 |
| 14 | `block_table` | 是 | `2,512` / `129,512` / `4,512` / `5,512` | page attention 的 block 映射表 |
| 15 | `query_padding_size` | 否 | - | 未出现 |
| 16 | `kv_padding_size` | 否 | - | 未出现 |
| 17 | `key_antiquant_scale` | 否 | - | 未出现 |
| 18 | `key_antiquant_offset` | 否 | - | 未出现 |
| 19 | `value_antiquant_scale` | 否 | - | 未出现 |
| 20 | `value_antiquant_offset` | 否 | - | 未出现 |
| 21 | `key_shared_prefix` | 否 | - | 未出现 |
| 22 | `value_shared_prefix` | 否 | - | 未出现 |
| 23 | `actual_shared_prefix_len` | 否 | - | 未出现 |
| 24 | `query_rope` | 是 | `4,16,1,64` / `5,16,1,64` | MLA/rope 场景下的 query rope |
| 25 | `key_rope` | 是 | `892,1,128,64` | MLA/rope 场景下的 key rope |
| 26 | `key_rope_antiquant_scale` | 否 | - | 未出现 |

说明：
- 文档签名里 `*` 之后的参数均为 keyword 参数，但 CSV 仍按固定位置展开
- 当前库中只出现了前 27 个 tensor 参数中的部分槽位
- 标量参数如 `num_heads`、`scale`、`input_layout`、`sparse_mode` 不会出现在这张表里

## 3. 不在 31 个槽位中的标量参数

这些参数不在 CSV 的 `Input Shapes / Input Data Types / Input Formats` 31 槽位中，需要脚本额外推断或显式传入。

| 参数名 | 是否在 CSV 的 31 个输入槽位中 | 当前脚本如何确定 |
|---|---:|---|
| `num_heads` | 否 | 从 `query` shape 推断 |
| `scale` | 否 | 通常按 `1 / sqrt(head_dim)` 推断 |
| `pre_tokens` | 否 | 脚本固定传较大值 |
| `next_tokens` | 否 | 脚本固定传较大值 |
| `input_layout` | 否 | 根据 `query / key / query_rope` shape 模式推断 |
| `num_key_value_heads` | 否 | 根据 `key` shape 和场景推断 |
| `sparse_mode` | 否 | 根据 `atten_mask` 是否存在及形状推断 |
| `inner_precise` | 否 | 当前脚本未特别打开 |
| `block_size` | 否 | paged 场景下由 `key` shape 推断 |
| `antiquant_mode` | 否 | 当前未使用 |
| `softmax_lse_flag` | 否 | 按 `Output Shapes` 是否存在第二输出判断 |
| `key_antiquant_mode` | 否 | 当前未使用 |
| `value_antiquant_mode` | 否 | 当前未使用 |

## 4. 我们是如何确认这些槽位映射的

### 4.1 按接口签名顺序对齐

文档给出的签名如下：

```python
torch_npu.npu_fused_infer_attention_score(
    query,
    key,
    value,
    *,
    pse_shift=None,
    atten_mask=None,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    dequant_scale1=None,
    quant_scale1=None,
    dequant_scale2=None,
    quant_scale2=None,
    quant_offset2=None,
    antiquant_scale=None,
    antiquant_offset=None,
    block_table=None,
    query_padding_size=None,
    kv_padding_size=None,
    key_antiquant_scale=None,
    key_antiquant_offset=None,
    value_antiquant_scale=None,
    value_antiquant_offset=None,
    key_shared_prefix=None,
    value_shared_prefix=None,
    actual_shared_prefix_len=None,
    query_rope=None,
    key_rope=None,
    key_rope_antiquant_scale=None,
    num_heads=1,
    scale=1.0,
    pre_tokens=2147483647,
    next_tokens=2147483647,
    input_layout="BSH",
    num_key_value_heads=0,
    sparse_mode=0,
    inner_precise=0,
    block_size=0,
    antiquant_mode=0,
    softmax_lse_flag=False,
    key_antiquant_mode=0,
    value_antiquant_mode=0,
)
```

CSV 里的 31 个 tensor 槽位就是按这里的 tensor 参数顺序展开的。

### 4.2 用非空 index 反推具体参数

当前库中，常见行的非空 index 很稳定，例如：

- `0,1,2` 恒非空，对应 `query/key/value`
- `4` 常为 `2048,2048` 且 dtype 是 `INT8`，明显符合 `atten_mask`
- `5,6` 是单数字 shape 且 dtype 是 `INT64`，符合 `actual_seq_lengths / actual_seq_lengths_kv`
- `14` 是二维 `INT32`，如 `129,512`，符合 `block_table`
- `24,25` 仅在 MLA 样本中出现，且 dtype 是 `BF16`，符合 `query_rope / key_rope`

### 4.3 用 shape 语义做二次校验

- 普通行：`query` 常为 `(T, N, D)`，`key/value` 常为 `(block_num, block_size, D)` 或近似 page cache 布局，说明是 TND + page attention 路径
- MLA 行：`query` 为 `(B, N, S, D)`，同时有 `query_rope/key_rope`，说明是 MLA rope 形态
- `block_table` 第二维固定像 `max_blocks_per_seq`，和 page attention 文档一致

### 4.4 用仓内代码交叉验证

主要参考：
- `tensor_cast/ops/attention.py`
- `tensor_cast/ops/mla.py`
- `tensor_cast/core/input_generator.py`
- `tensor_cast/performance_model/__init__.py`

这些文件帮助确认了：
- `block_table` 的语义
- `query_lens / seq_lens` 与 TND/paged attention 的关系
- rope 与 MLA 相关 shape 的语义

## 5. 针对这份 CSV 的实际三种 case 分类

当前这份 `FusedInferAttentionScore.csv` 实际可分为三类。

### Case A: 普通 paged TND

这是当前库里数量最多的一类。

典型特征：
- `query`: 3D，形如 `(T, N, D)`，例如 `3072,4,128`
- `key/value`: 3D，形如 `(block_num, block_size, D)`，例如 `12235,128,128`
- `atten_mask`: 存在，通常是 `2048,2048`
- `actual_seq_lengths`: 存在
- `actual_seq_lengths_kv`: 存在
- `block_table`: 存在
- `query_rope/key_rope`: 不存在

对应槽位：
- 必填：`0,1,2`
- 常见非空：`4,5,6,14`

脚本侧推断：
- `input_layout = "TND"`
- `num_heads = query.shape[1]`
- `num_key_value_heads = 1`
- `block_size = key.shape[1]`
- `sparse_mode` 依据 `atten_mask` 形态推断

备注：
- 这类行的 `key/value` 首维更像整个 KV cache 池容量，而不是当前 batch 的真实上下文 block 数
- 仅凭 CSV 的 shape 无法恢复真实 `block_table` 内容和真实 `actual_seq_lengths_kv` 数值

### Case B: 非 paged TND

这类行数量较少，但仍是普通 attention，不带 rope。

典型特征：
- `query`: 3D，形如 `(T, N, D)`，例如 `41040,4,128`
- `key/value`: 3D，形如 `(T_kv, KV_N, D)`，例如 `41040,1,128`
- `atten_mask`: 存在
- `actual_seq_lengths`: 存在
- `actual_seq_lengths_kv`: 存在
- `block_table`: 不存在
- `query_rope/key_rope`: 不存在

对应槽位：
- 必填：`0,1,2`
- 常见非空：`4,5,6`

脚本侧推断：
- `input_layout = "TND"`
- `num_heads = query.shape[1]`
- `num_key_value_heads = key.shape[1]`
- `block_size = 0`

与 Case A 的关键区别：
- 没有 `block_table`
- `key/value` 的 shape 直接表达真实 KV 长度
- 不是 page attention

### Case C: MLA rope

这类行只出现少量样本，但结构最特殊。

典型特征：
- `query`: 4D，形如 `(B, N, S, D)`，例如 `4,16,1,512`
- `key/value`: 4D，形如 `(block_num, KV_N, block_size, D)`，例如 `892,1,128,512`
- `actual_seq_lengths`: 通常为空
- `actual_seq_lengths_kv`: 存在
- `block_table`: 存在
- `query_rope`: 存在，例如 `4,16,1,64`
- `key_rope`: 存在，例如 `892,1,128,64`

对应槽位：
- 必填：`0,1,2`
- 常见非空：`6,14,24,25`

脚本侧推断：
- `input_layout = "BNSD_NBSD"`
- `num_heads = query.shape[1]`
- `num_key_value_heads = 1`
- `block_size = key.shape[2]`
- `scale` 需结合 `query` 与 `query_rope` 的最后一维理解

与前两类的关键区别：
- `query` 是 4D 而不是 3D
- 带 `query_rope / key_rope`
- 明显是 MLA 相关路径，而不是普通 TND attention

## 6. 为什么“同样 shape”不等于“同样输入”

从 replay 角度，CSV 只完整记录了：
- shape
- dtype
- format

但没有记录真实运行时的：
- `query/key/value` 实际数值
- `actual_seq_lengths` 实际列表值
- `actual_seq_lengths_kv` 实际列表值
- `block_table` 实际映射内容
- `atten_mask` 实际内容
- 部分标量参数的真实取值

因此，即使 shape/dtype/format 一样，也不代表是“同样输入”。

这也是为什么 replay 脚本只能做到：
- 严格复原输入结构
- 近似复原部分标量参数
- 合法构造缺失的动态内容

但不能保证与原 profiling 完全一致。

## 7. 当前脚本中与 CSV 对齐时的特别注意点

### 7.1 `softmax_lse_flag`

不能根据 `Output Data Types` 是否有第二项来判断。

原因：
- 某些 CSV 行里 `Output Data Types` 会写成 `DT_BF16;FLOAT`
- 但 `Output Shapes` 实际只有一个输出 shape

因此更合理的判断方式是：
- 只有当 `Output Shapes` 中确实存在第二个 shape 时，才认为 `softmax_lse_flag=True`

### 7.2 paged attention 的 `actual_seq_lengths_kv`

对 paged 场景：
- `key/value` 的首维更像 cache 池大小
- 不能直接拿来当真实上下文 block 数

否则很容易构造出远大于 `block_table` 宽度的非法 `actual_seq_lengths_kv`。

### 7.3 `atten_mask` 与 `sparse_mode`

根据官方约束：
- 传入 `atten_mask` 时，`sparse_mode` 需要与 mask 形态匹配
- 对 `2048x2048` 优化 mask，通常需要走特定 sparse 路径

因此 `atten_mask` 是否存在、shape 是什么，会直接影响 `sparse_mode` 的推断。

## 8. 推荐使用方式

如果需要继续分析 `FusedInferAttentionScore.csv`，建议按以下顺序看：

1. 先看某一行非空槽位分布
2. 再判断它属于 Case A / B / C 哪一类
3. 再结合该类对应的 `input_layout / block_table / rope` 语义理解参数
4. 最后再看 replay 脚本里如何补齐标量参数和动态输入

这样会比直接从 31 个槽位硬读更清晰。
