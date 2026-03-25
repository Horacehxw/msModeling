# FusedInferAttentionScore CSV 列说明

本文说明以下文件中每一列的含义，以及它们与 `torch_npu.npu_fused_infer_attention_score` 接口参数和模型推理物理语义的对应关系：

- `tensor_cast/performance_model/perf_database/data/{device}/vllm_ascend/{version}/FusedInferAttentionScore.csv`

适用对象：

- 数据库使用者
- FIA replay 使用者
- profiling 与 microbench 对齐分析人员

---

## 1. 这张表是什么

这张表描述的是算子 `FusedInferAttentionScore` 的 profiling 聚合结果，以及为了 replay 该算子而额外补充的 runtime metadata。

它不是单次调用日志，而是“按 shape / dtype / format 等签名聚合后的算子画像”。  
因此一行通常表示一类 FIA 调用，而不是某一次具体调用。

这张表里的数据主要来自两部分：

1. `parse_kernel_details.py` 从整网 profiling 目录解析得到的统计信息
2. `start_microbench.py` / replay 脚本回填的单算子执行信息

---

## 2. FIA 接口参数概览

当前 CSV 主要围绕下面这个接口展开：

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

其中 CSV 主要覆盖：

- tensor 参数的 shape / dtype / format
- 部分 runtime 推断出来的标量参数
- profiling 统计值

---

## 3. 表中各列分组说明

### 3.1 基础签名列

这些列描述这一行 FIA 调用的“算子签名”。

#### `OP State`

- 含义：算子状态，例如 `static` / `dynamic`
- 对应接口：不直接对应某个参数
- 物理含义：表示这条 FIA kernel 在图编译或执行阶段属于哪种状态，常用于区分不同执行路径

#### `Accelerator Core`

- 含义：执行核心类型，例如 `MIX_AIC`
- 对应接口：不直接对应某个参数
- 物理含义：表示该 kernel 落在哪类硬件执行单元上

#### `Input Shapes`

- 含义：FIA tensor 输入参数的 shape 列表，按固定槽位顺序展开，共 31 槽
- 对应接口：
  - `query`
  - `key`
  - `value`
  - `atten_mask`
  - `actual_seq_lengths`
  - `actual_seq_lengths_kv`
  - `block_table`
  - `query_rope`
  - `key_rope`
  - 以及其他当前大多未出现的 tensor 参数
- 物理含义：描述 Q/K/V、mask、KV block 映射、rope 张量等在实际推理中的张量结构

当前常用槽位：

- `0`: `query`
- `1`: `key`
- `2`: `value`
- `4`: `atten_mask`
- `5`: `actual_seq_lengths`
- `6`: `actual_seq_lengths_kv`
- `14`: `block_table`
- `24`: `query_rope`
- `25`: `key_rope`

#### `Input Data Types`

- 含义：与 `Input Shapes` 一一对应的输入 dtype 列表
- 对应接口：对应各 tensor 参数的 dtype
- 物理含义：表示 Q/K/V、mask、长度列表、block_table 等输入在实际推理中的数据类型

#### `Input Formats`

- 含义：与 `Input Shapes` 一一对应的输入 format 列表
- 对应接口：对应各 tensor 参数的 format
- 物理含义：描述张量在 NPU 上的布局格式，例如 `ND`

#### `Output Shapes`

- 含义：输出 shape 列表
- 对应接口：
  - 主输出 `attention output`
  - 可选第二输出 `softmax_lse`
- 物理含义：描述 FIA 输出的结果张量形状

#### `Output Data Types`

- 含义：输出 dtype 列表
- 对应接口：对应输出张量 dtype
- 物理含义：描述 attention 输出和可选辅助输出的类型

#### `Output Formats`

- 含义：输出 format 列表
- 对应接口：对应输出张量 format
- 物理含义：描述输出张量在设备侧的布局格式

---

### 3.2 Profiling 统计列

这些列来自整网 profiling 聚合结果。

#### `Profiling Average Duration(us)`

- 含义：整网运行时，该类 FIA 调用的平均耗时
- 对应接口：不直接对应单个接口参数
- 物理含义：这是 FIA 在真实模型推理过程中表现出的平均执行时间

#### `Profiling Median Duration(us)`

- 含义：整网运行时的中位耗时
- 物理含义：比平均值更抗异常值，用于观察典型延迟

#### `Profiling Std Duration(us)`

- 含义：整网运行时耗时标准差
- 物理含义：反映这类 FIA 调用的时延波动程度

#### `Profiling Average aicore_time(us)` 等 `Profiling Average ...`

- 含义：AIC/AIV 等硬件执行统计的平均值
- 对应接口：不直接对应单个参数
- 物理含义：描述算子在硬件执行过程中的各类时间、比例和资源利用情况

重点示例：

- `Profiling Average aicore_time(us)`：AICore 执行时间
- `Profiling Average aic_mac_time(us)`：MAC 计算相关时间
- `Profiling Average aic_scalar_time(us)`：标量执行时间
- `Profiling Average aic_mte1_time(us)` / `aic_mte2_time(us)`：搬运相关时间
- `Profiling Average cube_utilization(%)`：Cube 单元利用率

---

### 3.3 Runtime metadata 列

这些列是为了让 FIA replay 更接近整网执行状态而补充的。

#### `Runtime source_profile`

- 含义：该行数据来源的 profiling 目录名
- 物理含义：标识这条 FIA 样本来自哪次整网 profiling

#### `Runtime operator_input_shapes_raw`

- 含义：从 `operator_details` / `trace` 视角看到的原始输入 shape 槽位
- 对应接口：对应完整 FIA 调用参数槽位
- 物理含义：比 `Input Shapes` 更接近上层 API 记录的原始参数布局

#### `Runtime operator_input_dtypes_raw`

- 含义：原始输入参数的 dtype / 参数类型槽位
- 物理含义：可以看出哪些参数是 Tensor、`ScalarList`、`Scalar`、`TensorList`

#### `Runtime operator_extra_shape_slots`

- 含义：31 个标准 tensor 槽位之外，额外出现的 shape 槽位
- 物理含义：用于保留上层 API 中 kernel summary 没有显式保留的信息

#### `Runtime operator_shape_slot_count`

- 含义：原始参数槽位总数
- 物理含义：反映 FIA 调用在上层 API 记录里一共展开了多少个位置参数/关键字参数槽位

#### `Runtime operator_scalar_list_slots`

- 含义：原始参数中 `ScalarList` 类型槽位索引
- 对应接口：通常对应 `actual_seq_lengths_kv` 这类运行时长度列表
- 物理含义：帮助定位动态长度类参数

#### `Runtime operator_scalar_slots`

- 含义：原始参数中 `Scalar` 类型槽位索引
- 对应接口：通常对应 `num_heads`、`scale`、`sparse_mode`、`block_size` 等标量
- 物理含义：帮助识别哪些控制参数会影响 kernel 路径

#### `Runtime operator_tensor_list_slots`

- 含义：原始参数中 `TensorList` 类型槽位索引
- 物理含义：表示该调用里存在 list-of-tensor 形式的参数

#### `Runtime actual_seq_lengths_shape`

- 含义：`actual_seq_lengths` 的 shape
- 对应接口：`actual_seq_lengths`
- 物理含义：表示 query 侧真实长度列表的尺寸

#### `Runtime actual_seq_lengths_values`

- 含义：`actual_seq_lengths` 的实际值列表
- 对应接口：`actual_seq_lengths`
- 物理含义：表示当前 batch 中每条 query 的真实长度或累计长度信息

#### `Runtime actual_seq_lengths_kv_shape`

- 含义：`actual_seq_lengths_kv` 的 shape
- 对应接口：`actual_seq_lengths_kv`
- 物理含义：表示 KV 侧长度列表的尺寸，通常与 batch size 相关

#### `Runtime actual_seq_lengths_kv_values`

- 含义：`actual_seq_lengths_kv` 的实际值列表
- 对应接口：`actual_seq_lengths_kv`
- 物理含义：表示每个 request 在 KV cache 中的真实上下文长度

这是 FIA replay 最关键的字段之一。  
如果缺失，paged attention replay 往往只能做近似构造。

#### `Runtime avg_seq_len`

- 含义：`mean(actual_seq_lengths_kv)`
- 对应接口：不是原始接口参数，是辅助分析列
- 物理含义：表示这一类 FIA 调用对应的平均 KV 上下文长度

当前规则：

- 如果能解析到 `Runtime actual_seq_lengths_kv_values`，则取其均值
- 如果解析不到，则默认写 `4096`

#### `Runtime block_table_shape`

- 含义：`block_table` 的 shape
- 对应接口：`block_table`
- 物理含义：表示 paged attention 中 block 映射表的大小，通常是：
  - 第一维：batch size
  - 第二维：每条 request 允许的最大 block 数

#### `Runtime block_table_valid_blocks`

- 含义：每条 request 实际使用的有效 block 数信息
- 对应接口：`block_table`
- 物理含义：比单纯 `block_table_shape` 更接近真实 KV cache 占用情况

#### `Runtime num_heads`

- 含义：实际 attention head 数
- 对应接口：`num_heads`
- 物理含义：query 侧 head 数，决定并行粒度和计算规模

#### `Runtime num_key_value_heads`

- 含义：实际 KV head 数
- 对应接口：`num_key_value_heads`
- 物理含义：决定是否是 MHA / GQA / MQA 风格，对 kernel 路径影响很大

#### `Runtime sparse_mode`

- 含义：实际 sparse mode
- 对应接口：`sparse_mode`
- 物理含义：决定 attention mask 的解释方式和 kernel 实现路径

对当前 vLLM Ascend FIA 来说，这是 replay 成败和时延对齐的关键字段之一。

#### `Runtime input_layout`

- 含义：输入布局，如 `TND`、`BNSD_NBSD`
- 对应接口：`input_layout`
- 物理含义：描述 Q/K/V 张量维度的语义排列方式，直接影响 kernel 选路

常见情况：

- `TND`：普通 TND attention
- `BNSD_NBSD`：MLA rope 路径

#### `Runtime block_size`

- 含义：paged KV 的 block 大小
- 对应接口：`block_size`
- 物理含义：表示 KV cache 中每个 block 存储多少 token

#### `Runtime attn_state`

- 含义：attention 所处状态，例如：
  - `prefill_no_cache`
  - `paged_runtime`
  - `mla_paged_runtime`
- 对应接口：不是原始参数，是 replay 辅助状态
- 物理含义：反映当前 FIA 属于哪类推理阶段

#### `Runtime kv_cache_mode`

- 含义：K/V 的组织方式，例如：
  - `non_paged_direct`
  - `paged_cache_pool`
- 对应接口：不是原始参数，是 replay 辅助状态
- 物理含义：说明 K/V 是真实序列张量，还是从 KV cache pool 中访问

#### `Runtime metadata_completeness`

- 含义：runtime metadata 的完备度
- 物理含义：表示这条数据到底是“真实 runtime 支撑”还是“部分推断”

当前可能值：

- `runtime_values`
- `shape_and_dtype_only`
- `shape_only`
- `inferred_only`

---

## 4. 这张表里最重要的物理对象

### `query`

- 表示当前 step 要做 attention 的 Q
- 在 prefill 中通常对应一段新输入 token
- 在 decode 中通常对应当前新增 token 的 query

### `key` / `value`

- 表示用于 attention 的历史上下文
- 在非 paged 场景下，shape 往往直接反映真实 KV 长度
- 在 paged 场景下，shape 更可能对应整个 KV cache pool 大小，而不是当前 batch 的真实已用长度

### `actual_seq_lengths` / `actual_seq_lengths_kv`

- 表示 query 侧和 KV 侧的真实长度信息
- 决定了 attention 实际参与计算的 token 范围
- 是整网与 microbench 对齐时最关键的 runtime metadata

### `block_table`

- 表示 paged attention 中“逻辑序列 -> KV cache block”之间的映射关系
- 是 decode/paged attention 路径的核心数据结构之一

### `query_rope` / `key_rope`

- 表示 rope 相关输入
- 在 MLA 路径中特别重要

### `sparse_mode` / `atten_mask`

- 一起决定 mask 语义和 attention kernel 路径
- 二者不匹配时，算子可能直接执行失败

---

## 5. 如何读这张表

推荐顺序：

1. 先看 `Input Shapes / Input Data Types / Input Formats`
2. 再看 `Runtime input_layout / Runtime sparse_mode / Runtime block_size`
3. 再看 `Runtime actual_seq_lengths_kv_shape / Runtime actual_seq_lengths_kv_values / Runtime avg_seq_len`
4. 如果是 paged 场景，再看 `Runtime block_table_shape / Runtime kv_cache_mode / Runtime attn_state`
5. 最后再结合 `Profiling Average Duration(us)` 看该类 FIA 在整网中的性能表现

---

## 6. 使用注意事项

### 6.1 同 shape 不等于同 runtime

即使 `Input Shapes` 相同，也不代表：

- `actual_seq_lengths_kv` 相同
- `block_table` 有效映射相同
- `num_key_value_heads` 相同
- `sparse_mode` 相同

因此 FIA 的性能不能只看 shape。

### 6.2 paged 场景最容易失真

如果没有真实 `actual_seq_lengths_kv_values` 和 `block_table` 信息，replay 往往只能构造一个合法输入，而不是整网真实输入。

### 6.3 `Runtime avg_seq_len` 是分析辅助列

它不是原始接口参数，也不能替代真实 `actual_seq_lengths_kv_values`。  
它的作用是帮助快速估计该类 FIA 的平均上下文长度级别。

---

## 7. 一句话总结

这张表本质上是在回答三个问题：

1. 这类 FIA 调用长什么样：`Input/Output Shapes + DType + Format`
2. 它在真实推理里大概处于什么状态：`Runtime ...`
3. 它在整网里跑得怎么样：`Profiling Average Duration(us)` 和各类硬件统计

如果要做 replay 对齐，最需要关注的是：

- `Runtime actual_seq_lengths_kv_values`
- `Runtime avg_seq_len`
- `Runtime block_table_shape`
- `Runtime num_key_value_heads`
- `Runtime sparse_mode`
- `Runtime input_layout`
