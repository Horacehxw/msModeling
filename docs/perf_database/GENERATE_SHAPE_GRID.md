# `generate_shape_grid.py` 使用说明

本文档说明 [`tools/perf_data_collection/generate_shape_grid.py`](G:\仿真开发\msmodeling\tools\perf_data_collection\generate_shape_grid.py) 的用途、参数、输入输出、主要算子规则，以及它和 [`tools/perf_data_collection/op_replay`](G:\仿真开发\msmodeling\tools\perf_data_collection\op_replay) 的兼容关系。

## 1. 脚本定位

`generate_shape_grid.py` 的作用不是从零设计 shape，而是基于已有 perf database CSV 模板追加新的合成样本：

1. 递归扫描 perf database 目录下的 CSV。
2. 从每个 CSV 的已有行读取 `Input Shapes`、`Output Shapes`、`Input Formats`。
3. 按算子类型生成约束感知的新 shape。
4. 追加到原 CSV，保留已有真实数据。

这个脚本当前有三个目标：

1. 让新增样本尽量接近整网 profiling 中真实出现过的 shape 家族。
2. 保持关键维度关系不被破坏，例如 matmul contract 维、norm hidden 维、cache 结构、rope 结构。
3. 让重点算子的 CSV 可以被 `op_replay/*_run.py` 直接读取并执行，减少 replay 报错。

## 2. 入口文件

- 脚本入口：
  - [`generate_shape_grid.py`](G:\仿真开发\msmodeling\tools\perf_data_collection\generate_shape_grid.py)
- replay 目录：
  - [`op_replay`](G:\仿真开发\msmodeling\tools\perf_data_collection\op_replay)
- attention CSV 结构参考：
  - [`FusedInferAttentionScore_CSV_MAPPING.md`](G:\仿真开发\msmodeling\docs\perf_database\FusedInferAttentionScore_CSV_MAPPING.md)

## 3. 基本用法

直接对默认根目录执行：

```powershell
python .\tools\perf_data_collection\generate_shape_grid.py
```

按设备和版本目录执行：

```powershell
python .\tools\perf_data_collection\generate_shape_grid.py `
  --device ATLAS_800_A3_752T_128G_DIE `
  --vllm-ascend-version 0.15.0_torch2.9.0_cann8.5 `
  --rows 1000 `
  --min-value 1 `
  --max-value 20000 `
  --seed 123
```

显式指定目录执行：

```powershell
python .\tools\perf_data_collection\generate_shape_grid.py `
  --data-dir .\tensor_cast\performance_model\perf_database\data `
  --rows 1000 `
  --min-value 1 `
  --max-value 20000 `
  --seed 123
```

## 4. 参数说明

- `--data-dir`
  - 显式指定 CSV 根目录。
  - 如果传了该参数，优先使用它。
- `--device`
  - 设备名。
  - 规则与 [`parse_kernel_details.py`](G:\仿真开发\msmodeling\tools\perf_data_collection\parse_kernel_details.py) 一致。
  - 必须和 `--vllm-ascend-version` 一起使用。
- `--vllm-ascend-version`
  - vLLM-Ascend 版本。
  - 规则与 [`parse_kernel_details.py`](G:\仿真开发\msmodeling\tools\perf_data_collection\parse_kernel_details.py) 一致。
  - 如果不以 `v` 开头，脚本会自动补 `v`。
  - 必须和 `--device` 一起使用。
- `--rows`
  - 每个 CSV 追加的行数。
- `--min-value`
  - 随机维度最小值。
- `--max-value`
  - 随机维度最大值。
- `--seed`
  - 可选随机种子，方便复现。

## 5. 目录解析规则

- 如果传了 `--data-dir`，直接使用该目录。
- 如果没有传 `--data-dir`，但传了 `--device` 和 `--vllm-ascend-version`，则使用：
  - `tensor_cast/performance_model/profiling_database/data/{device}/vllm_ascend/{version}/`
- 如果三者都没传，则回退到默认根目录：
  - `tensor_cast/performance_model/profiling_database/data`

## 6. 运行时行为

### 6.1 进度条

脚本会显示两级进度：

- 总文件进度：`Files [####----] x/y`
- 当前文件内进度：`Rows [####----] x/y`

### 6.2 跳过策略

下列文件会被跳过：

- 没有 `Input Shapes` 列的 CSV。
- 有 `Input Shapes` 列，但没有可用模板的 CSV。
- 特例：`Range` 可以仅依赖 `Output Shapes` 模板生成。

### 6.3 输出列处理

脚本会保留以下列：

- `OP State`
- `Accelerator Core`
- `Input Data Types`
- `Input Formats`
- `Output Data Types`
- `Output Formats`

性能指标类列如果列名包含以下关键词，会被填成 `0`：

- `duration`
- `latency`
- `time`
- `cycles`
- `ratio`
- `miss`
- `utilization`

## 7. 通用 shape 生成规则

### 7.1 模板解析

脚本把 `Input Shapes` / `Output Shapes` 解析成分号分隔的 shape 槽位列表。

例如：

```text
"16,5120;320,48,16,16"
```

会解析成：

- `(16, 5120)`
- `(320, 48, 16, 16)`

空槽位保留为 `()`。

### 7.2 随机维度规则

- 模板维度等于 `1` 时，生成后仍保持 `1`。
- 普通维度优先在模板维度附近波动，通常约在 `[1/2, 2x]` 范围内。
- 某些维度会按 `8` 或 `16` 对齐。
- 对同一个模板数字，脚本会尽量在输入和输出间保持一致映射关系。

## 8. 已支持的算子类别

### 8.1 Binary Elementwise

统一规则：

- 输入 0 作为主 shape。
- 输入 1 保持模板中的同形或广播关系。
- 输出 shape 等于输入 0。

覆盖算子：

- `Add`
- `Equal`
- `FloorDiv`
- `FloorMod`
- `GreaterEqual`
- `Less`
- `LessAiCore`
- `LogicalAnd`
- `LogicalAndAiCore`
- `MaskedFill`
- `MaskedFillAiCore`
- `Mul`
- `MulAiCore`
- `NotEqual`
- `RealDiv`
- `Sub`
- `SubAiCore`

### 8.2 Unary / Same-shape

统一规则：

- 输入 shape 扰动。
- 输出 shape 与输入一致。

覆盖算子：

- `Cast`
- `CastAiCore`
- `Fill`
- `Log`
- `LogicalNot`
- `LogicalNotAiCore`
- `Muls`
- `Neg`
- `SoftmaxV2`
- `TensorMove`
- `ZerosLike`

### 8.3 MatMul / Quant MatMul

统一原则：

- 保持 matmul contract 维合法。
- 保持 `ND` / `FRACTAL_NZ` 结构不乱。
- 常见维度按 `8` / `16` 对齐。

覆盖算子：

- `MatMul`
- `MatMulCommon`
- `MatMulV2`
- `MatMulV3`
- `BatchMatMulV2`
- `MatmulReduceScatterV2`
- `QuantBatchMatmulV3`
- `GroupedMatmul`
- `GroupedMatmulSwigluQuant`

相关 replay：

- [`MatMulV2_run.py`](G:\仿真开发\msmodeling\tools\perf_data_collection\op_replay\MatMulV2_run.py)
- [`MatMulV3_run.py`](G:\仿真开发\msmodeling\tools\perf_data_collection\op_replay\MatMulV3_run.py)
- [`QuantBatchMatmulV3_run.py`](G:\仿真开发\msmodeling\tools\perf_data_collection\op_replay\QuantBatchMatmulV3_run.py)

### 8.4 Norm / Quant / Fused FFN

覆盖算子：

- `RmsNorm`
- `AddRmsNorm`
- `AddRmsNormBias`
- `AddRmsNormDynamicQuant`
- `AscendQuantV2`
- `DynamicQuant`
- `SwiGlu`

重点规则：

- `RmsNorm`
  - `gamma` 必须是 `(hidden,)`
- `AddRmsNormBias`
  - `x1` / `x2` 必须同形
  - `gamma` / `beta` 必须是一维 hidden 向量
- `DynamicQuant`
  - 当前 replay 只接受单输入
- `AscendQuantV2`
  - 维持 `x + scale (+ zero_points)` 的结构

相关 replay：

- [`AddRmsNormBias_run.py`](G:\仿真开发\msmodeling\tools\perf_data_collection\op_replay\AddRmsNormBias_run.py)
- [`RmsNorm_run.py`](G:\仿真开发\msmodeling\tools\perf_data_collection\op_replay\RmsNorm_run.py)
- [`AscendQuantV2_run.py`](G:\仿真开发\msmodeling\tools\perf_data_collection\op_replay\AscendQuantV2_run.py)
- [`DynamicQuant_run.py`](G:\仿真开发\msmodeling\tools\perf_data_collection\op_replay\DynamicQuant_run.py)
- [`SwiGlu_run.py`](G:\仿真开发\msmodeling\tools\perf_data_collection\op_replay\SwiGlu_run.py)

### 8.5 Rope / Attention / Cache

覆盖算子：

- `ApplyRotaryPosEmb`
- `InterleaveRope`
- `AtbRopeKernel`
- `_triton_rope`
- `split_qkv_rmsnorm_rope_kernel`
- `split_qkv_rmsnorm_rope_kernel_0`
- `FusedInferAttentionScore`
- `ReshapeAndCacheNdKernel`
- `reshape_and_cache_200000000`
- `KvRmsNormRopeCache`
- `PagedCacheLoadNdKernel`
- `RINGMLAPrefillBF16Kernel`

重点规则：

- `InterleaveRope`
  - 生成三个 4D 输入
  - `x=(B,N,S,D)`，`cos=(B,1,1,D)`，`sin=(B,1,1,D)`
- `split_qkv_rmsnorm_rope_kernel`
  - 生成：
    - `qkv=(tokens, q_hidden + 2 * kv_hidden)`
    - `cos_sin_cache=(max_position_embeddings, rope_dim)`
    - `positions=(tokens,)`
- `ReshapeAndCacheNdKernel`
  - 生成：
    - `key=(tokens, kv_heads, head_dim)`
    - `value=(tokens, kv_heads, head_dim)`
    - `key_cache=(num_blocks, block_size, kv_heads, head_dim)`
    - `value_cache=(num_blocks, block_size, kv_heads, head_dim)`
    - `slot_mapping=(tokens,)`
- `KvRmsNormRopeCache`
  - 按 replay 需要的 12 槽位家族生成
- `FusedInferAttentionScore`
  - 按模板区分：
    - TND 3D attention
    - 带 `query_rope/key_rope` 的 4D MLA attention
  - 保持 31 个输入槽位

相关 replay：

- [`InterleaveRope_run.py`](G:\仿真开发\msmodeling\tools\perf_data_collection\op_replay\InterleaveRope_run.py)
- [`split_qkv_rmsnorm_rope_kernel_run.py`](G:\仿真开发\msmodeling\tools\perf_data_collection\op_replay\split_qkv_rmsnorm_rope_kernel_run.py)
- [`FusedInferAttentionScore_run.py`](G:\仿真开发\msmodeling\tools\perf_data_collection\op_replay\FusedInferAttentionScore_run.py)
- [`ReshapeAndCacheNdKernel_run.py`](G:\仿真开发\msmodeling\tools\perf_data_collection\op_replay\ReshapeAndCacheNdKernel_run.py)
- [`KvRmsNormRopeCache_run.py`](G:\仿真开发\msmodeling\tools\perf_data_collection\op_replay\KvRmsNormRopeCache_run.py)

### 8.6 Gather / Index / Scatter / Shape Transform

覆盖算子：

- `GatherV2`
- `GatherV2AiCore`
- `GatherV3`
- `GatherElementsV2`
- `Index`
- `IndexPutV2`
- `BroadcastTo`
- `Slice`
- `SliceAiCore`
- `Transpose`
- `TransposeBatchMatMul`
- `AsStrided`
- `TransData`
- `ConcatD`
- `PadV3`
- `Tile`
- `ScatterElementsV2`
- `SelectV2`
- `Range`
- `RepeatInterleave`
- `expand_kernel`

### 8.7 MoE 相关

覆盖算子：

- `MoeGatingTopK`
- `DispatchFFNCombine`
- `MoeDistributeDispatchV2`
- `MoeDistributeCombineV2`
- `MoeTokenPermute`
- `MoeTokenUnpermute`

这些规则会尽量保持 `tokens`、`topk`、`experts`、`hidden`、`intermediate`、routed token 数之间的结构关系。

## 9. 与 `op_replay` 的兼容性约束

当前脚本已针对下列 replay 重点算子做兼容修正：

- `Add`
- `AddRmsNormBias`
- `AscendQuantV2`
- `DynamicQuant`
- `FusedInferAttentionScore`
- `GatherV2`
- `InterleaveRope`
- `KvRmsNormRopeCache`
- `MaskedFill`
- `MatMulV2`
- `MatMulV3`
- `QuantBatchMatmulV3`
- `ReshapeAndCacheNdKernel`
- `RmsNorm`
- `SwiGlu`
- `TensorMove`
- `split_qkv_rmsnorm_rope_kernel`

兼容原则：

- 输入槽位数必须与 replay 脚本一致。
- 输入 rank 必须满足 replay 中的显式检查。
- 可选输入位必须保留空槽位位置。
- 对 `FRACTAL_NZ`、cache、rope、paged attention 等特殊格式不能只做 generic 扰动。

## 10. profiling 依赖来源

本轮规则增强主要参考：

- `G:\仿真开发\profiling\最新profiling_0317`
- [`tools/perf_data_collection/op_replay`](G:\仿真开发\msmodeling\tools\perf_data_collection\op_replay)

其中：

- profiling 用来学习真实 shape 家族。
- replay 脚本用来约束哪些 shape 家族真的能跑起来。

## 11. 当前限制

目前仍未强建模或只能保守处理的主要是：

- 纯通信类：
  - `hcom_allReduce_`
  - `hcom_allGather_`
  - `hcom_alltoallv_`
  - `hcom_reduceScatter_`
- 一些低频或模板质量不稳定的算子
- 某些算子虽然已有规则，但还没有做真实设备侧的全量 replay 回归

## 12. 维护建议

后续新增或修改算子规则时，建议按下面顺序做：

1. 先看目标算子的 perf CSV 模板。
2. 再看对应的 `op_replay/*_run.py` 是否对槽位数、rank、dtype、格式有硬约束。
3. 如果 replay 有显式检查，优先满足 replay 契约。
4. 如果 profiling 中存在多种 shape 家族，不要用一个规则硬合并。
5. 修改后至少执行：
   - `py -3 -m py_compile tools/perf_data_collection/generate_shape_grid.py`
   - 对目标 CSV 做 `--rows 1` 或 `--rows 2` 小规模试跑

## 13. 推荐工作流

```powershell
py -3 -m py_compile .\tools\perf_data_collection\generate_shape_grid.py

python .\tools\perf_data_collection\generate_shape_grid.py `
  --device ATLAS_800_A3_752T_128G_DIE `
  --vllm-ascend-version 0.15.0_torch2.9.0_cann8.5 `
  --rows 100 `
  --seed 123
```

如果后续需要验证 replay：

```powershell
py -3 .\tools\perf_data_collection\op_replay\MatMulV2_run.py `
  --device ATLAS_800_A3_752T_128G_DIE `
  --vllm-ascend-version 0.15.0
```

