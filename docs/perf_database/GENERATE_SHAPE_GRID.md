# `generate_shape_grid.py` 浣跨敤璇存槑

鏈枃妗ｈ鏄?[`tools/perf_data_collection/generate_shape_grid.py`](G:\浠跨湡寮€鍙慭msmodeling\tools\perf_data_collection\generate_shape_grid.py) 鐨勭敤閫斻€佸弬鏁般€佽緭鍏ヨ緭鍑恒€佷富瑕佺畻瀛愯鍒欙紝浠ュ強瀹冨拰 [`tools/perf_data_collection/op_replay`](G:\浠跨湡寮€鍙慭msmodeling\tools\perf_data_collection\op_replay) 鐨勫吋瀹瑰叧绯汇€?

## 1. 鑴氭湰瀹氫綅

`generate_shape_grid.py` 鐨勪綔鐢ㄤ笉鏄粠闆惰璁?shape锛岃€屾槸鍩轰簬宸叉湁 perf database CSV 妯℃澘杩藉姞鏂扮殑鍚堟垚鏍锋湰锛?

1. 閫掑綊鎵弿 perf database 鐩綍涓嬬殑 CSV銆?
2. 浠庢瘡涓?CSV 鐨勫凡鏈夎璇诲彇 `Input Shapes`銆乣Output Shapes`銆乣Input Formats`銆?
3. 鎸夌畻瀛愮被鍨嬬敓鎴愮害鏉熸劅鐭ョ殑鏂?shape銆?
4. 杩藉姞鍒板師 CSV锛屼繚鐣欏凡鏈夌湡瀹炴暟鎹€?

杩欎釜鑴氭湰褰撳墠鏈変笁涓洰鏍囷細

1. 璁╂柊澧炴牱鏈敖閲忔帴杩戞暣缃?profiling 涓湡瀹炲嚭鐜拌繃鐨?shape 瀹舵棌銆?
2. 淇濇寔鍏抽敭缁村害鍏崇郴涓嶈鐮村潖锛屼緥濡?matmul contract 缁淬€乶orm hidden 缁淬€乧ache 缁撴瀯銆乺ope 缁撴瀯銆?
3. 璁╅噸鐐圭畻瀛愮殑 CSV 鍙互琚?`op_replay/*_run.py` 鐩存帴璇诲彇骞舵墽琛岋紝鍑忓皯 replay 鎶ラ敊銆?

## 2. 鍏ュ彛鏂囦欢

- 鑴氭湰鍏ュ彛锛?
  - [`generate_shape_grid.py`](G:\浠跨湡寮€鍙慭msmodeling\tools\perf_data_collection\generate_shape_grid.py)
- replay 鐩綍锛?
  - [`op_replay`](G:\浠跨湡寮€鍙慭msmodeling\tools\perf_data_collection\op_replay)
- attention CSV 缁撴瀯鍙傝€冿細
  - [`FusedInferAttentionScore_CSV_MAPPING.md`](G:\浠跨湡寮€鍙慭msmodeling\docs\perf_database\FusedInferAttentionScore_CSV_MAPPING.md)

## 3. 鍩烘湰鐢ㄦ硶

鐩存帴瀵归粯璁ゆ牴鐩綍鎵ц锛?

```powershell
python .\tools\perf_data_collection\generate_shape_grid.py
```

鎸夎澶囧拰鐗堟湰鐩綍鎵ц锛?

```powershell
python .\tools\perf_data_collection\generate_shape_grid.py `
  --device ATLAS_800_A3_752T_128G_DIE `
  --vllm-ascend-version 0.15.0_torch2.9.0_cann8.5 `
  --rows 1000 `
  --min-value 1 `
  --max-value 20000 `
  --seed 123
```

鏄惧紡鎸囧畾鐩綍鎵ц锛?

```powershell
python .\tools\perf_data_collection\generate_shape_grid.py `
  --data-dir .\tensor_cast\performance_model\perf_database\data `
  --rows 1000 `
  --min-value 1 `
  --max-value 20000 `
  --seed 123
```

## 4. 鍙傛暟璇存槑

- `--data-dir`
  - 鏄惧紡鎸囧畾 CSV 鏍圭洰褰曘€?
  - 濡傛灉浼犱簡璇ュ弬鏁帮紝浼樺厛浣跨敤瀹冦€?
- `--device`
  - 璁惧鍚嶃€?
  - 瑙勫垯涓?[`parse_kernel_details.py`](G:\浠跨湡寮€鍙慭msmodeling\tools\perf_data_collection\parse_kernel_details.py) 涓€鑷淬€?
  - 蹇呴』鍜?`--vllm-ascend-version` 涓€璧蜂娇鐢ㄣ€?
- `--vllm-ascend-version`
  - vLLM-Ascend 鐗堟湰銆?
  - 瑙勫垯涓?[`parse_kernel_details.py`](G:\浠跨湡寮€鍙慭msmodeling\tools\perf_data_collection\parse_kernel_details.py) 涓€鑷淬€?
  - 濡傛灉涓嶄互 `v` 寮€澶达紝鑴氭湰浼氳嚜鍔ㄨˉ `v`銆?
  - 蹇呴』鍜?`--device` 涓€璧蜂娇鐢ㄣ€?
- `--rows`
  - 姣忎釜 CSV 杩藉姞鐨勮鏁般€?
- `--min-value`
  - 闅忔満缁村害鏈€灏忓€笺€?
- `--max-value`
  - 闅忔満缁村害鏈€澶у€笺€?
- `--seed`
  - 鍙€夐殢鏈虹瀛愶紝鏂逛究澶嶇幇銆?

## 5. 鐩綍瑙ｆ瀽瑙勫垯

- 濡傛灉浼犱簡 `--data-dir`锛岀洿鎺ヤ娇鐢ㄨ鐩綍銆?
- 濡傛灉娌℃湁浼?`--data-dir`锛屼絾浼犱簡 `--device` 鍜?`--vllm-ascend-version`锛屽垯浣跨敤锛?
  - `tensor_cast/performance_model/perf_database/data/{device}/vllm_ascend/{version}/`
- 濡傛灉涓夎€呴兘娌′紶锛屽垯鍥為€€鍒伴粯璁ゆ牴鐩綍锛?
  - `tensor_cast/performance_model/perf_database/data`

## 6. 杩愯鏃惰涓?

### 6.1 杩涘害鏉?

鑴氭湰浼氭樉绀轰袱绾ц繘搴︼細

- 鎬绘枃浠惰繘搴︼細`Files [####----] x/y`
- 褰撳墠鏂囦欢鍐呰繘搴︼細`Rows [####----] x/y`

### 6.2 璺宠繃绛栫暐

涓嬪垪鏂囦欢浼氳璺宠繃锛?

- 娌℃湁 `Input Shapes` 鍒楃殑 CSV銆?
- 鏈?`Input Shapes` 鍒楋紝浣嗘病鏈夊彲鐢ㄦā鏉跨殑 CSV銆?
- 鐗逛緥锛歚Range` 鍙互浠呬緷璧?`Output Shapes` 妯℃澘鐢熸垚銆?

### 6.3 杈撳嚭鍒楀鐞?

鑴氭湰浼氫繚鐣欎互涓嬪垪锛?

- `OP State`
- `Accelerator Core`
- `Input Data Types`
- `Input Formats`
- `Output Data Types`
- `Output Formats`

鎬ц兘鎸囨爣绫诲垪濡傛灉鍒楀悕鍖呭惈浠ヤ笅鍏抽敭璇嶏紝浼氳濉垚 `0`锛?

- `duration`
- `latency`
- `time`
- `cycles`
- `ratio`
- `miss`
- `utilization`

## 7. 閫氱敤 shape 鐢熸垚瑙勫垯

### 7.1 妯℃澘瑙ｆ瀽

鑴氭湰鎶?`Input Shapes` / `Output Shapes` 瑙ｆ瀽鎴愬垎鍙峰垎闅旂殑 shape 妲戒綅鍒楄〃銆?

渚嬪锛?

```text
"16,5120;320,48,16,16"
```

浼氳В鏋愭垚锛?

- `(16, 5120)`
- `(320, 48, 16, 16)`

绌烘Ы浣嶄繚鐣欎负 `()`銆?

### 7.2 闅忔満缁村害瑙勫垯

- 妯℃澘缁村害绛変簬 `1` 鏃讹紝鐢熸垚鍚庝粛淇濇寔 `1`銆?
- 鏅€氱淮搴︿紭鍏堝湪妯℃澘缁村害闄勮繎娉㈠姩锛岄€氬父绾﹀湪 `[1/2, 2x]` 鑼冨洿鍐呫€?
- 鏌愪簺缁村害浼氭寜 `8` 鎴?`16` 瀵归綈銆?
- 瀵瑰悓涓€涓ā鏉挎暟瀛楋紝鑴氭湰浼氬敖閲忓湪杈撳叆鍜岃緭鍑洪棿淇濇寔涓€鑷存槧灏勫叧绯汇€?

## 8. 宸叉敮鎸佺殑绠楀瓙绫诲埆

### 8.1 Binary Elementwise

缁熶竴瑙勫垯锛?

- 杈撳叆 0 浣滀负涓?shape銆?
- 杈撳叆 1 淇濇寔妯℃澘涓殑鍚屽舰鎴栧箍鎾叧绯汇€?
- 杈撳嚭 shape 绛変簬杈撳叆 0銆?

瑕嗙洊绠楀瓙锛?

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

缁熶竴瑙勫垯锛?

- 杈撳叆 shape 鎵板姩銆?
- 杈撳嚭 shape 涓庤緭鍏ヤ竴鑷淬€?

瑕嗙洊绠楀瓙锛?

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

缁熶竴鍘熷垯锛?

- 淇濇寔 matmul contract 缁村悎娉曘€?
- 淇濇寔 `ND` / `FRACTAL_NZ` 缁撴瀯涓嶄贡銆?
- 甯歌缁村害鎸?`8` / `16` 瀵归綈銆?

瑕嗙洊绠楀瓙锛?

- `MatMul`
- `MatMulCommon`
- `MatMulV2`
- `MatMulV3`
- `BatchMatMulV2`
- `MatmulReduceScatterV2`
- `QuantBatchMatmulV3`
- `GroupedMatmul`
- `GroupedMatmulSwigluQuant`

鐩稿叧 replay锛?

- [`MatMulV2_run.py`](G:\浠跨湡寮€鍙慭msmodeling\tools\perf_data_collection\op_replay\MatMulV2_run.py)
- [`MatMulV3_run.py`](G:\浠跨湡寮€鍙慭msmodeling\tools\perf_data_collection\op_replay\MatMulV3_run.py)
- [`QuantBatchMatmulV3_run.py`](G:\浠跨湡寮€鍙慭msmodeling\tools\perf_data_collection\op_replay\QuantBatchMatmulV3_run.py)

### 8.4 Norm / Quant / Fused FFN

瑕嗙洊绠楀瓙锛?

- `RmsNorm`
- `AddRmsNorm`
- `AddRmsNormBias`
- `AddRmsNormDynamicQuant`
- `AscendQuantV2`
- `DynamicQuant`
- `SwiGlu`

閲嶇偣瑙勫垯锛?

- `RmsNorm`
  - `gamma` 蹇呴』鏄?`(hidden,)`
- `AddRmsNormBias`
  - `x1` / `x2` 蹇呴』鍚屽舰
  - `gamma` / `beta` 蹇呴』鏄竴缁?hidden 鍚戦噺
- `DynamicQuant`
  - 褰撳墠 replay 鍙帴鍙楀崟杈撳叆
- `AscendQuantV2`
  - 缁存寔 `x + scale (+ zero_points)` 鐨勭粨鏋?

鐩稿叧 replay锛?

- [`AddRmsNormBias_run.py`](G:\浠跨湡寮€鍙慭msmodeling\tools\perf_data_collection\op_replay\AddRmsNormBias_run.py)
- [`RmsNorm_run.py`](G:\浠跨湡寮€鍙慭msmodeling\tools\perf_data_collection\op_replay\RmsNorm_run.py)
- [`AscendQuantV2_run.py`](G:\浠跨湡寮€鍙慭msmodeling\tools\perf_data_collection\op_replay\AscendQuantV2_run.py)
- [`DynamicQuant_run.py`](G:\浠跨湡寮€鍙慭msmodeling\tools\perf_data_collection\op_replay\DynamicQuant_run.py)
- [`SwiGlu_run.py`](G:\浠跨湡寮€鍙慭msmodeling\tools\perf_data_collection\op_replay\SwiGlu_run.py)

### 8.5 Rope / Attention / Cache

瑕嗙洊绠楀瓙锛?

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

閲嶇偣瑙勫垯锛?

- `InterleaveRope`
  - 鐢熸垚涓変釜 4D 杈撳叆
  - `x=(B,N,S,D)`锛宍cos=(B,1,1,D)`锛宍sin=(B,1,1,D)`
- `split_qkv_rmsnorm_rope_kernel`
  - 鐢熸垚锛?
    - `qkv=(tokens, q_hidden + 2 * kv_hidden)`
    - `cos_sin_cache=(max_position_embeddings, rope_dim)`
    - `positions=(tokens,)`
- `ReshapeAndCacheNdKernel`
  - 鐢熸垚锛?
    - `key=(tokens, kv_heads, head_dim)`
    - `value=(tokens, kv_heads, head_dim)`
    - `key_cache=(num_blocks, block_size, kv_heads, head_dim)`
    - `value_cache=(num_blocks, block_size, kv_heads, head_dim)`
    - `slot_mapping=(tokens,)`
- `KvRmsNormRopeCache`
  - 鎸?replay 闇€瑕佺殑 12 妲戒綅瀹舵棌鐢熸垚
- `FusedInferAttentionScore`
  - 鎸夋ā鏉垮尯鍒嗭細
    - TND 3D attention
    - 甯?`query_rope/key_rope` 鐨?4D MLA attention
  - 淇濇寔 31 涓緭鍏ユЫ浣?

鐩稿叧 replay锛?

- [`InterleaveRope_run.py`](G:\浠跨湡寮€鍙慭msmodeling\tools\perf_data_collection\op_replay\InterleaveRope_run.py)
- [`split_qkv_rmsnorm_rope_kernel_run.py`](G:\浠跨湡寮€鍙慭msmodeling\tools\perf_data_collection\op_replay\split_qkv_rmsnorm_rope_kernel_run.py)
- [`FusedInferAttentionScore_run.py`](G:\浠跨湡寮€鍙慭msmodeling\tools\perf_data_collection\op_replay\FusedInferAttentionScore_run.py)
- [`ReshapeAndCacheNdKernel_run.py`](G:\浠跨湡寮€鍙慭msmodeling\tools\perf_data_collection\op_replay\ReshapeAndCacheNdKernel_run.py)
- [`KvRmsNormRopeCache_run.py`](G:\浠跨湡寮€鍙慭msmodeling\tools\perf_data_collection\op_replay\KvRmsNormRopeCache_run.py)

### 8.6 Gather / Index / Scatter / Shape Transform

瑕嗙洊绠楀瓙锛?

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

### 8.7 MoE 鐩稿叧

瑕嗙洊绠楀瓙锛?

- `MoeGatingTopK`
- `DispatchFFNCombine`
- `MoeDistributeDispatchV2`
- `MoeDistributeCombineV2`
- `MoeTokenPermute`
- `MoeTokenUnpermute`

杩欎簺瑙勫垯浼氬敖閲忎繚鎸?`tokens`銆乣topk`銆乣experts`銆乣hidden`銆乣intermediate`銆乺outed token 鏁颁箣闂寸殑缁撴瀯鍏崇郴銆?

## 9. 涓?`op_replay` 鐨勫吋瀹规€х害鏉?

褰撳墠鑴氭湰宸查拡瀵逛笅鍒?replay 閲嶇偣绠楀瓙鍋氬吋瀹逛慨姝ｏ細

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

鍏煎鍘熷垯锛?

- 杈撳叆妲戒綅鏁板繀椤讳笌 replay 鑴氭湰涓€鑷淬€?
- 杈撳叆 rank 蹇呴』婊¤冻 replay 涓殑鏄惧紡妫€鏌ャ€?
- 鍙€夎緭鍏ヤ綅蹇呴』淇濈暀绌烘Ы浣嶄綅缃€?
- 瀵?`FRACTAL_NZ`銆乧ache銆乺ope銆乸aged attention 绛夌壒娈婃牸寮忎笉鑳藉彧鍋?generic 鎵板姩銆?

## 10. profiling 渚濊禆鏉ユ簮

鏈疆瑙勫垯澧炲己涓昏鍙傝€冿細

- `G:\浠跨湡寮€鍙慭profiling\鏈€鏂皃rofiling_0317`
- [`tools/perf_data_collection/op_replay`](G:\浠跨湡寮€鍙慭msmodeling\tools\perf_data_collection\op_replay)

鍏朵腑锛?

- profiling 鐢ㄦ潵瀛︿範鐪熷疄 shape 瀹舵棌銆?
- replay 鑴氭湰鐢ㄦ潵绾︽潫鍝簺 shape 瀹舵棌鐪熺殑鑳借窇璧锋潵銆?

## 11. 褰撳墠闄愬埗

鐩墠浠嶆湭寮哄缓妯℃垨鍙兘淇濆畧澶勭悊鐨勪富瑕佹槸锛?

- 绾€氫俊绫伙細
  - `hcom_allReduce_`
  - `hcom_allGather_`
  - `hcom_alltoallv_`
  - `hcom_reduceScatter_`
- 涓€浜涗綆棰戞垨妯℃澘璐ㄩ噺涓嶇ǔ瀹氱殑绠楀瓙
- 鏌愪簺绠楀瓙铏界劧宸叉湁瑙勫垯锛屼絾杩樻病鏈夊仛鐪熷疄璁惧渚х殑鍏ㄩ噺 replay 鍥炲綊

## 12. 缁存姢寤鸿

鍚庣画鏂板鎴栦慨鏀圭畻瀛愯鍒欐椂锛屽缓璁寜涓嬮潰椤哄簭鍋氾細

1. 鍏堢湅鐩爣绠楀瓙鐨?perf CSV 妯℃澘銆?
2. 鍐嶇湅瀵瑰簲鐨?`op_replay/*_run.py` 鏄惁瀵规Ы浣嶆暟銆乺ank銆乨type銆佹牸寮忔湁纭害鏉熴€?
3. 濡傛灉 replay 鏈夋樉寮忔鏌ワ紝浼樺厛婊¤冻 replay 濂戠害銆?
4. 濡傛灉 profiling 涓瓨鍦ㄥ绉?shape 瀹舵棌锛屼笉瑕佺敤涓€涓鍒欑‖鍚堝苟銆?
5. 淇敼鍚庤嚦灏戞墽琛岋細
   - `py -3 -m py_compile tools/perf_data_collection/generate_shape_grid.py`
   - 瀵圭洰鏍?CSV 鍋?`--rows 1` 鎴?`--rows 2` 灏忚妯¤瘯璺?

## 13. 鎺ㄨ崘宸ヤ綔娴?

```powershell
py -3 -m py_compile .\tools\perf_data_collection\generate_shape_grid.py

python .\tools\perf_data_collection\generate_shape_grid.py `
  --device ATLAS_800_A3_752T_128G_DIE `
  --vllm-ascend-version 0.15.0_torch2.9.0_cann8.5 `
  --rows 100 `
  --seed 123
```

濡傛灉鍚庣画闇€瑕侀獙璇?replay锛?

```powershell
py -3 .\tools\perf_data_collection\op_replay\MatMulV2_run.py `
  --device ATLAS_800_A3_752T_128G_DIE `
  --vllm-ascend-version 0.15.0
```


