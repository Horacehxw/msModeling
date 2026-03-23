# NPU Profiling 鍒嗘瀽鎶ュ憡 (涓€): 绠楀瓙绋冲畾鎬т笌 Perf-Database 鎺ュ叆鍙鎬?

**杞欢鏍?*: CANN 8.5 / vLLM 0.15.0 / PyTorch 2.9.0 / Eager 妯″紡
**纭欢**: Atlas 800 A3 (Ascend 910B)
**妯″瀷**: DeepSeek-V3 (W8A8, TP=8/DP=2/EP, 16 NPU) + Qwen3-32B (BF16, TP=16)
**鏁版嵁璺緞**: `/mnt/d/Data/Profiling/dsv3_qwen3_full_torch2.9.0_vllm0.15.0_cann8.5_eager/`
**鍒嗘瀽鑴氭湰**: `docs/perf_database/reports/profiling_analysis_eager/`

---

## 鏍稿績闂

瀵?profiling 涓瘡涓畻瀛? 鍦ㄧ浉鍚?shape 涓嬬殑鎵ц鏃堕棿鏄惁绋冲畾? 鑳藉惁閫氳繃鏌ヨ〃 (mean/P50) 鏉ヤ及绠?kernel duration? 鍝簺绠楀瓙闇€瑕佺壒娈婂鐞?

**鍒嗘瀽鏂规硶**: 鎸夌畻瀛愮被鍨?+ input shape 鍒嗙粍, 璁＄畻缁勫唴 CV (Coefficient of Variation = std/mean). CV<0.1 璁や负绋冲畾鍙煡琛? CV>0.1 闇€娣卞叆鍒嗘瀽鎶栧姩鏉ユ簮.

---

## 涓€銆佸叏绠楀瓙鎺ュ叆鍙鎬ф€昏〃

涓嬭〃姹囨€讳簡 **鎵€鏈?* profiling 涓嚭鐜扮殑绠楀瓙, 鎸夐€氫俊/閫氱畻铻嶅悎/璁＄畻鍒嗙被, 缁欏嚭 perf-database 鎺ュ叆鍙鎬ц瘎浼?

### 1.1 閫氫俊绠楀瓙 (涓嶅彲鏌ヨ〃, 闇€甯﹀寤烘ā)

| 绠楀瓙 | 妯″瀷 | 鍗?e2e (浠ｈ〃鎬у満鏅? | 骞冲潎鑰楁椂 | CV 鑼冨洿 | 鎺ュ叆绛栫暐 |
|------|------|---------|---------|---------|---------|
| hcom_allReduce_ | Qwen3 | **75-97%** (batch鈫戝垯鈫? | 2764 us | >0.5 | 甯﹀妯″瀷: `latency + bytes/BW` |
| hcom_reduceScatter_ | DSv3 | 涓?allGather 鍚堣 **17-56%** | 3208 us | 0.3-2.1 | 甯﹀妯″瀷 |
| hcom_allGather_ | DSv3/Qwen3 | (鍚屼笂, 鍚?DP 閫氫俊) | 1336 us | 0.6-1.9 | 甯﹀妯″瀷 |
| RINGMLAPrefillBF16Kernel | DSv3 | <1% (浠?prefill) | 137 us | 0.01-0.05 | 甯﹀妯″瀷 (MLA Ring Attn) |
| reduce_scatterAicpuKernel | DSv3/Qwen3 | <1% (浠?prefill_4096) | 175-221 us | 鈥?| 甯﹀妯″瀷 (AICPU 璺緞) |
| allgatherAicpuKernel | DSv3/Qwen3 | <1% (浠?prefill_4096) | 183-263 us | 鈥?| 甯﹀妯″瀷 (AICPU 璺緞) |

> **閫氫俊绠楀瓙 CV 澶╃劧 >0.3**, 鍙?HCCL 鎷撴墤銆佺綉缁滅珵浜夈€佸悓姝ョ瓑寰呭奖鍝? 涓嶉€傚悎鏌ヨ〃. communication.json 甯﹀瀛楁鍏ㄤ负 0, 闇€閫氳繃 HCCL dump 鎴?microbench 鑾峰彇.
> 娉? "鍗?e2e"鍩轰簬 step_trace_time.csv 鐨勬纭垎瑙? 鑰岄潪 kernel_details.csv 绠€鍗曟眰鍜? 璇﹁濮婂鎶ュ憡銆妅ernel duration 涓?e2e 鏃堕棿鍏崇郴銆?

### 1.2 閫氱畻铻嶅悎绠楀瓙 (涓嶅彲鐩存帴鏌ヨ〃)

| 绠楀瓙 | 妯″瀷 | 鍗?e2e (浠ｈ〃鎬у満鏅? | 骞冲潎鑰楁椂 | CV 鑼冨洿 | 鎺ュ叆绛栫暐 |
|------|------|---------|---------|---------|---------|
| DispatchFFNCombine | DSv3 | **鍚湪 Computing 鍐? 鍗?e2e 18-82%** | 5467 us | **0.14-2.1** | 鍒嗚В涓?sub_kernels 鍒嗗埆寤烘ā |

> DispatchFFNCombine 鍐呭惈 all-to-all + GroupedMatmul + SwiGlu, AIV time 鈮?Duration (mac 浠?17us), 鏂瑰樊鏉ユ簮鏄唴宓岄€氫俊鍜?MoE 璺敱涓嶅潎. 鏋佺 outlier 杈?275ms (姝ｅ父鍊?600 鍊?.
> 娉? DispatchFFNCombine 鍦?profiling 涓褰掔被鍦?compute stream, 鍥犳 step_trace 鐨?"Computing" 鏃堕棿鍖呭惈浜嗗畠.

### 1.3 璁＄畻绠楀瓙 (鍗犵函璁＄畻鏃堕棿绱 >99%)

**DSv3 绾绠楁€绘椂闂? 1,033,857 us**
> 娉? 涓嬭〃"鍗犺绠?"鏄浉瀵逛簬绾绠楃畻瀛愭€绘椂闂寸殑鍗犳瘮. DSv3 鐨勮绠?stream 杩樺寘鍚?DispatchFFNCombine (閫氱畻铻嶅悎), 鍥犳 compute stream 鎬绘椂闂磋繙澶т簬姝?

| 绠楀瓙 | 鍗犺绠? | 绱% | N | 骞冲潎(us) | CV 鑼冨洿 | 绋冲畾鎬?| 鎺ュ叆绛栫暐 |
|------|---------|------|---|---------|---------|--------|---------|
| QuantBatchMatmulV3 | 39.4% | 39.4 | 15616 | 26.1 | **0.005-0.069** | 鏋佺ǔ瀹?| 鐩存帴鏌ヨ〃鍙?mean |
| ~~Neg~~ | ~~8.9%~~ | 鈥?| 38 | 2414 | 0.79-1.73 | 寮傚父 | **蹇界暐** (warmup artifact, wait_time 鍗?99%) |
| AscendQuantV2 | 6.8% | 46.2 | 6954 | 10.1 | 0.007-0.126 | 绋冲畾 | 鏌ヨ〃鍙?mean |
| MatMulV2 | 6.7% | 52.9 | 4583 | 15.1 | 0.002-0.169 | 涓瓑 | 鏌ヨ〃鍙?P50 |
| AddRmsNormBias | 3.9% | 56.8 | 8662 | 4.7 | 0.030-0.094 | 绋冲畾 | 鐩存帴鏌ヨ〃 |
| DynamicQuant | 3.6% | 60.4 | 8662 | 4.3 | 0.002-0.152 | 绋冲畾 | 鏌ヨ〃鍙?mean |
| TransData | 3.4% | 63.8 | 2074 | 16.9 | **0.009** | 鏋佺ǔ瀹?| 鐩存帴鏌ヨ〃 |
| FusedInferAttentionScore | 3.1% | 66.9 | 2074 | 15.5 | 0.030-**0.503** | 鈿狅笍 batch=1 楂楥V | 鏌ヨ〃鍙?P50, 闇€琛ラ噰闆?|
| MoeGatingTopK | 2.6% | 69.5 | 4118 | 6.6 | 0.090-0.126 | 涓瓑 | 鏌ヨ〃鍙?P50 |
| TransposeBatchMatMul | 2.5% | 72.0 | 2074 | 12.5 | 0.065-0.077 | 绋冲畾 | 鐩存帴鏌ヨ〃 |
| Slice | 2.3% | 74.3 | 5551 | 4.2 | 0.047-0.241 | 涓瓑 | 鏌ヨ〃鍙?P50 |
| InterleaveRope | 2.2% | 76.5 | 2318 | 9.9 | 0.019-0.045 | 绋冲畾 | 鐩存帴鏌ヨ〃 |
| Add | 2.1% | 78.6 | 6545 | 3.3 | 0.005-0.205 | 涓瓑 | 鏌ヨ〃鍙?mean |
| BatchMatMulV2 | 1.7% | 80.3 | 2074 | 8.4 | 0.025-0.033 | 鏋佺ǔ瀹?| 鐩存帴鏌ヨ〃 |
| SwiGlu | 1.7% | 82.0 | 4331 | 4.0 | 0.002-0.127 | 绋冲畾 | 鐩存帴鏌ヨ〃 |
| Transpose | 1.4% | 83.4 | 868 | 16.8 | 0.013-0.251 | 涓瓑 | 鏌ヨ〃鍙?P50 |
| KvRmsNormRopeCache | 1.3% | 84.7 | 2318 | 5.9 | 0.038-0.099 | 绋冲畾 | 鐩存帴鏌ヨ〃 |
| RmsNorm | 1.3% | 86.0 | 2389 | 5.6 | 0.026-0.066 | 绋冲畾 | 鐩存帴鏌ヨ〃 |
| Muls | 1.0% | 87.0 | 4118 | 2.4 | 0.015-0.045 | 绋冲畾 | 鐩存帴鏌ヨ〃 |
| Cast | 0.9% | 87.9 | 7070 | 1.3 | 0.002-0.272 | 灏忕畻瀛?| zero_cost 鎴栨煡琛?|
| BroadcastTo | 0.5% | 88.4 | 509 | 9.2 | 0.055-0.133 | 涓瓑 | 鏌ヨ〃鍙?P50 |
| TensorMove | 0.4% | 88.8 | 492 | 8.8 | 0.080-0.231 | 涓瓑 | 鏌ヨ〃鍙?P50 |
| GreaterEqual | 0.4% | 89.2 | 109 | 36.8 | 0.005-**4.573** | 鈿狅笍 寮傚父 | **蹇界暐** (sampling artifact) |
| Fill | 0.4% | 89.6 | 2227 | 1.7 | 0.044-0.185 | 灏忕畻瀛?| zero_cost |
| AsStrided | 0.3% | 89.9 | 854 | 4.0 | 0.075-0.096 | 绋冲畾 | 鐩存帴鏌ヨ〃 |
| PagedCacheLoadNdKernel | 0.2% | 90.1 | 183 | 11.7 | 0.061-0.117 | 绋冲畾 | 鏌ヨ〃鍙?P50 |
| MaskedFill | 0.2% | 90.3 | 109 | 16.9 | 0.006-0.156 | 涓瓑 | 鏌ヨ〃鍙?P50 |
| 鍏朵綑 18 涓畻瀛?| 0.8% | 鈥?| 鈥?| 鈥?| 鈥?| 鈥?| zero_cost |

**Qwen3-32B 绾绠楁€绘椂闂? 700,212 us**
> 娉? Qwen3 鏃犻€氱畻铻嶅悎绠楀瓙. 鎸?step_trace 鍒嗚В, 璁＄畻鍗?e2e 鐨?2-7% (batch鈫戝垯鈫?, 閫氫俊鍗?75-97%.

| 绠楀瓙 | 鍗犺绠? | 绱% | N | 骞冲潎(us) | CV 鑼冨洿 | 绋冲畾鎬?| 鎺ュ叆绛栫暐 |
|------|---------|------|---|---------|---------|--------|---------|
| MatMulV2 | 42.0% | 42.0 | 15549 | 18.9 | 0.002-**0.125** | 涓瓑 | 鏌ヨ〃鍙?P50 (mte2 鏂瑰樊) |
| FusedInferAttentionScore | 14.1% | 56.1 | 3904 | 25.2 | 0.019-**0.342** | 鈿狅笍 鐗瑰畾shape楂楥V | 鏌ヨ〃鍙?P50, 闇€琛ラ噰闆?|
| RmsNorm | 13.3% | 69.4 | 7869 | 11.9 | 0.032-0.084 | 绋冲畾 | 鐩存帴鏌ヨ〃 |
| AddRmsNormBias | 6.8% | 76.3 | 7808 | 6.1 | 0.065-0.094 | 绋冲畾 | 鐩存帴鏌ヨ〃 |
| MatMulV3 | 5.1% | 81.3 | 128 | 277.7 | **0.008-0.022** | 鏋佺ǔ瀹?| 鐩存帴鏌ヨ〃 (浠?prefill) |
| Slice | 4.1% | 85.4 | 11066 | 2.6 | 0.007-0.150 | 绋冲畾 | 鐩存帴鏌ヨ〃 |
| SwiGlu | 3.1% | 88.5 | 3904 | 5.6 | 0.038-0.120 | 绋冲畾 | 鐩存帴鏌ヨ〃 |
| _triton_rope | 3.1% | 91.6 | 3904 | 5.5 | 0.014-0.079 | 绋冲畾 | 鐩存帴鏌ヨ〃 |
| ReshapeAndCacheNdKernel | 2.1% | 93.8 | 3904 | 3.9 | 0.014-0.138 | 绋冲畾 | 鐩存帴鏌ヨ〃 |
| Sort | 1.9% | 95.7 | 61 | 223.7 | 0.003-0.013 | 鏋佺ǔ瀹?| 鐩存帴鏌ヨ〃 |
| ArgMaxV2 | 0.8% | 96.5 | 61 | 88.2 | <0.01 | 鏋佺ǔ瀹?| 鐩存帴鏌ヨ〃 |
| DSARandomUniform | 0.8% | 97.2 | 61 | 88.2 | 鈥?| 鈥?| 鐩存帴鏌ヨ〃 |
| SoftmaxV2 | 0.5% | 97.7 | 61 | 52.3 | 0.006-0.034 | 鏋佺ǔ瀹?| 鐩存帴鏌ヨ〃 |
| MaskedFill | 0.3% | 98.0 | 122 | 17.8 | 0.013-0.129 | 涓瓑 | 鏌ヨ〃鍙?P50 |
| ApplyTopKTopPCustom | 0.3% | 98.3 | 61 | 30.0 | 0.013-0.030 | 绋冲畾 | 鐩存帴鏌ヨ〃 |
| RealDiv | 0.2% | 98.5 | 122 | 12.4 | 0.000-0.140 | 涓瓑 | 鏌ヨ〃鍙?P50 |
| TensorMove | 0.2% | 98.7 | 128 | 10.0 | 0.114-0.117 | 涓瓑 | 鏌ヨ〃鍙?P50 |
| Cast | 0.2% | 98.8 | 488 | 2.4 | 0.002-0.130 | 灏忕畻瀛?| zero_cost |
| Index | 0.1% | 99.0 | 93 | 10.1 | 0.004-0.264 | 涓瓑 | 鏌ヨ〃鍙?P50 |
| Mul | 0.1% | 99.1 | 183 | 4.7 | 0.001-0.089 | 绋冲畾 | 鐩存帴鏌ヨ〃 |
| 鍏朵綑 15 涓畻瀛?| 0.9% | 鈥?| 鈥?| 鈥?| 鈥?| 鈥?| zero_cost |

### 1.4 楂樻柟宸?(CV>0.1) 璁＄畻绠楀瓙姹囨€?

| 绠楀瓙 | 妯″瀷 | CV max | 鍗犺绠? | 鏂瑰樊鏉ユ簮 | 浠跨湡褰卞搷 |
|------|------|--------|---------|----------|----------|
| **DispatchFFNCombine** | DSv3 | **2.1** | 42.6% 鎬?| 鍐呭祵 all-to-all 閫氫俊 + MoE 璺敱涓嶅潎 | **澶?* 鈥?闇€鍒嗚В寤烘ā |
| **Neg** | DSv3 | 1.73 | 8.9% | wait_time (warmup artifact) | 鏃?鈥?蹇界暐 |
| **GreaterEqual** | DSv3 | 4.57 | 0.4% | 鏋佸皯璋冪敤 (24娆?, sampling artifact | 鏃?鈥?蹇界暐 |
| **FusedInferAttentionScore** | DSv3 | 0.50 | 3.1% | batch=1 shape 涓嶇ǔ瀹?| 涓瓑 鈥?闇€琛ラ暱搴忓垪閲囬泦 |
| FusedInferAttentionScore | Qwen3 | 0.34 | 14.1% | batch=2 prefill 褰㈢姸 | 涓瓑 鈥?闇€琛ラ噰闆?|
| MatMulV2 | Qwen3 | 0.125 | 42.0% | **mte2 甯﹀绔炰簤** (mac 鎭掑畾) | 浣?鈥?P50 鍙秷闄?|
| MatMulV2 | DSv3 | 0.169 | 6.7% | 澶?shape mte2 | 浣?鈥?P50 鍙秷闄?|
| Transpose | DSv3 | 0.251 | 1.4% | mte 娉㈠姩 | 浣?|
| Slice | DSv3 | 0.241 | 2.3% | 澶?seq 鍦烘櫙 | 浣?|
| TensorMove | DSv3 | 0.231 | 0.4% | mte 娉㈠姩 | 鏋佷綆 |
| DynamicQuant | DSv3 | 0.152 | 3.6% | 鍋跺彂 outlier | 浣?|
| AscendQuantV2 | DSv3 | 0.126 | 6.8% | scalar + mte2 | 浣?(缁濆鍊?卤1.2us) |

---

## 浜屻€侀珮鏂瑰樊绠楀瓙鎶栧姩鏉ユ簮鍒嗗眰鍒嗘瀽

瀵逛笂琛ㄤ腑 **浠跨湡褰卞搷涓?澶?鎴?涓瓑"** 鐨勯珮鏂瑰樊绠楀瓙, 浠?vLLM 妗嗘灦 鈫?PyTorch 鈫?CANN 鈫?鏄囪吘寰灦鏋勫洓灞傚垎鏋愭姈鍔ㄦ牴鍥? 骞剁粰鍑轰豢鐪熷缓妯″缓璁?

### 2.1 DispatchFFNCombine (CV=0.14-2.1, DSv3)

**鎶栧姩鏉ユ簮鍒嗗眰:**

| 灞傜骇 | 鎶栧姩鍥犵礌 | 褰卞搷绋嬪害 | 璇佹嵁 |
|------|---------|---------|------|
| **vLLM 妗嗘灦** | MoE routing 涓嶅潎: TopK gating 鍔ㄦ€侀€?expert, 涓嶅悓 step 鐨?token鈫抏xpert 鍒嗛厤涓嶅悓 | **涓昏** | 鍚?shape 涓?Duration 娉㈠姩 10x+; AIV 鏃堕棿 = Duration |
| **vLLM 妗嗘灦** | all-to-all 閫氫俊閲忎笌 routing 鐩稿叧: 涓嶅悓 expert 鍒嗗竷瀵艰嚧璺ㄨ妭鐐规惉杩愰噺涓嶅悓 | **涓昏** | batch=4 outlier 275ms = 閫氫俊鎷ュ |
| **CANN** | DispatchFFNCombine 鏄?CANN 瓒呯骇铻嶅悎 kernel, 鍐呴儴璋冨害涓嶉€忔槑 | 涓瓑 | 鏃犳硶浠?profiling 鍒嗚В鍐呴儴瀛愭搷浣滄椂闂?|
| **鏄囪吘寰灦鏋?* | AIV core 鎵ц閫氫俊鏁版嵁鎼繍鏃跺彈 HCCL stream 绔炰簤褰卞搷 | 涓瓑 | mac=17us 鎭掑畾, AIV=8261us 娉㈠姩 |

**浠跨湡寤烘ā寤鸿:**
- **涓嶅彲鐩存帴鏌ヨ〃**. 蹇呴』鍒嗚В涓?sub_kernels: `permute_tokens + GroupedMatmul脳2 + GroupedMatmulSwigluQuant + unpermute_tokens + all_to_all脳2`
- 鑻ュ垎瑙?profiling 鏁版嵁涓嶅彲寰? 鍙 DispatchFFNCombine 鍙?**P25** (鈮堢函璁＄畻涓嬮檺), 閫氫俊閮ㄥ垎鐢ㄥ甫瀹芥ā鍨嬪彔鍔?
- 闀挎湡: 鐢ㄤ笉甯?FUSED_MC2 鐨?eager profiling 鑾峰彇鍒嗚В鍚庡悇瀛?kernel 鐨勭嫭绔嬫暟鎹?

### 2.2 FusedInferAttentionScore (CV 鏈€楂?0.50/0.34)

**鎶栧姩鏉ユ簮鍒嗗眰:**

| 灞傜骇 | 鎶栧姩鍥犵礌 | 褰卞搷绋嬪害 | 璇佹嵁 |
|------|---------|---------|------|
| **vLLM 妗嗘灦** | Decode bench `input_len=1` 鈫?KV cache 浠?3-4 tokens, 灞炰簬閫€鍖栧満鏅?| **涓昏** | batch=1 shape CV=0.50, 鍏朵粬 shape CV<0.08 |
| **PyTorch** | PagedAttention KV cache block-table 瀵诲潃: 鏋佺煭搴忓垪涓?block-table overhead 姣斾緥澶?| 涓瓑 | scalar time 娉㈠姩 |
| **CANN** | FusedInferAttention kernel 瀵规瀬灏?seq 鏈?tile 鏁堢巼閫€鍖?| 涓瓑 | mac=0, 涓昏鍦?scalar+AIV |
| **鏄囪吘寰灦鏋?* | AIC scalar core 鎺у埗閫昏緫 + AIV 骞惰鎵ц鐨勫悓姝ュ紑閿€ | 浣?| AIC scalar std ~1us |

**浠跨湡寤烘ā寤鸿:**
- 褰撳墠 CV=0.50 鐨?shape 鏄?`batch=1, KV_cache=891 blocks` 鐨勯€€鍖栧満鏅? **涓嶄唬琛ㄧ湡瀹炴帹鐞?*
- **琛ュ厖閲囬泦**: decode bench 鏀圭敤 `input_len=512` 鎴?`input_len=2048`, 纭繚 KV cache 鏈夊悎鐞嗘繁搴?
- 姝ｅ父 shape (batch>2, 鍚堢悊 KV depth) CV<0.08, **鍙煡琛ㄥ彇 P50**
- Prefill 鍦烘櫙 (M=3072) 鐨?CV=0.019, 闈炲父绋冲畾

### 2.3 MatMulV2 on Qwen3 (CV 鏈€楂?0.125)

**鎶栧姩鏉ユ簮鍒嗗眰:**

| 灞傜骇 | 鎶栧姩鍥犵礌 | 褰卞搷绋嬪害 | 璇佹嵁 |
|------|---------|---------|------|
| **vLLM 妗嗘灦** | 鏃犵洿鎺ュ奖鍝?鈥?MatMul shape 鐢辨ā鍨嬬粨鏋勫喅瀹? 鏃犲姩鎬佸彉鍖?| 鏃?| 鈥?|
| **PyTorch** | 鏃?鈥?eager 妯″紡涓?MatMul 鐩存帴璋冨害鍒?CANN | 鏃?| 鈥?|
| **CANN** | ND鈫扚RACTAL_NZ 鏍煎紡杞崲鍙兘褰卞搷 L1 cache 鍛戒腑鐜?| 浣?| TransData CV=0.009 璇存槑杞崲鏈韩绋冲畾 |
| **鏄囪吘寰灦鏋?* | **mte2 (DDR鈫扡1 璇绘暟鎹? 鏄敮涓€鏂瑰樊婧?*: mac std=0.00, mte2 CV=0.13 | **涓昏** | TP=16 涓?16 NPU 鍏变韩 HBM 鎬荤嚎, 甯﹀绔炰簤瀵艰嚧 mte2 娉㈠姩 |

**浠跨湡寤烘ā寤鸿:**
- mac 鏃堕棿瀹屽叏纭畾 (std=0.00), **璁＄畻閮ㄥ垎瀹屽叏鍙煡琛?*
- mte2 鏂瑰樊 ~13% 鏉ヨ嚜 HBM 甯﹀绔炰簤, 鏄井鏋舵瀯灞傞潰鐨勯殢鏈哄洜绱?
- 寤鸿鍙?**P50**, 璇樊 卤15% 鍦?e2e 灞傞潰褰卞搷鏈夐檺 (MatMulV2 浠呭崰 e2e 0.67%)
- TP 鏁板噺灏?(濡?TP=4) 鍙樉钁楅檷浣庡甫瀹界珵浜? 棰勬湡 CV 浼氫笅闄?

### 2.4 鍏朵粬 CV>0.1 绠楀瓙鐨勫叡鎬ц寰?

| 鎶栧姩妯″紡 | 娑夊強绠楀瓙 | 鏍瑰洜灞傜骇 | 浠跨湡澶勭悊 |
|---------|---------|---------|---------|
| **mte2 甯﹀绔炰簤** | MatMulV2, Transpose, TensorMove, Slice (澶hape) | 鏄囪吘寰灦鏋?| P50 鏌ヨ〃; TP 鏁板奖鍝?|
| **scalar 鎺у埗閫昏緫娉㈠姩** | AscendQuantV2, DynamicQuant | CANN kernel | mean 鏌ヨ〃 (缁濆鍊?<2us) |
| **warmup/璋冨害 artifact** | Neg, GreaterEqual | 妗嗘灦/OS | zero_cost 蹇界暐 |
| **鍔ㄦ€佽矾鐢变笉鍧?* | DispatchFFNCombine | vLLM MoE routing | 鍒嗚В寤烘ā |
| **閫€鍖栧満鏅?* | FusedInferAttentionScore (batch=1) | bench 璁捐 | 琛ラ噰闆?|

---

## 涓夈€丳rofiling 瑕嗙洊搴︿笌閰嶇疆璇勪及

### 3.1 瑕嗙洊搴?

| 缁村害 | DSv3 | Qwen3-32B | 璇勪及 |
|------|------|-----------|------|
| Prefill | input=256/1024/4096 | input=256/1024/4096 | OK |
| Decode | batch=1/4/8/16/32 | batch=1/4/8/16/32 | OK |
| 閲忓寲 | W8A8 (ascend) | BF16 (鏃犻噺鍖? | OK |
| 骞惰 | TP=8, DP=2, EP | TP=16 | 娉ㄦ剰宸紓 |
| 妯″紡 | enforce-eager | enforce-eager | OK |

### 3.2 闇€鍏虫敞鐨勯厤缃棶棰?

1. **DSv3 骞惰閰嶇疆**: profiling 鐢?TP=8/DP=2/EP (16 NPU), 宸查獙璇佺敓浜ч厤缃负 TP=4/EP=8 (32 NPU). 璁＄畻 kernel 鏁版嵁鍙€氱敤, 浣嗛€氫俊 pattern 鍜?expert 鍒嗗竷涓嶅悓.
2. **Qwen3 TP=16 閫氫俊姣斾緥鏋侀珮**: 32B 妯″瀷 16 鍗″苟琛? 閫氫俊鍗?e2e 鐨?75-97%, 涓嶄唬琛ㄧ敓浜?(TP=4/8). 浣嗙函璁＄畻 kernel 鏁版嵁浠嶅彲鐢?
3. **Decode KV cache 閫€鍖?*: `input_len=1, output_len=3` 鈫?KV cache 浠?3-4 tokens, FusedInferAttentionScore 涓嶄唬琛ㄧ湡瀹為暱搴忓垪.
4. **DSv3 max_num_batched_tokens=2048**: 闄愬埗浜?prefill batch size, prefill 鍧囦负 batch=1.
5. **communication.json 甯﹀涓洪浂**: HCCL profiler 鏈噰闆?transit bandwidth.

---

## 鍥涖€侀噰闆嗚ˉ鍏呭缓璁?

| 浼樺厛绾?| 寤鸿 | 鐩殑 |
|--------|------|------|
| P0 | Decode bench 澧炲姞 `input_len=512/2048` | 淇 FusedInferAttentionScore KV cache 閫€鍖?|
| P0 | DSv3 鍏抽棴 FUSED_MC2 閲囬泦涓€娆?| 鑾峰彇 DispatchFFNCombine 鍒嗚В鍚庡瓙 kernel 鏁版嵁 |
| P1 | DSv3 琛ュ厖 TP=4/EP=8 (32 NPU) 閰嶇疆 | 瀵归綈鐢熶骇閰嶇疆 |
| P1 | Qwen3 琛ュ厖 TP=4 鎴?TP=8 | 闄嶄綆閫氫俊姣斾緥, 鎻愰珮璁＄畻绠楀瓙浠ｈ〃鎬?|
| P2 | 閫氫俊 microbench 鑾峰彇 HCCL 甯﹀ | communication.json 涓嶅彲鐢?|

---

## 闄勫綍

### A. 鏁版嵁鏂囦欢

**Eager 妯″紡** (`dsv3_qwen3_full_torch2.9.0_vllm0.15.0_cann8.5_eager/`):
姣忎釜鍦烘櫙鍖呭惈: `kernel_details.csv` (閫?kernel 璇︽儏), `op_statistic.csv` (鑱氬悎缁熻), `step_trace_time.csv` (compute/comm/free 鍒嗚В), `communication.json`, `trace_view.json`.

### B. 鍒嗘瀽鑴氭湰

```bash
python3.10 docs/perf_database/reports/profiling_analysis/analyze_profiling.py    # 鍒嗙被姹囨€?
python3.10 docs/perf_database/reports/profiling_analysis/profiling_analysis.py   # CV/outlier 娣卞害鍒嗘瀽
```

### C. 鐩稿叧鎶ュ憡

- 濮婂鎶ュ憡: [kernel duration 涓?e2e 鏃堕棿鍏崇郴](profiling_analysis_kernel_vs_e2e_zh.md) 鈥?鍒嗘瀽 kernel_duration 鍔犳€讳笌绔埌绔椂闂寸殑宸窛銆丆PU gap銆乧ompute/comm overlap

