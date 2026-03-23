# Phase 1 Profiling 鍒嗘瀽: 绠楀瓙绋冲畾鎬т笌 Perf-Database 鎺ュ叆鍙鎬?

**杞欢鏍?*: CANN 8.5 / vLLM 0.15.0 / PyTorch 2.9.0 / **aclgraph** (FULL_DECODE_ONLY cudagraph)
**纭欢**: Atlas 800 A3 (Ascend 910B)
**妯″瀷**: DeepSeek-V3 (W8A8, TP=8/DP=2/EP) + Qwen3-32B (BF16, TP=16)
**鏁版嵁**: `/mnt/d/Data/Profiling/Profiling-0313-phase1-e2e-test/`

---

## 鏍稿績缁撹

1. **Qwen3-32B Prefill 璁＄畻绠楀瓙鏋佸叾绋冲畾** 鈥?鍗?99% 鏃堕棿鐨?Top8 绠楀瓙 CV 鍧?<0.01~0.09锛屽彲鐩存帴鏌ヨ〃
2. **DSv3 璁＄畻绠楀瓙鏁翠綋绋冲畾 (CV<0.1)**锛屼絾 DispatchFFNCombine 浠嶉渶鍒嗚В寤烘ā
3. **Qwen3 Decode (graph mode) 鐨?CV 姣?eager 妯″紡鍋忛珮 (0.12-0.17)** 鈥?aclgraph 涓嬪皬 kernel 鐨?CPU dispatch jitter 鍗犳瘮鏇村ぇ
4. **op_mapping 瑕嗙洊鐜?~95%**锛?8 涓?CSV 鏁版嵁鏂囦欢鍙敤锛屽彲鏀寔 mini e2e 楠屾敹

---

## 涓€銆侀噰闆嗗満鏅笌閰嶇疆

| 鍦烘櫙 | 妯″瀷 | 閲忓寲 | 骞惰 | input | output | kernel 鏁?| e2e |
|------|------|------|------|-------|--------|----------|-----|
| Prefill | DSv3 | W8A8 | TP=8,DP=2,EP | 4096 | 1 | 11K | 4.56s |
| Decode | DSv3 | W8A8 | TP=8,DP=2,EP | 4096 | 1536 | 11.5K | 4.88s |
| Prefill | Qwen3-32B | BF16 | TP=16 | 4096 | 1 | 6.4K | 5.82s |
| Decode | Qwen3-32B | BF16 | TP=16 | 4096 | ~1536 | 127K | 3.10s |

鍏抽敭閰嶇疆:
- 鍧囦负 aclgraph 妯″紡 (`--no-enforce-eager`, `FULL_DECODE_ONLY` cudagraph)
- DSv3: `FUSED_MC2=1`, `max_num_seqs=8`, `max_num_batched_tokens=2048`
- Qwen3: `enable-prefix-caching`, `block-size=128`, `max-num-batched-tokens=65536`
- Decode 鍦烘櫙 input=4096锛岃В鍐充簡涔嬪墠 eager 閲囬泦涓?KV cache 閫€鍖?(input_len=1) 鐨勯棶棰?

---

## 浜屻€丵wen3-32B Prefill: 璁＄畻绠楀瓙鏋佺ǔ瀹?

**绾绠楁€绘椂闂? 3,245 ms** (鍗?e2e 55.8%)

| 绠楀瓙 | 鍗犺绠? | 绱% | N | Mean(us) | **Max CV** | 鎺ュ叆绛栫暐 |
|------|---------|------|---|---------|-----------|---------|
| MatMulV3 | 55.7% | 55.7 | 640 | 2825.6 | **0.0024** | 鐩存帴鏌ヨ〃 (鏋佺ǔ瀹? |
| MatMulV2 | 16.5% | 72.3 | 640 | 838.2 | **0.0142** | 鐩存帴鏌ヨ〃 |
| FusedInferAttentionScore | 10.7% | 82.9 | 320 | 1083.6 | **0.0111** | 鐩存帴鏌ヨ〃 |
| split_qkv_rmsnorm_rope | 9.5% | 92.5 | 315 | 983.5 | **0.0026** | 鐩存帴鏌ヨ〃 |
| SwiGlu | 2.8% | 95.3 | 320 | 281.0 | **0.0110** | 鐩存帴鏌ヨ〃 |
| AddRmsNormBias | 2.1% | 97.4 | 640 | 108.5 | **0.0331** | 鐩存帴鏌ヨ〃 |
| ReshapeAndCacheNdKernel | 1.5% | 98.9 | 320 | 151.5 | **0.0934** | 鏌ヨ〃鍙?P50 |
| TensorMove | 0.2% | 99.2 | 325 | 32.7 | 0.0468 | 鏌ヨ〃鍙?P50 |

> 鍗?99% 浠ヤ笂鏃堕棿鐨勭畻瀛?CV 鍧?<0.1, 鏄墍鏈夊満鏅腑鏈€绋冲畾鐨勩€備粎鏈夊嚑涓?1-2us 鐨勬瀬灏忕畻瀛?(Fill, Slice) CV>0.1锛屽 e2e 褰卞搷鍙拷鐣ャ€?
> **Qwen3 Prefill 鏄?mini e2e 楠屾敹鐨勭悊鎯宠捣鐐广€?*

---

## 涓夈€丵wen3-32B Decode (aclgraph): 鍥炬ā寮?CV 鍋忛珮

**绾绠楁€绘椂闂? 2,976 ms** (鎺掗櫎 AivKernel comm stream)

| 绠楀瓙 | 鍗犺绠? | 绱% | N | Mean(us) | **Max CV** | 鎺ュ叆绛栫暐 |
|------|---------|------|---|---------|-----------|---------|
| AivKernel (HCCL) | 鈥?| 鈥?| 17688 | 79.9 | 鈥?| **閫氫俊, 涓嶆煡琛?* |
| MatMulV2 (graph) | 35.6% | 35.6 | 25728 | 20.6 | **0.1496** | 鈿狅笍 P50 鏌ヨ〃 |
| FusedInferAttentionScore | 35.2% | 70.8 | 8576 | 61.2 | **0.0839** | 鏌ヨ〃鍙?P50 |
| split_qkv_rmsnorm_rope | 7.8% | 78.6 | 8442 | 13.8 | **0.0300** | 鐩存帴鏌ヨ〃 |
| AddRmsNormBias (graph) | 6.1% | 84.7 | 17152 | 5.3 | **0.1168** | 鈿狅笍 P50 鏌ヨ〃 |
| MatMulV2_230083 (graph) | 5.0% | 89.7 | 8576 | 8.6 | **0.1375** | 鈿狅笍 P50 鏌ヨ〃 |
| SwiGlu (graph) | 3.0% | 92.7 | 8576 | 5.2 | **0.1701** | 鈿狅笍 P50 鏌ヨ〃 |
| reshape_and_cache | 2.4% | 95.1 | 8576 | 4.2 | 0.1600 | 鈿狅笍 P50 鏌ヨ〃 |

> **鍏抽敭鍙戠幇: 鍥炬ā寮?decode 鐨?CV (0.12-0.17) 鏄庢樉楂樹簬 eager 妯″紡鍚岀畻瀛?(<0.05)銆?*
> 鍘熷洜鍒嗘瀽: decode 鍗?kernel 鏋佺煭 (5-30us), CPU dispatch/cudagraph replay overhead 鍗犳瘮澶? 鍔犱笂 TP=16 鐨?HBM 甯﹀绔炰簤, 瀵艰嚧 mte2 娉㈠姩鍦ㄧ浉瀵瑰€间笂鏇存樉钁椼€?
> **浣嗙粷瀵硅宸湁闄?*: MatMulV2 CV=0.15, 缁濆鍊?卤3us, 鍦?e2e=3098ms 涓崰姣旀瀬浣庛€?
> 娉ㄦ剰: graph-compiled kernel 鍚嶇О甯?hash 鍚庣紑, perf-database CSV 鏌ヨ闇€瑕佸悕绉板綊涓€鍖栥€?

---

## 鍥涖€丏Sv3 Prefill: DispatchFFNCombine 浠嶆槸鎸戞垬

**绾绠楁椂闂?(涓嶅惈 DispatchFFNCombine): 332 ms** (浠呭崰 e2e 7.3%)
**DispatchFFNCombine: 2,515 ms** (鍗?e2e 55.2%, 232 娆¤皟鐢?

| 绠楀瓙 | 鍗犺绠? | 绱% | N | Mean(us) | **Max CV** | 鎺ュ叆绛栫暐 |
|------|---------|------|---|---------|-----------|---------|
| **DispatchFFNCombine** | *鍗曠嫭* | 鈥?| 232 | 10,840 | 鈥?| 鍒嗚В寤烘ā |
| QuantBatchMatmulV3 | 26.8% | 26.8 | 1220 | 72.9 | **0.0857** | 鐩存帴鏌ヨ〃 |
| MatMulV2 | 5.8% | 32.6 | 366 | 53.0 | **0.0220** | 鐩存帴鏌ヨ〃 |
| AscendQuantV2 | 5.4% | 38.0 | 732 | 24.4 | **0.0641** | 鏌ヨ〃鍙?mean |
| Slice | 4.0% | 42.0 | 1342 | 9.8 | **0.0890** | 鏌ヨ〃鍙?P50 |
| AddRmsNormBias | 3.4% | 45.4 | 476 | 24.0 | **0.0487** | 鐩存帴鏌ヨ〃 |
| MoeGatingTopK | 2.5% | 47.9 | 232 | 35.7 | **0.1269** | 鈿狅笍 P50 鏌ヨ〃 |

> 绾绠楃畻瀛愭暣浣撶ǔ瀹?(浠?MoeGatingTopK CV>0.1)銆侱ispatchFFNCombine 鏃犳硶鍋?per-shape CV 鍒嗘瀽 (CANN 瓒呯骇铻嶅悎涓嶉€忔槑), 蹇呴』鍒嗚В涓?sub_kernels 鎴栫敤 P25 杩戜技銆?

---

## 浜斻€丏Sv3 Decode: 閫氫俊鍗犱富瀵?

**绾绠楁椂闂?(涓嶅惈 AivKernel comm): 108 ms** (浠呭崰 e2e 2.2%)
**AivKernel (HCCL comm on Stream 37): 3,425 ms** (鍗?e2e 70.2%)
**DispatchFFNCombine: 1,203 ms** (鍗?e2e 24.7%)

| 绠楀瓙 | 鍗犺绠? | 绱% | N | Mean(us) | **Max CV** | 鎺ュ叆绛栫暐 |
|------|---------|------|---|---------|-----------|---------|
| QuantBatchMatmulV3 | 33.5% | 33.5 | 1525 | 23.7 | **0.0603** | 鐩存帴鏌ヨ〃 |
| FusedInferAttentionScore | 16.9% | 50.4 | 305 | 59.8 | **0.0521** | 鐩存帴鏌ヨ〃 |
| AscendQuantV2 | 8.5% | 58.9 | 915 | 10.1 | **0.1462** | 鈿狅笍 P50 鏌ヨ〃 |
| MoeGatingTopK | 1.6% | 鈥?| 290 | 5.9 | **0.1325** | 鈿狅笍 P50 鏌ヨ〃 |

> FusedInferAttentionScore CV=0.052 (input=4096 context 涓嬪緢绋冲畾), 姣斾箣鍓?eager input_len=1 鐨?CV=0.50 澶у箙鏀瑰杽銆?
> 鏈満鏅腑璁＄畻浠呭崰 e2e 2.2%, 閫氫俊 (70.2%) + DispatchFFNCombine (24.7%) 鍗犵粷瀵逛富瀵笺€備豢鐪熺簿搴︿富瑕佸彇鍐充簬閫氫俊寤烘ā鍜?DispatchFFNCombine 鍒嗚В銆?

---

## 鍏€乷p_mapping 瑕嗙洊搴︿笌 mini e2e 楠屾敹璇勪及

### 6.1 瑕嗙洊搴?

**鏁翠綋 ~95% 瑕嗙洊銆?* 48 涓?CSV 鏁版嵁鏂囦欢瑕嗙洊鎵€鏈変富瑕佽绠楃畻瀛愩€?

| 绫诲埆 | 鐘舵€?| 璇存槑 |
|------|------|------|
| 鏍稿績璁＄畻 (MatMul, Quant, Attention, Norm, SwiGlu) | 鉁?鍏ㄨ鐩?| CSV 鏁版嵁榻愬叏 |
| 閫氫俊 (hcom_allReduce, reduceScatter, allGather) | 鉁?鏄犲皠鏈?| 鏃?CSV锛岀敤甯﹀妯″瀷 |
| DispatchFFNCombine | 鉁?鏄犲皠鏈?| CSV 鏈夛紝composite 鍒嗚В |
| split_qkv_rmsnorm_rope_kernel | 鉁?瑕嗙洊 | vLLM-ascend Triton fused |
| Neg, MaskedFill | 鉂?缂烘槧灏?| 闇€琛ュ厖 op_mapping (鍗犳瘮鏋佷綆) |
| PagedCacheLoadNdKernel, AivKernel | 鈿狅笍 鍐呴儴 kernel | CANN/ATB 鍐呴儴锛屼笉闇€ TC 寤烘ā |
| Sampling (Sort, ArgMax, DSARandomUniform) | 鈿狅笍 | TC 涓嶄豢鐪?sampling锛屽彲蹇界暐 |

### 6.2 communication.json

| 鍦烘櫙 | 甯﹀鏁版嵁 | 閫氫俊绠楀瓙 |
|------|---------|---------|
| DSv3 Prefill | 鉁?鏈?(HCCS ~100 GB/s) | hcom_allGather, hcom_reduceScatter |
| DSv3 Decode | 鉂?鍏ㄩ浂 | hcom_allGather, hcom_reduceScatter |
| Qwen3 Prefill | 鉁?鏈?(HCCS ~115 GB/s) | hcom_allGather, hcom_reduceScatter |
| Qwen3 Decode | 鉂?鍏ㄩ浂 | hcom_allGather, hcom_allReduce |

> Decode 鍦烘櫙閫氫俊甯﹀涓洪浂锛岄渶渚濊禆 microbench 鎴?Prefill 甯﹀澶栨帹銆?

### 6.3 mini e2e 楠屾敹灏辩华搴?

| Workplan 瑕佹眰 | 鏁版嵁灏辩华 | 娉ㄦ剰浜嬮」 |
|--------------|---------|---------|
| Qwen3-32B Prefill | 鉁?| 鏈€绋冲畾鍦烘櫙锛岀悊鎯宠捣鐐?|
| DSv3 Decode | 鉁?| 闇€澶勭悊 DispatchFFNCombine + graph kernel 鍚嶇О褰掍竴鍖?|
| HIT/MISS ratio <50% | 鉁?棰勬湡杈炬爣 | 浠?Neg/MaskedFill 缂烘槧灏?|
| Fallback ops <30% e2e | 鈿狅笍 闇€楠岃瘉 | DispatchFFNCombine 鐨?composite 鏌ヨ鏄惁璧?fallback |

### 6.4 Graph-compiled kernel 鍚嶇О褰掍竴鍖?

Qwen3 Decode 鐨?kernel 鍚嶇О甯︾紪璇?hash 鍚庣紑:
```
MatMulV2_NDNZ_ND_FP16_FP16_false_true_all_229955  鈫? MatMulV2
FusedInferAttentionScore_3b093497fc536d61a77a7a329...  鈫? FusedInferAttentionScore
AddRmsNormBias_352d2859a07d64c080e8dfc836d9897f_33  鈫? AddRmsNormBias
SwiGlu_3_high_performance_27  鈫? SwiGlu
```

perf-database 鐨?`parse_kernel_details.py` 鎴?`ProfilingDataSource` 闇€瑕佸鍔犲悕绉板綊涓€鍖栭€昏緫銆?

---

## 涓冦€侀噰闆嗗缓璁?

| 浼樺厛绾?| 寤鸿 | 鐩殑 |
|--------|------|------|
| P0 | DSv3 鍏抽棴 FUSED_MC2 閲囬泦涓€娆?| 鑾峰彇 DispatchFFNCombine 鍒嗚В鍚庡瓙 kernel 鏁版嵁 |
| P0 | 琛ュ厖 Neg銆丮askedFill 鐨?op_mapping | 娑堥櫎 MISS |
| P1 | DSv3 琛?TP=4/EP=8 (32 NPU) | 瀵归綈鐢熶骇閰嶇疆 |
| P1 | Qwen3 琛?TP=4 鎴?TP=8 | 闄嶄綆閫氫俊姣斾緥 |
| P2 | 閫氫俊 microbench | Decode 鍦烘櫙 communication.json 涓嶅彲鐢?|

---

## 闄勫綍

### A. 涓?eager 妯″紡 CV 瀵规瘮

| 绠楀瓙 | 鍦烘櫙 | eager CV | aclgraph CV | 鍙樺寲 |
|------|------|---------|-------------|------|
| MatMulV2 | Qwen3 Decode | 0.002-0.125 | 0.12-0.15 | 鈫?graph jitter |
| FusedInferAttentionScore | Qwen3 Decode | 0.019-0.342 | 0.084 | 鈫?input=4096 鏀瑰杽 |
| FusedInferAttentionScore | DSv3 Decode | 0.030-0.503 | 0.052 | 鈫撯啌 input=4096 澶у箙鏀瑰杽 |
| AddRmsNormBias | Qwen3 Decode | 0.065-0.094 | 0.117 | 鈫?graph jitter |
| SwiGlu | Qwen3 Decode | 0.038-0.120 | 0.170 | 鈫?graph jitter |
| QuantBatchMatmulV3 | DSv3 Prefill | 0.005-0.069 | 0.011-0.086 | 鈮?鏃犲彉鍖?|

### B. 鍒嗘瀽鑴氭湰

```bash
python3.10 docs/perf_database/reports/profiling_analysis_aclgraph/analyze_phase1.py          # 鍏ㄥ満鏅垎绫绘眹鎬?
python3.10 docs/perf_database/reports/profiling_analysis_aclgraph/analyze_per_shape_cv.py    # Per-shape CV 娣卞害鍒嗘瀽
```

### C. 鐩稿叧鎶ュ憡

- 濮婂鎶ュ憡: [kernel duration 涓?e2e 鏃堕棿鍏崇郴](report_kernel_vs_e2e_zh.md)
- 鍓嶅簭鎶ュ憡 (eager): [绠楀瓙绋冲畾鎬?(eager)](../profiling_analysis_op_stability_zh.md)

