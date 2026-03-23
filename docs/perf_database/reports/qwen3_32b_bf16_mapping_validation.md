# Qwen3-32B BF16 閲忓寲鍦烘櫙鏄犲皠楠岃瘉鎶ュ憡锛圕3 鍙傝€冿級

**鏁版嵁鏉ユ簮**: `kernel_details_qwen3-32b_cann85.csv`
**纭欢**: ATLAS_800_A3_752T_128G_DIE锛?6 鍗★紝BF16 aclgraph锛孴P=16
**CANN 鐗堟湰**: 8.5 / vLLM-ascend 0.15.0
**op_mapping 璺緞**: `tensor_cast/performance_model/perf_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5/op_mapping.yaml`
**璐熻矗浜?*: ZZY锛圚DY 鍗忓姪楠岃瘉锛?
**鏃ユ湡**: 2026-03-11

---

## 涓€銆佽鐩栫巼姹囨€?

| 绫诲埆 | 鑰楁椂 (us) | 鍗犳瘮 | 璇存槑 |
|------|----------|------|------|
| 璁＄畻绠楀瓙锛堢洿鎺ユ槧灏勶級 | 534,098 | 11.97% | TC op 鈫?kernel_type CSV 鏌ヨ |
| 閫氫俊绠楀瓙锛堢洿鎺ユ槧灏勶級 | 3,794,466 | 85.04% | hcom_allReduce_ + hcom_allGather_ + allgatherAicpuKernel |
| TC 涓嶆ā鎷?| 133,323 | 2.99% | Sampling pipeline锛圖SARandomUniform/Sort 绛夛級 |
| **鎬昏鐩?* | **4,328,564** | **97.01%** | 鈥?|

**缁撹**锛歰p_mapping 瑕嗙洊绔埌绔€楁椂 97.01%锛屾弧瓒?>90% 鐩爣銆傛棤 MoE 铻嶅悎绠楀瓙锛屾棤 DispatchFFNCombine锛岀粨鏋勬瘮 DSV3 绠€鍗曘€?

---

## 浜屻€乀op-15 op_mapping 瑕嗙洊鐘舵€?

| 鎺掑悕 | kernel Type | 鑰楁椂鍗犳瘮 | TC op 鏄犲皠 | 鐘舵€?| 澶囨敞 |
|------|------------|---------|-----------|------|------|
| 1 | hcom_allReduce_ | 84.17% | `tensor_cast.all_reduce.default` | 宸查厤缃?| TP all-reduce锛岀粷瀵逛富瀵?|
| 2 | MatMulV2 | 5.49% | `aten.mm.default` | 宸查厤缃?| BF16 GEMM锛孎RACTAL_NZ 鏉冮噸 |
| 3 | FusedInferAttentionScore | 2.66% | `tensor_cast.attention.default` | 宸查厤缃?| attention_special 鏌ヨ |
| 4 | AddRmsNormBias | 1.58% | `tensor_cast.add_rms_norm.default` | 宸查厤缃?| CANN 8.5 fused norm |
| 5 | DSARandomUniform | 1.22% | 鈥?| TC 涓嶆ā鎷?| Sampling pipeline |
| 6 | Sort | 1.06% | 鈥?| TC 涓嶆ā鎷?| Sampling top-k |
| 7 | SwiGlu | 0.75% | `tensor_cast.swiglu.default` | 宸查厤缃?| FFN 婵€娲?|
| 8 | split_qkv_rmsnorm_rope_kernel | 0.70% | `profiling.split_qkv_rmsnorm_rope_kernel` | TC 鍒嗚В | Triton 铻嶅悎锛孴C 鍒嗚В涓?rms_norm + apply_rope |
| 9 | hcom_allGather_ | 0.63% | `tensor_cast.all_gather.default` | 宸查厤缃?| Sampling 闃舵 |
| 10 | ReshapeAndCacheNdKernel | 0.54% | `tensor_cast.reshape_and_cache.default` | 宸查厤缃?| KV cache 鍐欏叆 |
| 11 | allgatherAicpuKernel | 0.24% | `tensor_cast.all_gather.default`锛坅lternate锛?| 宸查厤缃?| AICPU all-gather 鍙樹綋 |
| 12 | SoftmaxV2 | 0.14% | 鈥?| TC 涓嶆ā鎷?| Sampling softmax |
| 13 | RealDiv | 0.10% | `aten.div.Tensor`锛坅lternate RealDiv锛?| 宸查厤缃?| Sampling 鍐呴儴 |
| 14 | MaskedFill | 0.10% | 鈥?| TC 涓嶆ā鎷?| Sampling mask |
| 15 | ArgMaxV2 | 0.10% | 鈥?| TC 涓嶆ā鎷?| Sampling argmax |

**Top-15 瑕嗙洊鐜?*锛?0/15 鏈夌洿鎺?TC op 鏄犲皠锛? 绉?TC 鍒嗚В锛坰plit_qkv_rmsnorm_rope_kernel锛夛紝4 绉?TC 涓嶆ā鎷燂紙Sampling pipeline锛?

---

## 涓夈€佽绠楃畻瀛愰€愭潯楠岃瘉

### 3.1 MatMulV2锛?.49%锛?710 娆★級

**TC op**锛歚aten.mm.default`
**鏄犲皠璺緞**锛歚aclnnMm / aclnnMatmulWeightNz` 鈫?`MatMulV2`

**Profiling 杈撳叆 shape锛團RACTAL_NZ 瑙ｇ爜鍚庯紝BF16 鏍煎紡 (a,b,16,16) 鈫?K=b脳16, N=a脳16锛?*锛?

| 璋冪敤娆℃暟 | 婵€娲?(M,K) | 鏉冮噸 ND (K,N) | 骞冲潎鑰楁椂 | 灞傛帹鏂?|
|---------|-----------|-------------|---------|--------|
| 320脳澶氭壒 | (M, 5120) | (5120, 3200) | ~45 us | FFN gate+up proj锛孨=3200=2脳1600=2脳(25600/16) |
| 320脳澶氭壒 | (M, 1600) | (1600, 5120) | ~35 us | FFN down proj锛孠=1600=25600/16锛孨=hidden=5120 |

**鏋舵瀯楠岃瘉**锛圦wen3-32B锛孴P=16锛夛細
- hidden_size=5120锛宨ntermediate_size=25600锛堟帹鏂級
- 姣?TP锛歩ntermediate/TP = 25600/16 = 1600 鉁?
- gate+up 鍚堝苟锛?脳1600 = 3200 鉁?
- Q proj锛?M, 5120) 脳 (5120, 512) = (M, 512)锛?12=64/16脳128锛? heads 脳 head_dim锛夆€?鏈湪 MatMulV2 涓嚭鐜帮紝璇存槑 Q/K/V proj 琚?split_qkv_rmsnorm_rope_kernel 铻嶅悎

**FRACTAL_NZ 瑙ｇ爜瑙勫垯楠岃瘉**锛?
- BF16 鏍煎紡锛歚(a, b, 16, 16)` 鈫?K=b脳16, N=a脳16
- 绀轰緥锛歚(320, 200, 16, 16)` 鈫?K=200脳16=3200, N=320脳16=5120 鉁擄紙down proj锛?
- 绀轰緥锛歚(320, 48, 16, 16)` 鈫?K=48脳16=768? 涓嶅... 瀹為檯 `(320, 200, 16, 16)` 鈫?K=3200, N=5120
- 娉細`(100, 320, 16, 16)` 鈫?K=320脳16=5120, N=100脳16=1600 鉁擄紙down proj 杞疆瑙嗚锛?

**楠岃瘉鐘舵€?*锛欶RACTAL_NZ 瑙ｇ爜瑙勫垯纭锛宻hape 涓庢灦鏋勫惢鍚?鉁?

### 3.2 FusedInferAttentionScore锛?.66%锛?920 娆★級

**TC op**锛歚tensor_cast.attention.default`锛坄query_mode: attention_special`锛?

**Profiling 杈撳叆 shape**锛堝绉?batch_size锛孠V cache 鍥哄畾锛夛細
```
Q:   (M, 4, 128)          鈥?batch=M锛?28/224/256/272/368锛夛紝4 Q heads/TP锛宧ead_dim=128
K:   (12308, 128, 128)    鈥?paged KV cache
V:   (12308, 128, 128)    鈥?paged KV cache
mask: (2048, 2048) INT8   鈥?attention mask
seq_len: (M,) INT64
```

**KV cache shape 瑙ｈ**锛?
- `(12308, 128, 128)` = `(num_blocks, block_size 脳 num_kv_heads, head_dim)`
- block_size=16, num_kv_heads=8 鈫?16脳8=128 鉁?
- 鎴栵細`(num_blocks, block_size=128, head_dim=128)`锛坆lock_size=128 涔熷彲鑳斤級

**attention_special 鏌ヨ缁村害**锛堣璁℃枃妗?搂4.8锛夛細
- batch_size = M锛堝彉鍖栵級
- num_q_heads = 4锛堝浐瀹氾紝TP=16锛?
- head_dim = 128锛堝浐瀹氾級
- kv_seq = 12308锛坧aged blocks锛岄潪鏍囧噯 seq_len锛?

**楠岃瘉鐘舵€?*锛?
- Q shape 涓庢灦鏋勫惢鍚堬紙4 heads/TP = 64/16锛夆湏
- KV cache 涓?paged 鏍煎紡锛宎ttention_special 鏌ヨ闇€姝ｇ‘鎻愬彇 seq_len锛堜粠 seq_len 鍙傛暟锛岄潪 K shape锛?
- FusedInferAttentionScore.csv 闇€鍖呭惈 Prefill + Decode 澶氱 batch_size 鏁版嵁

### 3.3 AddRmsNormBias锛?.58%锛?840 娆★級

**TC op**锛歚tensor_cast.add_rms_norm.default` / `add_rms_norm2.default`

**Profiling shape**锛歚(M, 5120) + (M, 5120) + (5120,)` 鈥?鍏?ND锛孊F16 鉁?

**楠岃瘉鐘舵€?*锛歴hape 鐩存帴鍖归厤锛屾棤鏍煎紡杞崲 鉁?

### 3.4 SwiGlu锛?.75%锛?920 娆★級

**TC op**锛歚tensor_cast.swiglu.default`

**Profiling shape**锛歚(M, 3200)` 鈥?鍗曡緭鍏ワ紙gate+up 鍚堝苟锛夛紝BF16 鉁?

**TC vs Profiling shape 宸紓**锛堣璁℃枃妗?搂4.5 绫诲瀷5锛夛細
- TC 杈撳叆锛?脳`(M, 1600)`锛坓ate 鍜?up 鍒嗗紑锛?
- Profiling 杈撳叆锛?脳`(M, 3200)`锛堝悎骞讹級
- `profiling_data_source.py` 鐨?SwiGlu shape 瑙勫垯锛歝oncat last dim 鉁?

**楠岃瘉鐘舵€?*锛歴hape 宸紓宸茬煡锛屾煡璇㈣鍒欏凡澶勭悊 鉁?

### 3.5 split_qkv_rmsnorm_rope_kernel锛?.70%锛?890 娆★級

**TC op**锛氭棤鐩存帴瀵瑰簲锛坄profiling.split_qkv_rmsnorm_rope_kernel` 涓?placeholder锛?

**Profiling shape**锛歚(M, 768)` + `(81920, 128)` + `(M,)`
- 768 = (4 Q + 1 K + 1 V) 脳 128 = 6 脳 128锛? Q heads + 1 K head + 1 V head per TP锛夆湏
- 81920 = 640 脳 128锛圧oPE 棰戠巼琛紝max_seq_len=640 脳 head_dim=128锛?

**TC 鍒嗚В**锛歚rms_norm` + `apply_rope`锛堝垎鍒煡璇?RmsNorm.csv 鍜?_triton_rope.csv锛?

**楠岃瘉鐘舵€?*锛?
- TC 鍒嗚В鍚庤€楁椂浼扮畻璇樊闇€楠岃瘉锛圕3 楠岃瘉閲嶇偣锛?
- 63/64 灞傝蛋姝よ瀺鍚?kernel锛?/64 灞傝蛋 `_triton_rope`锛?0 娆★紝0.01%锛?
- 鍒嗚В璇樊棰勬湡鍙帴鍙楋紙rms_norm + rope 鍚勮嚜鑰楁椂涔嬪拰 鈮?铻嶅悎 kernel 鑰楁椂锛?

### 3.6 ReshapeAndCacheNdKernel锛?.54%锛?920 娆★級

**TC op**锛歚tensor_cast.reshape_and_cache.default`

**Profiling shape**锛歚(seq, 1, 128)` 脳 2 + `(12308, 128, 1, 128)` 脳 2 + `(seq,)`
- 鍐欏叆 paged KV cache锛宻eq 鍙樺寲锛?27/252/333/483 绛夛級

**楠岃瘉鐘舵€?*锛歴hape 涓?ND锛孴C 鐩存帴鍖归厤 鉁?

---

## 鍥涖€侀€氫俊绠楀瓙楠岃瘉

### 4.1 hcom_allReduce_锛?4.17%锛?748 娆★級

**TC op**锛歚tensor_cast.all_reduce.default`
**骞冲潎鑰楁椂**锛?84.74 us/娆?

**楠岃瘉鐘舵€?*锛?
- 宸查厤缃负涓?kernel_type 鉁?
- 闇€ C10 閲囬泦 `hcom_allReduce_.csv`锛坢essage_bytes 脳 topology_tier 缃戞牸锛?
- **椋庨櫓 R7**锛氶€氫俊鍗犳瘮 84.2%锛孋ommAnalytic 绮惧害鐩存帴鍐冲畾绔埌绔宸紝Phase 1 mini 楠岃瘉閲嶇偣

### 4.2 hcom_allGather_锛?.63%锛?0 娆★級+ allgatherAicpuKernel锛?.24%锛?0 娆★級

**TC op**锛歚tensor_cast.all_gather.default`锛坅llgatherAicpuKernel 涓?alternate锛?
**骞冲潎鑰楁椂**锛歨com_allGather_ 930.68 us锛宎llgatherAicpuKernel 358.46 us

**楠岃瘉鐘舵€?*锛?
- 涓よ€呭潎涓?Sampling 闃舵鐨?all-gather锛孴C 缁熶竴鏄犲皠鍒?all_gather 鉁?
- allgatherAicpuKernel 涓?AICPU 鎵ц璺緞锛屽姛鑳界瓑浠?

---

## 浜斻€乀C 涓嶆ā鎷熺畻瀛愶紙Sampling Pipeline锛?

| kernel Type | 鑰楁椂鍗犳瘮 | 璇存槑 |
|------------|---------|------|
| DSARandomUniform | 1.22% | 闅忔満鏁扮敓鎴愶紝Sampling 鍐呴儴 |
| Sort | 1.06% | top-k sort |
| SoftmaxV2 | 0.14% | Sampling softmax |
| RealDiv | 0.10% | Sampling 闄ゆ硶锛堝凡閰嶇疆 alternate锛屼絾 Sampling 鍦烘櫙涓嶆ā鎷燂級 |
| MaskedFill | 0.10% | Sampling mask |
| ArgMaxV2 | 0.10% | Sampling argmax |
| Neg | 0.09% | Sampling 鍐呴儴 |
| ApplyTopKTopPCustom | 0.06% | vllm-ascend 鑷畾涔?sampling op |
| GreaterEqual | 0.05% | Sampling 姣旇緝 |
| Transpose | 0.04% | 鏁版嵁甯冨眬杞崲 |
| Log | 0.03% | Sampling 鍐呴儴 |

**鍚堣**锛?.99%锛孴C 涓嶆ā鎷燂紝涓嶅奖鍝嶆帹鐞嗚€楁椂浼扮畻锛圫ampling 涓嶅湪 TC 妯℃嫙鑼冨洿鍐咃級

---

## 鍏€佷笌 DSV3 W8A8 瀵规瘮

| 缁村害 | Qwen3-32B BF16 | DSV3 W8A8 |
|------|---------------|-----------|
| 閫氫俊鍗犳瘮 | 85.0%锛坅ll_reduce 涓诲锛?| 52.1%锛坮educe_scatter + all_gather锛?|
| 璁＄畻鍗犳瘮 | 12.0% | 10.7% |
| 铻嶅悎 MoE op | 鏃?| DispatchFFNCombine锛?5.3%锛?|
| 閲忓寲绠楀瓙 | 鏃?| QuantBatchMatmulV3 + AscendQuantV2 + DynamicQuant |
| 涓昏椋庨櫓 | hcom_allReduce_ 绮惧害锛圧7锛?| DispatchFFNCombine 閲嶅璁℃暟 |
| op_mapping 瑕嗙洊 | 97.01% | 98.02% |

---

## 涓冦€侀獙璇佺粨璁轰笌閬楃暀椤?

### 宸查獙璇侊紙鍙洿鎺ヤ娇鐢級

| kernel Type | 楠岃瘉鐘舵€?| 缃俊搴?|
|------------|---------|--------|
| MatMulV2 | FRACTAL_NZ 瑙ｇ爜瑙勫垯纭锛宻hape 涓庢灦鏋勫惢鍚?| 楂?|
| AddRmsNormBias | ND shape锛岀洿鎺ュ尮閰?| 楂?|
| SwiGlu | shape 宸紓锛?脳half 鈫?1脳full锛夊凡鐭ワ紝鏌ヨ瑙勫垯宸插鐞?| 楂?|
| ReshapeAndCacheNdKernel | ND shape锛岀洿鎺ュ尮閰?| 楂?|
| hcom_allReduce_ | 宸查厤缃紝闇€ C10 閲囬泦鏁版嵁 | 楂?|
| hcom_allGather_ + allgatherAicpuKernel | 宸查厤缃?| 楂?|

### 閬楃暀楠岃瘉椤?

| 椤圭洰 | 椋庨櫓绛夌骇 | 璇存槑 |
|------|---------|------|
| hcom_allReduce_ 绮惧害锛圧7锛?| 楂?| 鍗?84.2%锛孋ommAnalytic 绮惧害鍐冲畾绔埌绔宸紝Phase 1 mini 楠岃瘉蹇呴』纭 |
| FusedInferAttentionScore paged KV shape | 涓?| attention_special 闇€姝ｇ‘浠?seq_len 鍙傛暟鎻愬彇 seq锛岃€岄潪 K shape |
| split_qkv_rmsnorm_rope_kernel 鍒嗚В璇樊 | 涓?| TC 鍒嗚В涓?rms_norm + apply_rope锛岄渶楠岃瘉鍒嗚В鍚庤€楁椂浼扮畻璇樊 <20% |
| MatMulV2 Q/K/V proj shape | 浣?| Q/K/V proj 琚?split_qkv_rmsnorm_rope_kernel 铻嶅悎锛孧atMulV2 涓湭鍑虹幇锛岄渶 TC trace 纭 |

### 涓嬩竴姝ヨ鍔?

1. **C10**锛?.13锛夛細閲囬泦 `hcom_allReduce_.csv`锛坢essage_bytes 脳 topology_tier锛岃繖鏄?Qwen3 鏈€鍏抽敭鏁版嵁锛?
2. **Phase 1 Mini 楠岃瘉**锛?.13锛夛細閲嶇偣纭 CommAnalytic 鍦?hcom_allReduce_ 涓婄殑绮惧害锛圧7锛?
3. **Phase 2 E4**锛?.20锛夛細FusedInferAttentionScore Prefill + Decode microbenchmark 琛ュ厖
4. **C3 楠岃瘉**锛圸ZY锛?.11锛夛細TC dispatch trace 瀵煎嚭锛岄€愭潯瀵规瘮 split_qkv_rmsnorm_rope_kernel 鍒嗚В璇樊

