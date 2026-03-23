# DeepSeek-V3 W8A8 閲忓寲鍦烘櫙鏄犲皠楠岃瘉鎶ュ憡锛圕8锛?

**鏁版嵁鏉ユ簮**: `kernel_details_deepseekv3-cann85.csv`
**纭欢**: ATLAS_800_A3_752T_128G_DIE锛?2 鍗★紝W8A8 aclgraph锛孴P=4 EP=8
**CANN 鐗堟湰**: 8.5 / vLLM-ascend 0.15.0
**璐熻矗浜?*: HDY
**鏃ユ湡**: 2026-03-11
**鍓嶇疆**: C7 op_mapping 鎵╁睍鎶ュ憡锛坄dsv3_w8a8_op_mapping_verification.md`锛?

---

## 涓€銆佽鐩栫巼姹囨€?

| 绫诲埆 | 鑰楁椂 (us) | 鍗犳瘮 | 璇存槑 |
|------|----------|------|------|
| 璁＄畻绠楀瓙锛堢洿鎺ユ槧灏勶級 | 255,339 | 10.65% | TC op 鈫?kernel_type CSV 鏌ヨ |
| 閫氫俊绠楀瓙锛堢洿鎺ユ槧灏勶級 | 1,248,854 | 52.08% | hcom_reduceScatter_ + hcom_allGather_ |
| DispatchFFNCombine锛坈omposite 鍒嗚В锛?| 846,380 | 35.29% | TC 鍒嗚В涓?sub ops锛孭hase 2 浼樺寲 |
| TC 涓嶆ā鎷?| 47,483 | 1.98% | TransData/Transpose/Sampling 绛?|
| **鎬昏鐩?* | **2,350,574** | **98.02%** | 鈥?|

**缁撹**锛歰p_mapping 瑕嗙洊绔埌绔€楁椂 98.02%锛屾弧瓒?>90% 鐩爣銆?

---

## 浜屻€乄8A8 閲忓寲绠楀瓙閫愭潯楠岃瘉

### 2.1 QuantBatchMatmulV3锛?.81%锛?880 娆★級

**TC op**锛歚tensor_cast.static_quant_linear.default`
**鏄犲皠璺緞**锛歚aclnnWeightQuantBatchMatmulV2/V3` 鈫?`QuantBatchMatmulV3`

**Profiling 杈撳叆 shape锛團RACTAL_NZ 瑙ｇ爜鍚庯級**锛?

| 璋冪敤娆℃暟 | 婵€娲?(M,K) | 鏉冮噸 ND (K,N) | 骞冲潎鑰楁椂 | 灞傛帹鏂?|
|---------|-----------|-------------|---------|--------|
| 928 | (1, 7168) | (7168, 4096) | 32.39 us | shared expert gate+up 鎴?W_O锛圱P=4锛?|
| 976 | (1, 7168) | (7168, 2112) | 28.05 us | 寰?TC trace 纭锛圢=2112 闈炴爣鍑嗙淮搴︼級 |
| 976 | (8, 2048) | (2048, 7168) | 21.62 us | 璺敱涓撳 down proj锛圡=top_k=8锛?|
| 928 | (1, 2048) | (2048, 7168) | 20.29 us | 璺敱涓撳 down proj decode锛圡=1锛?|
| 976 | (4, 1536) | (1536, 3072) | 15.72 us | MLA Q up proj锛圞=q_lora_rank=1536锛孨=3072 寰呯‘璁わ級 |
| 48 | (8, 7168) | (7168, 4608) | 34.95 us | Prefill 鍦烘櫙锛圡=8锛?|
| 48 | (8, 2304) | (2304, 7168) | 22.07 us | Prefill 鍦烘櫙 |

**FRACTAL_NZ 瑙ｇ爜瑙勫垯楠岃瘉**锛?
- 鏍煎紡 `(a, b, bh, bw)` 鈫?ND `(K=b脳bh, N=a脳bw)`
- 绀轰緥锛歚(66, 448, 16, 32)` 鈫?K=448脳16=7168, N=66脳32=2112 鉁?
- 绀轰緥锛歚(128, 448, 16, 32)` 鈫?K=448脳16=7168, N=128脳32=4096 鉁?

**楠岃瘉鐘舵€?*锛?
- FRACTAL_NZ 瑙ｇ爜瑙勫垯宸茬‘璁わ紝`profiling_data_source.py` 鐨?`fractal_nz_to_nd()` 閫傜敤
- N=2112 鍜?N=3072 鐨勫眰鏄犲皠闇€ TC dispatch trace 纭锛圕8 閬楃暀椤癸級
- 涓昏 shape锛?168脳4096, 2048脳7168锛変笌 DSV3 鏋舵瀯鍚诲悎

### 2.2 AscendQuantV2锛?.16%锛?928 娆★級

**TC op**锛歚tensor_cast.quantize.default`
**鏄犲皠璺緞**锛歚aclnnAscendQuant/V3` 鈫?`AscendQuantV2`

**Profiling 杈撳叆 shape**锛?

| 璋冪敤娆℃暟 | 杈撳叆 (M, K) | 骞冲潎鑰楁椂 | 灞傛帹鏂?|
|---------|-----------|---------|--------|
| 976 | (1, 7168) | 9.47 us | 婵€娲婚噺鍖栵紙hidden=7168锛?|
| 976 | (4, 1536) | 鈥?| MLA Q lora 婵€娲婚噺鍖?|
| 976 | (8, 2048) | 鈥?| 涓撳婵€娲婚噺鍖?|

**楠岃瘉鐘舵€?*锛?
- shape 鏍煎紡涓?ND锛屾棤 FRACTAL_NZ 杞崲锛孴C 鐩存帴鍖归厤 鉁?
- 3 绉?shape 瀵瑰簲 3 绫婚噺鍖栫偣锛坔idden/q_lora/expert锛夛紝瑕嗙洊瀹屾暣

### 2.3 DynamicQuant锛?.28%锛?904 娆★級

**TC op**锛歚tensor_cast.dynamic_quantize_symmetric.default`
**鏄犲皠璺緞**锛歚aclnnDynamicQuant/V2` 鈫?`DynamicQuant`

**Profiling 杈撳叆 shape**锛?

| 璋冪敤娆℃暟 | 杈撳叆 (M, K) | 灞傛帹鏂?|
|---------|-----------|--------|
| 928 | (1, 7168) | 鍔ㄦ€侀噺鍖栵紙hidden锛?|
| 928 | (1, 2048) | 鍔ㄦ€侀噺鍖栵紙expert intermediate锛?|
| 48 | (8, 2304) | Prefill 鍦烘櫙 |

**楠岃瘉鐘舵€?*锛歴hape 涓?ND锛孴C 鐩存帴鍖归厤 鉁?

### 2.4 AddRmsNormDynamicQuant锛?.01%锛?8 娆★級

**TC op**锛歚tensor_cast.add_rms_norm_dynamic_quant_symmetric.default`
**鏄犲皠璺緞**锛歚aclnnAddRmsNormDynamicQuantV2` 鈫?`AddRmsNormDynamicQuant`

**Profiling 杈撳叆 shape**锛歚(1,7168)+(1,7168)+(7168)+(7168)` 鈥?4 涓?ND 杈撳叆 鉁?

**楠岃瘉鐘舵€?*锛歴hape 鍖归厤锛岃皟鐢ㄦ鏁板皯锛?8娆★級锛岃€楁椂鍗犳瘮 0.01%锛屼綆浼樺厛绾?

---

## 涓夈€丮LA 涓撶敤绠楀瓙楠岃瘉

### 3.1 FusedInferAttentionScore锛?.15%锛?76 娆★級

**TC op**锛歚tensor_cast.attention_quant.default`锛坄query_mode: attention_special`锛?

**Profiling 杈撳叆 shape**锛堝敮涓€ shape锛?76 娆★級锛?
```
Q:  (4, 16, 1, 512)   鈥?batch=4, heads=16/TP, seq=1, head_dim=512
K:  (892, 1, 128, 512) 鈥?kv_seq=892, 1, kv_heads=128, head_dim=512
V:  (892, 1, 128, 512) 鈥?鍚?K
...锛堝悗缁负 mask/scale 绛夊彲閫夊弬鏁帮級
output_Q: (4, 16, 1, 64) 鈥?batch=4, heads=16/TP, seq=1, qk_rope_head_dim=64
output_K: (892, 1, 128, 64)
```

**MLA 鐗规畩鎬?*锛?
- Q head_dim=512锛堥潪鏍囧噯锛? kv_lora_rank=512锛?
- K/V head_dim=512锛坘v_lora_rank锛岄潪 v_head_dim=128锛?
- 杈撳嚭 Q/K head_dim=64锛坬k_rope_head_dim锛?
- 杩欐槸 MLA absorb 鍚庣殑 attention锛屼笌鏍囧噯 MHA 缁村害瀹屽叏涓嶅悓

**楠岃瘉鐘舵€?*锛?
- `attention_special` 鏌ヨ璺緞闇€姝ｇ‘澶勭悊闈炲绉?head_dim锛圦=512, output=64锛?
- FusedInferAttentionScore.csv 涓殑 shape 绱㈠紩缁村害闇€鍖呭惈 MLA 鐨?kv_lora_rank
- **椋庨櫓**锛氳嫢 CSV 浠呮湁鏍囧噯 MHA shape锛孧LA 鍦烘櫙浼?MISS 鈫?fallback analytic

### 3.2 TransposeBatchMatMul锛?.51%锛?76 娆★級

**TC op**锛歚aten.bmm.default`锛坅lternate: TransposeBatchMatMul锛?

**Profiling shape**锛歚(16, 4, 512)` 脳 `(16, 512, 128)` 鈫?output `(16, 4, 128)`
- 瑙ｈ锛?6 heads, batch=4, K=512(kv_lora_rank), N=128(v_head_dim)
- 瀵瑰簲 MLA 涓?attn_output @ W_UV锛坅bsorb projection锛?

**楠岃瘉鐘舵€?*锛?
- shape 涓?ND锛屾棤鏍煎紡杞崲
- alternate 鏌ヨ锛氬厛鏌?BatchMatMulV2.csv锛岃嫢 shape 涓嶅尮閰嶅啀鏌?TransposeBatchMatMul.csv
- 闇€纭 TransposeBatchMatMul.csv 涓湁 `(16,4,512)脳(16,512,128)` 鐨勬暟鎹?

### 3.3 BatchMatMulV2锛?.34%锛?76 娆★級

**TC op**锛歚aten.bmm.default`

**Profiling shape**锛歚(16, 4, 128)` 脳 `(16, 128, 512)` 鈫?output `(16, 4, 512)`
- 瑙ｈ锛?6 heads, batch=4, K=128(v_head_dim), N=512(kv_lora_rank)
- 瀵瑰簲 MLA 涓?absorb projection锛圔MM 閮ㄥ垎锛?

**楠岃瘉鐘舵€?*锛歴hape 涓?ND锛孴C 鐩存帴鍖归厤 鉁?

### 3.4 InterleaveRope锛?.32%锛?76 娆★級

**TC op**锛歚tensor_cast.apply_rope.default`锛坅lternate: InterleaveRope锛?

**Profiling shape**锛歚(4, 16, 1, 64)` 脳 `(4, 1, 1, 64)` 脳 `(4, 1, 1, 64)`
- 鏍煎紡锛歚(batch, heads/TP, seq, qk_rope_head_dim/2)`
- 娉ㄦ剰锛歨ead_dim=64 = qk_rope_head_dim锛岄潪瀹屾暣 head_dim

**楠岃瘉鐘舵€?*锛?
- TC apply_rope 杈撳嚭 shape 涓?`(batch, heads, seq, head_dim)`锛屼笌 profiling 鐨?`head_dim=64` 涓嶅悓
- alternate 鏌ヨ鏃堕渶纭 shape 鍖归厤瑙勫垯鑳藉鐞?rope_head_dim 缁村害宸紓
- **椋庨櫓**锛歴hape 涓嶅尮閰嶅彲鑳藉鑷?MISS

### 3.5 KvRmsNormRopeCache锛?.20%锛?76 娆★級

**TC op**锛歚tensor_cast.kv_rmsnorm_rope_cache.default`

**Profiling shape**锛?
```
input:  (4, 1, 1, 576)  鈥?batch=4, 1, seq=1, kv_lora_rank+rope_dim=576
norm_w: (512,)           鈥?kv_lora_rank=512
q_out:  (4, 1, 1, 64)   鈥?qk_rope_head_dim=64
k_out:  (4, 1, 1, 64)
seq_len: (4,)
kv_cache: (892, 128, 1, 64)  鈥?kv_seq, kv_heads, 1, rope_dim
kv_cache2: (892, 128, 1, 512) 鈥?kv_seq, kv_heads, 1, kv_lora_rank
```

**楠岃瘉鐘舵€?*锛?
- 杈撳叆 576 = kv_lora_rank(512) + qk_rope_head_dim(64) 鉁?
- shape 涓?ND锛孴C 鐩存帴鍖归厤
- KvRmsNormRopeCache.csv 闇€鍖呭惈姝?shape 鐨勬暟鎹?

---

## 鍥涖€侀€氫俊绠楀瓙楠岃瘉

### 4.1 hcom_reduceScatter_锛?7.92%锛?082 娆★級

**TC op**锛歚tensor_cast.reduce_scatter.default`锛圕7 淇涓轰富 kernel_type锛?

**Profiling shape**锛歂/A锛圚CCL 閫氫俊绠楀瓙鏃?shape 璁板綍锛?
**骞冲潎鑰楁椂**锛?21.59 us/娆?

**楠岃瘉鐘舵€?*锛?
- kernel_type 宸蹭慨姝ｄ负 hcom_reduceScatter_锛圕7锛?
- 闇€ C10 閲囬泦 `hcom_reduceScatter_.csv`锛坢essage_bytes 脳 topology_tier 缃戞牸锛?
- 閫氫俊鏌ヨ璺緞锛歚_lookup_comm()` 鎸?message_bytes + topology_tier 鏌ヨ

### 4.2 hcom_allGather_锛?4.16%锛?164 娆★級

**TC op**锛歚tensor_cast.all_gather.default`

**骞冲潎鑰楁椂**锛?39.12 us/娆★紙绾︿负 reduce_scatter 鐨?43%锛岀鍚?allGather 鏁版嵁閲忔洿灏忕殑棰勬湡锛?

**楠岃瘉鐘舵€?*锛氬凡閰嶇疆锛岄渶 C10 閲囬泦 `hcom_allGather_.csv`

---

## 浜斻€丏ispatchFFNCombine 楠岃瘉锛堥噸鐐归闄╋級

**鑰楁椂**锛?46,380 us锛?5.29%锛屽钩鍧?912 us/娆★級

**褰撳墠澶勭悊**锛歍C 灏嗗叾鍒嗚В涓?sub ops锛屽悇 sub op 閫氳繃 alternate_kernel_types 鍥為€€鍒?DispatchFFNCombine.csv

**宸茬煡椋庨櫓**锛?
- 鑻?GroupedMatmul/MoeDistributeDispatchV2 绛?CSV 涓嶅瓨鍦紝4 涓?TC ops 鍧囨煡 DispatchFFNCombine.csv 鈫?閲嶅璁℃暟 4脳
- 鑻?DispatchFFNCombine.csv 涔熶笉瀛樺湪 鈫?鍏ㄩ儴 fallback analytic 鈫?35.3% 鑰楁椂浼扮畻涓嶅噯

**Phase 2 琛屽姩椤?*锛圚1/E3锛夛細
1. 涓?DispatchFFNCombine 閲囬泦 microbenchmark锛坄torch.ops._C_ascend.dispatch_ffn_combine`锛?
2. 浠呬繚鐣?`grouped_matmul_quant_swiglu` 鐨?DispatchFFNCombine alternate
3. 鍏朵綑 sub ops 绉婚櫎 alternate锛屾敼鐢?analytic fallback

---

## 鍏€侀獙璇佺粨璁轰笌閬楃暀椤?

### 宸查獙璇侊紙鍙洿鎺ヤ娇鐢級

| kernel Type | 楠岃瘉鐘舵€?| 缃俊搴?|
|------------|---------|--------|
| QuantBatchMatmulV3 | FRACTAL_NZ 瑙ｇ爜瑙勫垯纭 | 楂?|
| AscendQuantV2 | ND shape锛岀洿鎺ュ尮閰?| 楂?|
| DynamicQuant | ND shape锛岀洿鎺ュ尮閰?| 楂?|
| AddRmsNormBias | ND shape锛岀洿鎺ュ尮閰?| 楂?|
| BatchMatMulV2 | ND shape锛岀洿鎺ュ尮閰?| 楂?|
| KvRmsNormRopeCache | ND shape锛?76=512+64 纭 | 楂?|
| hcom_reduceScatter_ | kernel_type 宸蹭慨姝ｏ紙C7锛?| 楂?|
| hcom_allGather_ | 宸查厤缃?| 楂?|

### 閬楃暀楠岃瘉椤癸紙闇€ TC dispatch trace锛?

| 椤圭洰 | 椋庨櫓绛夌骇 | 璇存槑 |
|------|---------|------|
| QuantBatchMatmulV3 N=2112/3072 灞傛槧灏?| 浣?| 涓嶅奖鍝?CSV 鏌ヨ锛屼粎褰卞搷 shape 鐞嗚В |
| FusedInferAttentionScore MLA shape | 楂?| CSV 闇€鍖呭惈 MLA 鐨?kv_lora_rank=512 缁村害 |
| InterleaveRope head_dim=64 鍖归厤 | 涓?| alternate 鏌ヨ鏃?shape 宸紓鍙兘瀵艰嚧 MISS |
| TransposeBatchMatMul CSV 鏁版嵁 | 涓?| 闇€纭 CSV 涓湁 (16,4,512)脳(16,512,128) |
| DispatchFFNCombine 閲嶅璁℃暟 | 楂?| Phase 2 蹇呴』瑙ｅ喅锛屽惁鍒?MoE 鑰楁椂浼扮畻鍋忛珮 4脳 |

### 涓嬩竴姝ヨ鍔?

1. **C10**锛?.13锛夛細閲囬泦 `hcom_reduceScatter_.csv` 鍜?`hcom_allGather_.csv`
2. **Phase 2 H1**锛?.16锛夛細TC dispatch trace 纭 QuantBatchMatmulV3 灞傛槧灏?
3. **Phase 2 E3**锛?.19锛夛細DispatchFFNCombine microbenchmark 閲囬泦 + alternate 绛栫暐淇
4. **Phase 2 E4**锛?.20锛夛細FusedInferAttentionScore MLA shape microbenchmark 琛ュ厖

