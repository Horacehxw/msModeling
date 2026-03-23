# DeepSeek-V3 W8A8 op_mapping 楠岃瘉鎶ュ憡锛圕7锛?

**鏁版嵁鏉ユ簮**: `kernel_details_deepseekv3-cann85.csv`
**纭欢**: ATLAS_800_A3_752T_128G_DIE锛?2 鍗★紝W8A8 aclgraph
**CANN 鐗堟湰**: 8.5 / vLLM-ascend 0.15.0
**op_mapping 璺緞**: `tensor_cast/performance_model/perf_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5/op_mapping.yaml`
**璐熻矗浜?*: HDY
**鏃ユ湡**: 2026-03-11

---

## 缁熻姒傝

- **鎬昏€楁椂**: 2,398,057 us锛?.398 s锛?
- **kernel Type 绉嶆暟**: 43 绉?
- **Top-15 绱瑕嗙洊**: 99.13%
- **Top-15 涓?TC 鍙ā鎷?*: 12 绉嶏紙鎺掗櫎 TransData/Transpose/DispatchFFNCombine 鐗规畩澶勭悊锛?

---

## DSV3 Top-15 op_mapping 瑕嗙洊鐘舵€?

| 鎺掑悕 | kernel Type | 鑰楁椂鍗犳瘮 | TC op 鏄犲皠 | 鐘舵€?| 澶囨敞 |
|------|------------|---------|-----------|------|------|
| 1 | DispatchFFNCombine | 35.29% | composite 鍒嗚В锛堣涓嬶級 | 鐗规畩澶勭悊 | 瓒呯骇铻嶅悎 MoE op锛孴C 鍒嗚В涓哄涓?sub ops |
| 2 | hcom_reduceScatter_ | 27.92% | `tensor_cast.reduce_scatter.default` | 宸查厤缃紙涓伙級 | C7 淇锛氳涓轰富 kernel_type |
| 3 | hcom_allGather_ | 24.16% | `tensor_cast.all_gather.default` | 宸查厤缃?| 鈥?|
| 4 | QuantBatchMatmulV3 | 4.81% | `tensor_cast.static_quant_linear.default` | 宸查厤缃?| W8A8 涓诲姏 GEMM |
| 5 | AscendQuantV2 | 1.16% | `tensor_cast.quantize.default` | 宸查厤缃?| 闈欐€侀噺鍖?|
| 6 | FusedInferAttentionScore | 1.15% | `tensor_cast.attention_quant.default` | 宸查厤缃?| MLA attention锛宎ttention_special 鏌ヨ |
| 7 | TransData | 0.69% | `profiling.TransData` | TC 涓嶆ā鎷?| CANN 鍐呴儴鏍煎紡杞崲锛屼笉褰卞搷浼扮畻 |
| 8 | Transpose | 0.69% | `profiling.Transpose` | TC 涓嶆ā鎷?| 鏁版嵁甯冨眬杞崲锛屼笉褰卞搷浼扮畻 |
| 9 | MatMulV2 | 0.59% | `aten.mm.default` | 宸查厤缃?| shared expert / lm_head BF16 matmul |
| 10 | TransposeBatchMatMul | 0.51% | `aten.bmm.default`锛坅lternate锛?| 宸查厤缃?| MLA attn_output @ W_UV |
| 11 | BatchMatMulV2 | 0.34% | `aten.bmm.default` | 宸查厤缃?| MLA BMM锛坅bsorb projection锛?|
| 12 | InterleaveRope | 0.32% | `tensor_cast.apply_rope.default`锛坅lternate锛?| 宸查厤缃?| DeepSeek interleave RoPE |
| 13 | AddRmsNormBias | 0.31% | `tensor_cast.add_rms_norm.default` | 宸查厤缃?| CANN 8.5 fused norm |
| 14 | DynamicQuant | 0.28% | `tensor_cast.dynamic_quantize_symmetric.default` | 宸查厤缃?| 鍔ㄦ€侀噺鍖?|
| 15 | MoeGatingTopK | 0.21% | `tensor_cast.moe_gating_topk.default` | 宸查厤缃?| MoE gating + top-k routing |

**Top-15 瑕嗙洊鐜?*锛?3/15 鏈夌洿鎺?TC op 鏄犲皠锛?7%锛夛紝2 绉?TC 涓嶆ā鎷燂紙TransData/Transpose锛屽悎璁?1.38%锛?

---

## C7 淇敼璁板綍

### 淇敼1锛歳educe_scatter kernel_type 涓绘鍏崇郴璋冩暣

**闂**锛氬師閰嶇疆 `kernel_type: HcomReduceScatter`锛屼絾 DSV3 CANN 8.5 profiling 涓疄闄呭嚭鐜扮殑鏄?`hcom_reduceScatter_`锛?082娆★紝27.9%锛夛紝HcomReduceScatter 鏈嚭鐜般€?

**淇敼**锛?
```yaml
# 淇敼鍓?
"tensor_cast.reduce_scatter.default":
    kernel_type: HcomReduceScatter
    alternate_kernel_types: [hcom_reduceScatter_]

# 淇敼鍚?
"tensor_cast.reduce_scatter.default":
    kernel_type: hcom_reduceScatter_
    alternate_kernel_types: [HcomReduceScatter]
```

**褰卞搷**锛氭煡璇㈡椂浼樺厛浣跨敤 `hcom_reduceScatter_.csv`锛屼笌 DSV3 profiling 瀹炴祴瀵归綈銆?

### 淇敼2锛欴ispatchFFNCombine notes 琛ュ厖

鍦?`profiling.DispatchFFNCombine` 涓ˉ鍏呬簡鏌ヨ绛栫暐璇存槑鍜?Phase 2 寤鸿锛堣涓嬭妭锛夈€?

---

## DispatchFFNCombine 澶勭悊绛栫暐锛堥噸鐐癸級

### 鐜扮姸

DispatchFFNCombine 鏄?DSV3 鏈€閲嶈鐨勫崟涓€ kernel锛?5.3%锛夛紝铻嶅悎浜嗭細
- `InitRouting + DispatchV2`锛坅ll_to_all dispatch锛?
- `2脳GroupedMatmul`锛坓ate-up proj + down proj锛?
- `SwiGlu`
- `CombineV2 + Unpermute`锛坅ll_to_all combine锛?

TC 灏嗗叾鍒嗚В涓虹嫭绔?ops锛?
```
permute_tokens 鈫?all_to_all 鈫?grouped_matmul_quant_swiglu 鈫?grouped_matmul_quant 鈫?all_to_all 鈫?unpermute_tokens
```

### 褰撳墠 op_mapping 閰嶇疆

澶氫釜 TC ops 璁剧疆浜?`alternate_kernel_types: [DispatchFFNCombine]`锛?
- `tensor_cast.grouped_matmul_quant.default`
- `tensor_cast.grouped_matmul_quant_swiglu.default`
- `tensor_cast.permute_tokens.default`
- `tensor_cast.unpermute_tokens.default`

### 宸茬煡闂

鑻?GroupedMatmul/MoeDistributeDispatchV2 绛?CSV 涓嶅瓨鍦紝涓婅堪 4 涓?TC ops 鍧囦細 fallback 鍒?`DispatchFFNCombine.csv`锛屽鑷?DispatchFFNCombine 鑰楁椂琚绠?4 娆★紙閲嶅璁℃暟锛夈€?

### Phase 2 寤鸿锛圚1/E3 闃舵锛?

1. 涓?DispatchFFNCombine 鍗曠嫭閲囬泦 microbenchmark锛坄microbenchmark generation tool` 鏀寔 `torch.ops._C_ascend.dispatch_ffn_combine`锛?
2. 浠呬繚鐣?`grouped_matmul_quant_swiglu` 鐨?DispatchFFNCombine alternate锛堝畠鏄渶涓昏鐨勮绠楅儴鍒嗭級
3. 鍏朵綑 sub ops锛坧ermute_tokens, unpermute_tokens, grouped_matmul_quant锛夌Щ闄?DispatchFFNCombine alternate锛屾敼鐢?analytic fallback
4. 鎴栬€咃細涓?DispatchFFNCombine 娣诲姞 composite 鍒嗚В姣斾緥閰嶇疆锛堥渶瑕?profiling 鏁版嵁鏀拺锛?

---

## 棰濆瑕嗙洊锛圱op-15 涔嬪锛?

| kernel Type | 鑰楁椂鍗犳瘮 | TC op 鏄犲皠 | 鐘舵€?|
|------------|---------|-----------|------|
| KvRmsNormRopeCache | 0.20% | `tensor_cast.kv_rmsnorm_rope_cache.default` | 宸查厤缃紙MLA 涓撶敤锛?|
| Add | 0.18% | `aten.add.Tensor` | 宸查厤缃?|
| RmsNorm | 0.17% | `tensor_cast.rms_norm.default` | 宸查厤缃?|
| SwiGlu | 0.14% | `tensor_cast.swiglu.default` | 宸查厤缃?|
| Cast | 0.12% | `aten.to.dtype` | 宸查厤缃?|
| AddRmsNormDynamicQuant | 0.01% | `tensor_cast.add_rms_norm_dynamic_quant_symmetric.default` | 宸查厤缃?|

---

## 瑕嗙洊鐜囨眹鎬?

| 绫诲埆 | kernel 鏁?| 鑰楁椂鍗犳瘮 | 鐘舵€?|
|------|----------|---------|------|
| 宸茬洿鎺ユ槧灏勶紙TC op 鈫?kernel_type锛?| 12 | 61.8% | 鍙煡璇?|
| DispatchFFNCombine锛坈omposite 鍒嗚В锛?| 1 | 35.3% | 鐗规畩澶勭悊锛孭hase 2 浼樺寲 |
| TC 涓嶆ā鎷燂紙TransData/Transpose/AsStrided锛?| 3 | 1.5% | 涓嶅奖鍝嶄及绠?|
| **鍚堣 Top-15** | **15** | **99.1%** | 鈥?|

**缁撹**锛欴SV3 Top-15 op_mapping 瑕嗙洊瀹屾暣銆備富瑕侀闄╂槸 DispatchFFNCombine 鐨勯噸澶嶈鏁伴棶棰橈紝闇€鍦?Phase 2 瑙ｅ喅銆?

---

## 娉ㄦ剰浜嬮」锛圕8 楠岃瘉閲嶇偣锛?

1. **QuantBatchMatmulV3 shape 宸紓**锛?
   - Profiling 杈撳叆锛歚(1,7168)` 脳 `(66,448,16,32)` 鈥?绗簩涓緭鍏ヤ负 FRACTAL_NZ 鏍煎紡
   - TC 杈撳嚭锛歚(M, K)` 脳 `(K, N)` 鈥?ND 鏍煎紡
   - 闇€纭 `profiling_data_source.py` 鐨?`fractal_nz_to_nd()` 鑳芥纭鐞?W8A8 閲忓寲鐭╅樀

2. **FusedInferAttentionScore锛圡LA锛?*锛?
   - Profiling 杈撳叆锛歚(4,16,1,512)` Q + `(892,1,128,512)` K + `(892,1,128,512)` V
   - MLA 鐨?KV 缁村害涓庢爣鍑?MHA 涓嶅悓锛坔ead_dim=512 for KV, 64 for Q output锛?
   - 闇€纭 attention_special 鏌ヨ璺緞鑳芥纭鐞?MLA 鐨勯潪瀵圭О head_dim

3. **InterleaveRope shape**锛?
   - Profiling锛歚(4,16,1,64)` 鈥?鏍煎紡涓?`(batch, heads, seq, head_dim/2)`
   - TC apply_rope锛歚(batch, heads, seq, head_dim)` 鈥?闇€纭 alternate 鏌ヨ鏃?shape 鍖归厤瑙勫垯

4. **hcom_reduceScatter_ CSV 鏂囦欢**锛?
   - 闇€纭 HCCL 鏁版嵁鐩綍涓瓨鍦?`hcom_reduceScatter_.csv`锛圕10 閲囬泦浜х墿锛?
   - 鏂囦欢鍚嶅ぇ灏忓啓鏁忔劅锛岄渶涓?kernel_type 瀹屽叏涓€鑷?

