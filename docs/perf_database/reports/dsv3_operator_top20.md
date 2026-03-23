# DeepSeek-V3 绠楀瓙鑰楁椂 Top-20 娓呭崟

**鏁版嵁鏉ユ簮**: `kernel_details_deepseekv3-cann85.csv`
**纭欢**: ATLAS_800_A3_752T_128G_DIE锛?2 鍗★紝W8A8 aclgraph
**CANN 鐗堟湰**: 8.5
**璐熻矗浜?*: HDY
**鏃ユ湡**: 2026-03-10

---

## 缁熻姒傝

- **鎬昏€楁椂**: 2,398,057 us锛?.398 s锛?
- **kernel Type 绉嶆暟**: 43 绉?
- **Top-20 绱瑕嗙洊**: 99.30%

---

## Top-20 绠楀瓙鑰楁椂鎺掑悕

| 鎺掑悕 | kernel Type | 璋冪敤娆℃暟 | 鎬昏€楁椂 (us) | 鍗犳瘮 | 绱鍗犳瘮 | 绫诲埆 | 澶囨敞 |
|------|------------|--------:|----------:|-----:|--------:|------|------|
| 1 | DispatchFFNCombine | 928 | 846,381 | 35.29% | 35.29% | 璁＄畻+閫氫俊铻嶅悎 | MoE 瓒呯骇铻嶅悎 op锛歛ll_to_all脳2 + GMM脳2 + SwiGlu + routing |
| 2 | hcom_reduceScatter_ | 2,082 | 669,552 | 27.92% | 63.22% | 閫氫俊 | TP reduce-scatter |
| 3 | hcom_allGather_ | 4,164 | 579,302 | 24.16% | 87.37% | 閫氫俊 | TP all-gather |
| 4 | QuantBatchMatmulV3 | 4,880 | 115,447 | 4.81% | 92.19% | 璁＄畻 | W8A8 閲忓寲鐭╅樀涔橈紙涓诲姏 GEMM锛?|
| 5 | AscendQuantV2 | 2,928 | 27,732 | 1.16% | 93.34% | 璁＄畻 | 闈欐€侀噺鍖?|
| 6 | FusedInferAttentionScore | 976 | 27,542 | 1.15% | 94.49% | 璁＄畻 | MLA attention |
| 7 | TransData | 976 | 16,609 | 0.69% | 95.18% | 鏍煎紡杞崲 | ND 鈫?FRACTAL_NZ锛孴C 涓嶆ā鎷?|
| 8 | Transpose | 992 | 16,572 | 0.69% | 95.87% | 璁＄畻 | 鏁版嵁甯冨眬杞崲 |
| 9 | MatMulV2 | 944 | 14,172 | 0.59% | 96.47% | 璁＄畻 | BF16 鐭╅樀涔橈紙shared expert / lm_head锛?|
| 10 | TransposeBatchMatMul | 976 | 12,226 | 0.51% | 96.98% | 璁＄畻 | MLA attn_output @ W_UV |
| 11 | BatchMatMulV2 | 976 | 8,272 | 0.34% | 97.32% | 璁＄畻 | MLA BMM锛坅bsorb projection锛?|
| 12 | InterleaveRope | 976 | 7,669 | 0.32% | 97.64% | 璁＄畻 | DeepSeek interleave RoPE |
| 13 | AddRmsNormBias | 1,904 | 7,410 | 0.31% | 97.95% | 璁＄畻 | Add + RmsNorm 铻嶅悎锛圕ANN 8.5锛?|
| 14 | DynamicQuant | 1,904 | 6,608 | 0.28% | 98.23% | 璁＄畻 | 鍔ㄦ€侀噺鍖?|
| 15 | MoeGatingTopK | 928 | 5,109 | 0.21% | 98.44% | 璁＄畻 | MoE gating + top-k routing |
| 16 | KvRmsNormRopeCache | 976 | 4,832 | 0.20% | 98.64% | 璁＄畻 | MLA KV norm + RoPE + cache 铻嶅悎 |
| 17 | Add | 1,936 | 4,410 | 0.18% | 98.82% | 璁＄畻 | 娈嬪樊鍔犳硶 |
| 18 | RmsNorm | 992 | 4,012 | 0.17% | 98.99% | 璁＄畻 | 鐙珛 RmsNorm锛堟湭铻嶅悎灞傦級 |
| 19 | AsStrided | 976 | 3,871 | 0.16% | 99.15% | 鍐呭瓨 | stride 鎿嶄綔锛孴C 涓嶆ā鎷?|
| 20 | SwiGlu | 976 | 3,443 | 0.14% | 99.30% | 璁＄畻 | shared expert SwiGlu 婵€娲?|

---

## 鍏抽敭瑙傚療

### 鑰楁椂鍒嗗竷

- **閫氫俊涓诲**锛歨com_reduceScatter_ + hcom_allGather_ 鍚堣鍗?**52.1%**锛孴P 閫氫俊鏄渶澶х摱棰?
- **DispatchFFNCombine 寮傚父楂?*锛氬崟涓瀺鍚?kernel 鍗?**35.3%**锛屾槸 DSV3 MoE EP 璺敱 + FFN 鐨勫叏閮ㄨ€楁椂
- **涓夎€呭悎璁?87.4%**锛氬墠 3 鍚嶅凡瑕嗙洊缁濆ぇ閮ㄥ垎鑰楁椂锛屽叾浣?40 绉?kernel 鍚堣浠?12.6%

### 璁＄畻+閫氫俊铻嶅悎绠楀瓙纭

| 铻嶅悎绫诲瀷 | kernel Type | 鏄惁瀛樺湪 | 澶勭悊鏂瑰紡 |
|---------|------------|---------|---------|
| MC2锛圡atMul + AllReduce锛?| 鏃犱笓鐢?kernel | 鍚?| `composite` 鍒嗚В锛歚QuantBatchMatmulV3` + `hcom_allReduce_` |
| MoE EP 铻嶅悎 | DispatchFFNCombine | **鏄?*锛?5.3%锛?| `composite` 鍒嗚В锛歚permute_tokens + grouped_matmul脳2 + swiglu + unpermute_tokens + all_to_all脳2` |

### op_mapping 瑕嗙洊鐘舵€侊紙C7 鍙傝€冿級

| kernel Type | TC op 鏄犲皠 | 鐘舵€?|
|------------|-----------|------|
| DispatchFFNCombine | `tensor_cast.grouped_matmul_quant_swiglu` + `permute/unpermute_tokens` | composite锛屽凡閰嶇疆 |
| hcom_reduceScatter_ | `tensor_cast.reduce_scatter` | 宸查厤缃?|
| hcom_allGather_ | `tensor_cast.all_gather` | 宸查厤缃?|
| QuantBatchMatmulV3 | `tensor_cast.static_quant_linear` | 宸查厤缃?|
| AscendQuantV2 | `tensor_cast.quantize` | 宸查厤缃?|
| FusedInferAttentionScore | `tensor_cast.attention_quant` | 宸查厤缃紙attention_special锛?|
| TransData | 鈥?| TC 涓嶆ā鎷燂紝CANN 鍐呴儴鏍煎紡杞崲 |
| Transpose | 鈥?| TC 涓嶆ā鎷?|
| MatMulV2 | `aten.mm.default` | 宸查厤缃?|
| TransposeBatchMatMul | `aten.bmm.default` | alternate锛屽凡閰嶇疆 |
| BatchMatMulV2 | `aten.bmm.default` | 宸查厤缃?|
| InterleaveRope | `tensor_cast.apply_rope` | alternate锛屽凡閰嶇疆 |
| AddRmsNormBias | `tensor_cast.add_rms_norm` | 宸查厤缃紙CANN 8.5锛?|
| DynamicQuant | `tensor_cast.dynamic_quantize_*` | 宸查厤缃?|
| MoeGatingTopK | `tensor_cast.moe_gating_topk` | 宸查厤缃?|
| KvRmsNormRopeCache | `tensor_cast.kv_rmsnorm_rope_cache` | 宸查厤缃?|
| Add | `aten.add.Tensor` | 宸查厤缃?|
| RmsNorm | `tensor_cast.rms_norm` | 宸查厤缃?|
| AsStrided | 鈥?| TC 涓嶆ā鎷?|
| SwiGlu | `tensor_cast.swiglu` | 宸查厤缃?|

