# Qwen3-32B 绠楀瓙鑰楁椂 Top-20 娓呭崟

**鏁版嵁鏉ユ簮**: `kernel_details_qwen3-32b_cann85.csv`
**纭欢**: ATLAS_800_A3_752T_128G_DIE锛?6 鍗★紝BF16 aclgraph
**CANN 鐗堟湰**: 8.5
**璐熻矗浜?*: ZZY
**鏃ユ湡**: 2026-03-10

---

## 缁熻姒傝

- **鎬昏€楁椂**: 4,461,888 us锛?.462 s锛?
- **kernel Type 绉嶆暟**: 38 绉?
- **鎬?kernel 璋冪敤娆℃暟**: 28,641
- **Top-20 绱瑕嗙洊**: 99.76%

---

## Top-20 绠楀瓙鑰楁椂鎺掑悕

| 鎺掑悕 | kernel Type | 璋冪敤娆℃暟 | 鎬昏€楁椂 (us) | 鍗犳瘮 | 绱鍗犳瘮 | 绫诲埆 | 澶囨敞 |
|------|------------|--------:|----------:|-----:|--------:|------|------|
| 1 | hcom_allReduce_ | 7,748 | 3,755,792 | 84.17% | 84.17% | 閫氫俊 | TP all-reduce锛岀粷瀵逛富瀵?|
| 2 | MatMulV2 | 7,710 | 245,135 | 5.49% | 89.67% | 璁＄畻 | BF16 鐭╅樀涔橈紙涓诲姏 GEMM锛?|
| 3 | FusedInferAttentionScore | 1,920 | 118,485 | 2.66% | 92.32% | 璁＄畻 | Prefill + Decode attention |
| 4 | AddRmsNormBias | 3,840 | 70,568 | 1.58% | 93.91% | 璁＄畻 | Add + RmsNorm 铻嶅悎锛圕ANN 8.5锛?|
| 5 | DSARandomUniform | 30 | 54,614 | 1.22% | 95.13% | 鍏朵粬 | 闅忔満鏁扮敓鎴愶紝sampling pipeline锛孴C 涓嶆ā鎷?|
| 6 | Sort | 30 | 47,139 | 1.06% | 96.19% | 鍏朵粬 | Sampling top-k sort锛孴C 涓嶆ā鎷?|
| 7 | SwiGlu | 1,920 | 33,273 | 0.75% | 96.93% | 璁＄畻 | SwiGlu 婵€娲?|
| 8 | split_qkv_rmsnorm_rope_kernel | 1,890 | 31,102 | 0.70% | 97.63% | 璁＄畻 | Triton 铻嶅悎锛歈KV split + RmsNorm + RoPE锛?3/64 灞傦級 |
| 9 | hcom_allGather_ | 30 | 27,921 | 0.63% | 98.26% | 閫氫俊 | all-gather锛坰ampling 闃舵锛?|
| 10 | ReshapeAndCacheNdKernel | 1,920 | 23,929 | 0.54% | 98.79% | 璁＄畻 | KV cache 鍐欏叆 |
| 11 | allgatherAicpuKernel | 30 | 10,754 | 0.24% | 99.03% | 閫氫俊 | AICPU all-gather 鍙樹綋锛坓raph 缂栬瘧璺緞锛?|
| 12 | SoftmaxV2 | 30 | 6,273 | 0.14% | 99.17% | 鍏朵粬 | Sampling softmax锛孴C 涓嶆ā鎷?|
| 13 | RealDiv | 60 | 4,486 | 0.10% | 99.27% | 璁＄畻 | 闄ゆ硶 |
| 14 | MaskedFill | 60 | 4,387 | 0.10% | 99.37% | 鍏朵粬 | Sampling mask锛孴C 涓嶆ā鎷?|
| 15 | ArgMaxV2 | 30 | 4,281 | 0.10% | 99.47% | 鍏朵粬 | Sampling argmax锛孴C 涓嶆ā鎷?|
| 16 | Neg | 30 | 4,015 | 0.09% | 99.56% | 鍏朵粬 | Sampling 鍐呴儴锛孴C 涓嶆ā鎷?|
| 17 | ApplyTopKTopPCustom | 30 | 2,766 | 0.06% | 99.62% | 鍏朵粬 | vllm-ascend 鑷畾涔?sampling op锛孴C 涓嶆ā鎷?|
| 18 | Add | 30 | 2,412 | 0.05% | 99.67% | 璁＄畻 | 娈嬪樊鍔犳硶 |
| 19 | GreaterEqual | 60 | 2,094 | 0.05% | 99.72% | 鍏朵粬 | Sampling 姣旇緝锛孴C 涓嶆ā鎷?|
| 20 | Mul | 90 | 1,942 | 0.04% | 99.76% | 璁＄畻 | 鏍囬噺涔樻硶 |

---

## 鍏抽敭瑙傚療

### 鑰楁椂鍒嗗竷

- **閫氫俊缁濆涓诲**锛歚hcom_allReduce_` 鍗曢」鍗?**84.2%**锛屾槸 Qwen3-32B BF16 TP 鎺ㄧ悊鐨勬牳蹇冪摱棰?
- **璁＄畻渚ч泦涓?*锛歚MatMulV2`锛?.5%锛? `FusedInferAttentionScore`锛?.7%锛? `AddRmsNormBias`锛?.6%锛夊悎璁?9.8%
- **Sampling 寮€閿€涓嶅彲蹇借**锛歚DSARandomUniform`锛?.2%锛? `Sort`锛?.1%锛夊悎璁?2.3%锛屼絾 TC 涓嶆ā鎷?sampling锛屼笉褰卞搷鎺ㄧ悊鑰楁椂浼扮畻
- **涓?DSV3 瀵规瘮**锛歈wen3 鏃?MoE锛屾棤 `DispatchFFNCombine`锛岄€氫俊褰㈠紡涓?all-reduce锛圱P锛夛紝鑰岄潪 DSV3 鐨?reduce-scatter + all-gather锛圫P锛?

### 璁＄畻+閫氫俊铻嶅悎绠楀瓙纭

**缁撹锛歈wen3-32B Profiling 涓笉瀛樺湪璁＄畻+閫氫俊铻嶅悎绫?kernel銆?*

| 铻嶅悎绫诲瀷 | kernel Type | 鏄惁瀛樺湪 | 璇存槑 |
|---------|------------|---------|------|
| MC2锛圡atMul + AllReduce锛?| 鏃犱笓鐢?kernel | 鍚?| `MatMulV2` 鍜?`hcom_allReduce_` 鍒嗗紑璁板綍 |
| MoE EP 铻嶅悎锛圖ispatchFFNCombine锛?| 鏃?| 鍚?| Qwen3 鏃?MoE 缁撴瀯 |

`MatMulV2`锛?710 娆★級鍜?`hcom_allReduce_`锛?748 娆★級璋冪敤娆℃暟鍑犱箮鐩哥瓑锛屽嵃璇佷簡 TP 妯″紡涓嬫瘡涓?matmul 鍚庤窡涓€娆?all-reduce锛屼袱鑰呯嫭绔嬭褰曪紝鏃犺瀺鍚堛€?

### op_mapping 瑕嗙洊鐘舵€侊紙C3 鍙傝€冿級

| kernel Type | TC op 鏄犲皠 | 鐘舵€?|
|------------|-----------|------|
| hcom_allReduce_ | `tensor_cast.all_reduce` | 宸查厤缃?|
| MatMulV2 | `aten.mm.default` | 宸查厤缃?|
| FusedInferAttentionScore | `tensor_cast.attention` | 宸查厤缃紙attention_special锛?|
| AddRmsNormBias | `tensor_cast.add_rms_norm` / `add_rms_norm2` | 宸查厤缃紙CANN 8.5锛?|
| DSARandomUniform | 鈥?| TC 涓嶆ā鎷燂紙sampling锛?|
| Sort | 鈥?| TC 涓嶆ā鎷燂紙sampling锛?|
| SwiGlu | `tensor_cast.swiglu` | 宸查厤缃?|
| split_qkv_rmsnorm_rope_kernel | `tensor_cast.apply_rope`锛坈omposite锛?| Triton 铻嶅悎 kernel锛孴C 鍒嗚В涓?rms_norm + apply_rope |
| hcom_allGather_ | `tensor_cast.all_gather` | 宸查厤缃?|
| ReshapeAndCacheNdKernel | `tensor_cast.reshape_and_cache` | 宸查厤缃?|
| allgatherAicpuKernel | `tensor_cast.all_gather`锛坅lternate锛?| 宸查厤缃?|
| SoftmaxV2 | 鈥?| TC 涓嶆ā鎷燂紙sampling锛?|
| RealDiv | `aten.div.Tensor` | 宸查厤缃紙alternate RealDiv锛?|
| MaskedFill | 鈥?| TC 涓嶆ā鎷燂紙sampling锛?|
| ArgMaxV2 | 鈥?| TC 涓嶆ā鎷燂紙sampling锛?|
| Neg | 鈥?| TC 涓嶆ā鎷燂紙sampling锛?|
| ApplyTopKTopPCustom | 鈥?| TC 涓嶆ā鎷燂紙sampling锛?|
| Add | `aten.add.Tensor` | 宸查厤缃?|
| GreaterEqual | 鈥?| TC 涓嶆ā鎷燂紙sampling锛?|
| Mul | `aten.mul.Tensor` | 宸查厤缃?|

### 娉ㄦ剰浜嬮」锛圕3 楠岃瘉閲嶇偣锛?

- **`split_qkv_rmsnorm_rope_kernel`**锛堟帓鍚?8锛?.70%锛夛細vllm-ascend Triton 铻嶅悎 kernel锛孴C 鏃犲搴斿崟涓€ op锛屽垎瑙ｄ负 `rms_norm + apply_rope`銆傞渶纭鍒嗚В鍚庤€楁椂浼扮畻璇樊鏄惁鍙帴鍙?
- **`hcom_allReduce_` 鍗犳瘮寮傚父楂橈紙84%锛?*锛氬伐浣滆鍒掗闄?R7 宸叉爣娉紝Phase 1 mini 楠岃瘉闇€閲嶇偣纭 `CommAnalytic` 鍦ㄦ鍦烘櫙涓嬬殑绮惧害

