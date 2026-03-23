# 瀹炴祴绠楀瓙鎬ц兘鏁版嵁搴?Phase 1 E2E 楠岃瘉鎶ュ憡

**鏃ユ湡**: 2026-03-15
**Profiling 鏁版嵁**: CANN 8.5, vLLM 0.15.0, torch 2.9.0 (2026-03-13 閲囬泦, eager prefill + cudagraph decode)

---

## 1. 鑳屾櫙涓庣洰鏍?

TensorCast 姝ｅ湪鏋勫缓鍩轰簬瀹炴祴 Profiling 鏁版嵁鐨勭畻瀛愭€ц兘浼扮畻绯荤粺锛坄EmpiricalPerformanceModel + ProfilingDataSource`锛夈€傝绯荤粺閫氳繃 `op_mapping.yaml` 灏?TC 绠楀瓙鏄犲皠鍒?NPU kernel锛屽啀浠?CSV 鏁版嵁涓煡璇㈠尮閰?shape 鐨勫疄娴嬪欢杩燂紝鏇夸唬 Roofline 瑙ｆ瀽妯″瀷銆?

**Phase 1 鐩爣**: 鏌ヨ鍩虹璁炬柦鎵撻€?+ 鍏ㄩ噺 blocker 鏆撮湶 + 鎸囨爣浣撶郴寤虹珛銆備笉杩芥眰 <15% E2E 璇樊锛圵ork Plan Phase 3 鐩爣锛夈€?

**楠岃瘉鑼冨洿**: 4 涓?E2E 鍦烘櫙 脳 2 妯″瀷 (Qwen3-32B BF16, DeepSeek-V3 W8A8) 脳 2 闃舵 (Prefill, Decode)銆?

---

## 2. 缁撹

### 2.1 GO/NO-GO: GO

Phase 1 鐩爣杈炬垚:
1. 鏌ヨ鍩虹璁炬柦 4 鍦烘櫙鍏ㄩ儴鍙繍琛?
2. 鎵€鏈?MISS 鍘熷洜宸插垎绫伙紝瑙ｅ喅鏂规璺緞娓呮櫚锛屾瘡椤规湁 owner
3. 鎸囨爣浣撶郴寤虹珛锛圡1-M3 + 鎮茶瑙勫垯 + 铻嶅悎鍒嗙粍锛?

### 2.2 缁撴灉鎬昏〃

| 鍦烘櫙 | TC 鍙傛暟 | M1: Op-Count HR | M2: Fused Op HR (GO/NO-GO) | M3: Fused (涓嶅惈 zc) |
|------|---------|----------------|-----------------|---------------------|
| Qwen3 Prefill | nq=10, ql=4104, tp=16, BF16 | 78.6% (44/56) | **63.3%** (19/30) | 31.2% (5/16) |
| Qwen3 Decode | nq=16, ql=1, cl=4096, tp=16, BF16 | 81.8% (45/55) | **70.0%** (21/30) | 43.8% (7/16) |
| DSv3 Prefill | nq=1, ql=256, tp=8, dp=2, ep=16, W8A8 | 60.2% (62/103) | **38.6%** (17/44) | 12.9% (4/31) |
| DSv3 Decode | nq=16, ql=1, cl=4096, tp=8, dp=2, ep=16, W8A8 | 59.6% (62/104) | **40.9%** (18/44) | 16.1% (5/31) |

### 2.3 鍏抽敭鍙戠幇

1. **DSv3 鏈€澶?blocker: DispatchFFNCombine (DFC) 铻嶅悎 gap**銆侼PU 灏?MoE 璺緞鐨?`all_to_all脳2 + GroupedMatmul脳2 + SwiGlu + routing` 铻嶅悎涓哄崟涓€ DFC kernel锛屽崰 DSv3 寤惰繜 ~40%銆俆C 鏃犲搴?fusion pass锛岄渶 Phase 2 瀹炵幇銆?

2. **Qwen3 鏈€澶?blocker: FIA CSV 鏍煎紡**銆侳usedInferAttentionScore CSV 鏉ヨ嚜 profiling trace锛岄潪 microbench 鏍煎紡锛屾棤娉曟寜 `(batch, seq, heads, head_dim)` 缁撴瀯鍖栨煡璇€傚崰 Qwen3 寤惰繜 ~9%銆?

3. **Shape 鍖归厤瑙勫垯宸插缓绔?7 鏉?*锛堥檮褰?A锛夛紝瑕嗙洊 batch dim strip銆丗RACTAL_NZ 鎭㈠銆佹潈閲嶈浆缃€乥lock-padding銆丼wiGlu 鍚堝苟銆丷oPE layout銆乫latten batch銆備絾 RoPE 瀛樺湪 dtype gap锛圢PU 鍐呴儴 FP32 vs TC BF16锛夛紝Qwen3 Prefill 鐨?RoPE shape 鍖归厤姝ｇ‘浣?dtype 闃绘 HIT銆?

4. **TC 涓诲垎鏀渶淇 2 涓缓妯￠棶棰?*: (a) add_rms_norm2 SP 鍦烘櫙鏈 seq dim 闄や互 TP锛?b) W8A8 MLA output quantize shape 涓?NPU 涓嶄竴鑷达紙3D per-head vs 2D hidden锛屽悗鑰呭湪 CSV 涓凡瀛樺湪锛夈€?

### 2.4 鏀剁泭绮椾及

| 瀹屾垚椤?| Qwen3 M3 | DSv3 M3 |
|--------|----------|---------|
| 褰撳墠 | 31-44% | 13-16% |
| + P0 (DFC + FIA + MLA) | ~50-60% | ~40-50% |
| + P1 (shape data + dtype fix + tc_input_count) | ~60-70% | ~55-65% |
| + InterpolatingDataSource | ~75-85% | ~65-75% |

---

## 3. 鎸囨爣浣撶郴

| 鎸囨爣 | 瀹氫箟 | 鐢ㄩ€?|
|------|------|------|
| **M1: Raw Op-Count HR** | `HIT_invocations / total_invocations` | Debug 鐢紝鍚戝悗鍏煎 |
| **M2: Fused Op HR** | 鎸夎瀺鍚?op 璁℃暟锛堝惈 zero_cost锛夛紝鎮茶瑙勫垯 | **GO/NO-GO 鍒ゅ畾** |
| **M3: Fused Op HR (涓嶅惈 zc)** | 鍚?M2锛屾帓闄?zero_cost | 鐪熷疄璁＄畻瑕嗙洊 |

**鎮茶瑙勫垯**: 鍚屼竴 op 鑻ユ湁浠讳竴 shape MISS锛屾暣涓?op 璁′负 MISS銆?

**铻嶅悎鍒嗙粍**: DFC (6-8 涓?TC ops = 1 铻嶅悎 op)銆丮LAPO銆丮LA銆丮C2 鍚岀悊銆傚叏閮ㄦ垚鍛?HIT 鎵嶇畻 HIT銆?

**Phase 2 寰呭疄鐜?*: M4 (Per-Shape Match HR)銆丮5 (Latency-Weighted HR)銆?

---

## 4. MISS 鏍瑰洜鍒嗘瀽

### 4.1 鏍瑰洜鍒嗙被姹囨€?

| 鏍瑰洜鍒嗙被 | 瀹氫箟 | Qwen3 褰卞搷 | DSv3 褰卞搷 | 瑙ｅ喅鏂规 |
|---------|------|-----------|----------|---------|
| **fused_kernel_gap** | N:1 铻嶅悎 (DFC) 鏃?TC 瀹炵幇 | 鏃?| ~40% 寤惰繜 | TC fusion pass |
| **data_format_gap** | CSV 鏍煎紡/dtype 涓嶅尮閰?| ~9.5% (FIA + RoPE dtype) | ~1.5% (MLA FIA) | FIA microbench 鏍煎紡; RoPE dtype 瀹芥澗鍖归厤 |
| **shape_coverage_gap** | CSV 缂哄皯鍖归厤 M脳D 缁勫悎 | ~2% | ~12% (鍚?quantize ~10%) | Microbenchmark 琛ュ厖 |
| **tc_decomposition_mismatch** | TC 涓棿 shape 涓?NPU 涓嶅悓 | ~0.5% (KV cache) | ~2% (MLA quantize, KV cache) | TC 寤烘ā灞備慨澶?|
| **input_count_mismatch** | TC vs CSV input 鏁伴噺宸紓 | ~1% | ~3% | tc_input_count 鎵╁睍 |
| **sp_modeling_gap** | SP 鍦烘櫙 TC 鏈 seq dim 闄や互 TP | ~1% | ~0.1% | TC 涓诲垎鏀?Issue |
| **structural_miss** | Embedding TP 杈呭姪 ops 绛?| <0.01% | <0.01% | zero_cost 鏍囪 |

### 4.2 DSv3 quantize 脳3 MISS 娣卞叆鍒嗘瀽

DSv3 鐨?3 涓?quantize MISS 缁?debug trace 閫?shape 楠岃瘉锛屾牴鍥犲悇涓嶇浉鍚?

| TC Shape | 鏉ユ簮 | 鏍瑰洜 | 瑙ｅ喅鏂规 |
|----------|------|------|---------|
| `(8,16,128)` 3D | MLA attention output (tokens=8, heads=16, head_dim=128) | `tc_decomposition_mismatch`: TC quantize 淇濈暀 3D per-head 鏍煎紡锛孨PU 鍦?reshape 鍒?`(8,2048)` 鍚庡仛 quantize锛宍(8,2048)` 鍦?CSV 涓?*宸插瓨鍦?* | TC 涓诲垎鏀帓鏌?MLA output quantize shape 鏉ユ簮骞朵慨澶?|
| `(1,8,2304)` 鈫?strip 鈫?`(8,2304)` | 鍏变韩涓撳 dense FFN (D=18432/TP=2304) | `shape_coverage_gap`: CSV 鏃?D=2304 | Microbenchmark 琛ュ厖 |
| `(256,2048)` 2D | MoE routed expert (M=256, D=2048) | `shape_coverage_gap`: CSV 鏈?`(8,2048)` 浣嗘棤 `(256,2048)` | Microbenchmark 琛ュ厖 |

娉? `_FLATTEN_BATCH_KERNELS` 鐨?`(8*16,128)=(128,128)` 瀵圭涓€涓?case 璇箟閿欒锛堝簲鍚堝苟鍚庝袱缁?`(8,16*128)=(8,2048)`锛夛紝浣嗕笉浜х敓 false positive銆?

---

## 5. Phase 2 TODO

### 5.1 浼樺厛绾ф帓搴?

| # | 宸ヤ綔椤?| Owner | 浼樺厛绾?| 棰勪及宸ヤ綔閲?| 棰勬湡 M3 鎻愬崌 |
|---|--------|-------|--------|-----------|-------------|
| **P0-1** | **DFC TC fusion pass** (N:1) | LJW | P0 | 1 鍛?| DSv3: +15-30pp |
| **P0-2** | **FIA microbench CSV 鏍煎紡** | ZZY | P0 | 杩涜涓?| Qwen3: +5-10pp |
| **P0-3** | **MLA/MLAPO composite shape 淇** | ZH | P0 | 1-2 澶?| DSv3: +3-5pp |
| P1-1 | quantize/norm Microbenchmark shape 缃戞牸琛ュ厖 | TCX | P1 | 鏁版嵁閲囬泦 | DSv3: +3-5pp |
| P1-2 | RoPE dtype 瀹芥澗鍖归厤 (_triton_rope FLOAT vs BF16) | 寰呭畾 | P1 | ~10 琛?| Qwen3 PF: +1-2pp |
| P1-3 | tc_input_count 鎵╁睍 (add, mul, index) | 寰呭畾 | P1 | Config | 鍏ㄥ満鏅? +1-2pp |
| P1-4 | SP 寤烘ā淇 (add_rms_norm2 M/TP) | TC 涓诲垎鏀?| P1 | 闇€璇勪及 | Qwen3 PF: +1-2pp |
| P1-5 | MoE routing 杈呭姪 ops CSV 琛ュ厖 | TCX | P1 | 鏁版嵁閲囬泦 | DSv3: +1-2pp |
| P2-1 | Per-Shape Match HR (M4) 瀹炵幇 | ZH | P2 | ~60 琛?| 璇婃柇鏀瑰杽 |
| P2-2 | Latency-Weighted HR (M5) 瀹炵幇 | HXW | P2 | 澶栭儴鑴氭湰 | 鎬ц兘璇勪及 |
| P2-3 | Static cost 鎻愬彇涓哄叕鍏辩粍浠?| TC 涓诲簱 Issue | P2 | ~30 琛?| E2E +1-5% |

### 5.2 TC 涓诲垎鏀?Issue

| Issue | 鎻忚堪 | 棰勬湡鏀剁泭 |
|-------|------|---------|
| add_rms_norm2 SP 缁村害 | SP 鍦烘櫙涓?TC 鏈 seq dim 闄や互 TP锛屽鑷?M=41040 vs NPU M=2565 | 淇鍚?Qwen3 PF norm 浠?MISS鈫扝IT |
| MLA output quantize shape | W8A8 MLA 璺緞 TC quantize shape `(8,16,128)` (3D) vs NPU `(8,2048)` (2D)锛屽悗鑰呭湪 CSV 涓凡瀛樺湪 | 淇鍚?DSv3 姣忓眰 quantize 浠?MISS鈫扝IT (~1-2%) |

---

## 6. 浠ｇ爜鍙樻洿鎬荤粨

### 6.1 璁″垝鍐呭彉鏇?(C1-C6)

| 鍙樻洿 | 璇存槑 |
|------|------|
| MLA/MLAPO 瑙ｉ櫎纭紪鐮佹嫆缁?| composite lookup 鎭㈠ |
| moe_gating_topk op | 鍖归厤 NPU MoeGatingTopK kernel |
| tc_input_count 閰嶇疆 | 7 ops (quantize, embedding 绛? |
| MISS reason 淇 | tc_input_count 鍙屼晶鎴柇 |
| TC_ENABLE_INTERPOLATION 寮€鍏?| 榛樿 OFF |
| Fused Op HR 鎸囨爣 | 鍚偛瑙傝鍒欎慨姝?|

### 6.2 E2E 楠岃瘉杩囩▼涓彂鐜扮殑鏀硅繘

| 鍙樻洿 | 鍙戠幇杩囩▼ |
|------|---------|
| 閫氫俊 alpha-beta 妯″瀷鎻掑€?| E2E 鍙戠幇閫氫俊鍏?MISS锛宮essage_bytes 绮剧‘鍖归厤涓嶅悎鐞?|
| Embedding TP row 杈呭姪 ops (zero_cost) | `--word-embedding-tp row` 娣诲姞鏈槧灏?ops |
| FRACTAL_NZ 鏉冮噸杞疆淇 | ND 杞疆妫€鏌ヤ粎瀵?fmt=="ND" 鐢熸晥锛孎RACTAL_NZ 鎭㈠鍚庢湭灏濊瘯杞疆 |
| _MATMUL_KERNELS 琛ュ叏 | QuantBatchMatmulV3 绛変笉鍦ㄩ泦鍚堜腑 |
| EP=16 閰嶇疆淇 | TC `--ep-size` = vLLM EP = TP脳DP |
| HCCL num_devices=8 鏁版嵁鍚堝叆 | 淇 DSv3 TP=8 閫氫俊 MISS |
| 鎮茶瑙勫垯淇 | quantize/mm/add/swiglu 鍦?HIT 鍜?MISS 涓弻閲嶈鏁?|
| `_FLATTEN_BATCH_KERNELS` 3D鈫?D 鍖归厤 | quantize/norm 鐨?`(B,M,D)鈫?B*M,D)` flatten |
| `_ROPE_KERNELS` 鎵╁睍 + normalize 淇 | 娣诲姞 `_triton_rope`/`split_qkv_rmsnorm_rope_kernel`锛屾敮鎸?tc_input_count=2 |

### 6.3 宸蹭慨澶嶇殑 Bug

| Bug | 褰卞搷 | 淇 |
|-----|------|------|
| MLA 纭紪鐮佹嫆缁?| MLA/MLAPO 鏃犳硶 composite 鏌ヨ | 鍒犻櫎鏃╂湡杩斿洖 |
| MISS reason 鏈埅鏂?CSV 渚?| tc_input_count 鍚?reason 浠嶄负 input_count_mismatch | 鍙屼晶鎴柇 |
| FRACTAL_NZ 杞疆浠呭 ND 鐢熸晥 | QuantBatchMatmulV3 鏉冮噸杞疆涓嶅尮閰?| 绉婚櫎 fmt=="ND" 闄愬埗 |
| _MATMUL_KERNELS 涓嶅畬鏁?| 杞疆妫€鏌ヤ笉瀵?QuantBatchMatmulV3 绛夌敓鏁?| 琛ュ叏闆嗗悎 |
| Fused Op HR 鍙岄噸璁℃暟 | 鍚屽悕 op 閮ㄥ垎 HIT/MISS 鏃?inflate 鍒嗗瓙鍒嗘瘝 | 鎮茶瑙勫垯 |

### 6.4 娴嬭瘯

137 passed, 9 skipped, 0 failures銆?

---

## 7. 宸ヤ綔璁″垝鍜岃璁℃枃妗ｆ洿鏂板缓璁?

### 7.1 Work Plan 鏇存柊

1. **Phase 1 鐩爣** (宸茶揪鎴?: "鏌ヨ鍩虹璁炬柦鎵撻€?+ blocker 鍏ㄩ噺鏆撮湶 + 鎸囨爣浣撶郴寤虹珛"
2. **Phase 2 涓绘寚鏍?*: M3 (Fused Op HR 涓嶅惈 zc) > 50%
3. **Phase 3 鐩爣**: M5 (Latency-Weighted HR) > 80%
4. **搴熷純 Op-Count HR**: M1 浠呯敤浜?debug锛屼笉浣滀负璇勪及鎸囨爣

### 7.2 璁捐鏂囨。鏇存柊

1. **搂4.2 鏌ヨ鍒嗘淳**: 澧炲姞閫氫俊 alpha-beta 妯″瀷鎻掑€兼弿杩?
2. **搂4.5 op_mapping.yaml**: 澧炲姞 kernel dispatch 鏉′欢璁板綍绾﹀畾
3. **搂4.9 FRACTAL_NZ**: 琛ュ厖 "FRACTAL_NZ 鎭㈠鍚庝粛闇€灏濊瘯鏉冮噸杞疆" 瑙勫垯
4. **搂4.10 Flatten Batch 瑙勫垯**: `_FLATTEN_BATCH_KERNELS` 鐨?`(B,M,D)鈫?B*M,D)` 鍖归厤
5. **搂7 璇勪及鎸囨爣**: 鏇挎崲涓?M1-M5 浜斿眰鎸囨爣浣撶郴锛屽惈鎮茶瑙勫垯璇存槑
6. **鏂板**: EP 閰嶇疆璇存槑 (TC `--ep-size` = vLLM EP = TP 脳 DP)
7. **鏂板**: `_MATMUL_KERNELS` 搴斿寘鍚墍鏈?matmul 鍙樹綋
8. **宸插畬鎴?*: `_ROPE_KERNELS` 宸叉墿灞曪紝normalize 鏀寔 tc_input_count=2锛沝type gap 寰?Phase 2 澶勭悊

---

## 闄勫綍 A: Shape 鍖归厤瑙勫垯鎬昏

| # | 瑙勫垯 | Kernel 鑼冨洿 | 鍙樻崲 | 鐘舵€?|
|---|------|-----------|------|------|
| 1 | Batch dim=1 strip | 鎵€鏈?| `(1,M,D)鈫?M,D)` | 姝ｅ父 |
| 2 | FRACTAL_NZ 鎭㈠ | 鎵€鏈?| `[H,W,bh,bw]鈫?H*bw,W*bh)` | 姝ｅ父 |
| 3 | ND 鏉冮噸杞疆 | `_MATMUL_KERNELS` | `(K,N)鈫?N,K)` | 姝ｅ父 |
| 4 | Block-padding 瀹瑰繊 | 鎵€鏈?| `ceil(M/bs)*bs, bs鈭坽16,32,64}` | 姝ｅ父 |
| 5 | SwiGlu 杈撳叆鍚堝苟 | `_SWIGLU_KERNELS` | `2脳(M,D/2)鈫?M,D)` | 姝ｅ父 |
| 6 | RoPE layout 褰掍竴鍖?| `_ROPE_KERNELS` | `(B,H,S,D)鈫?B,S,H,D)` + Q鈫擪 閲嶆帓 | Shape 姝ｇ‘锛宒type gap 闃绘 HIT |
| 7 | Flatten batch | `_FLATTEN_BATCH_KERNELS` | `(B,M,D)鈫?B*M,D)` | 姝ｅ父锛屽緟鏁版嵁琛ュ厖 |

## 闄勫綍 B: TC 鍛戒护

```bash
DATA_DIR="$(pwd)/tensor_cast/performance_model/perf_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5"

# Qwen3 Prefill
python3.10 -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 10 --query-length 4104 --word-embedding-tp row \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --quantize-linear-action DISABLED \
  --performance-model profiling --compile --perf-database "$DATA_DIR"

# Qwen3 Decode
python3.10 -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 16 --query-length 1 --context-length 4096 --word-embedding-tp row \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --quantize-linear-action DISABLED \
  --performance-model profiling --compile --perf-database "$DATA_DIR"

# DSv3 Prefill
python3.10 -m tensor_cast.scripts.text_generate deepseek-ai/DeepSeek-V3 \
  --num-queries 1 --query-length 256 --word-embedding-tp row \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 8 --dp-size 2 --ep-size 16 \
  --quantize-linear-action W8A8_STATIC \
  --performance-model profiling --compile --perf-database "$DATA_DIR"

# DSv3 Decode
python3.10 -m tensor_cast.scripts.text_generate deepseek-ai/DeepSeek-V3 \
  --num-queries 16 --query-length 1 --context-length 4096 --word-embedding-tp row \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 8 --dp-size 2 --ep-size 16 \
  --quantize-linear-action W8A8_STATIC \
  --performance-model profiling --compile --perf-database "$DATA_DIR"
```

TC 鍙傛暟浠?CSV shapes 鍙嶆帹锛堥潪鐩存帴浣跨敤 vLLM 閰嶇疆锛夈€侲P 閰嶇疆: vLLM `--enable-expert-parallel` with TP=8, DP=2 鈫?TC `--ep-size 16`銆侲mbedding TP: `--word-embedding-tp row`锛坧rofiling 鏄剧ず vocab/TP 鍒嗙墖锛夈€傚姞 `--log-level debug` 鍙煡鐪嬫瘡涓?MISS 鐨?TC shape 鍜?CSV shape 瀵规瘮銆?

## 闄勫綍 C: 鍏ㄥ満鏅€?Shape MISS 鍒嗘瀽

鏈檮褰曞 4 涓?E2E 鍦烘櫙涓瘡涓€鏉?MISS 璁板綍杩涜閫?shape 鏍瑰洜鍒嗘瀽銆?

**鏍瑰洜鍒嗙被**:

| 浠ｅ彿 | 鍚箟 |
|------|------|
| `data_format_gap` | CSV 鏍煎紡/dtype 涓嶅尮閰?|
| `shape_coverage_gap` | CSV 鏈夋纭?kernel 浣嗙己灏?M/D 缁勫悎 |
| `tc_decomposition_mismatch` | TC 鍒嗚В鏂瑰紡涓?NPU 涓嶅悓 |
| `input_count_mismatch` | TC 涓?CSV 杈撳叆鏁伴噺涓嶄竴鑷?|
| `fused_kernel_gap` | NPU N:1 铻嶅悎锛孴C 鏃犲搴?pass |
| `sp_modeling_gap` | SP 鍦烘櫙 TC 鏈 seq dim 闄や互 TP |
| `structural_miss` | 鏍规湰璇箟涓嶅悓 |

---

### C.1 Qwen3 Decode (M1=81.8%, M2=70.0%, M3=43.8%)

| # | TC Op | NPU Kernel | TC Shape | CSV 鏈€杩戜技 | 鏍瑰洜 | 鍒嗘瀽 |
|---|-------|-----------|----------|-----------|------|------|
| 1 | `where.self` | SelectV2 | `(1,16)脳3` | 鏃?CSV | `structural_miss` | Embedding TP row 杈呭姪 op锛屽凡鏍囪 zero_cost |
| 2 | `embedding` | GatherV2 | `(9496,5120),(1,16)` | `(9496,5120;336;1)` | `shape_coverage_gap` | tc_input_count=2 宸查厤缃紝indices M=16 vs CSV M=336 |
| 3 | `mul.Tensor` | Mul | `(1,16,5120),(1,16,1)` | `"336;"` | `input_count_mismatch` | TC 2 inputs vs CSV 1 input |
| 4 | `index.Tensor` 脳2 | Index | `(40960,256),(16,)` | `(336,5120;1;2;128)` | `input_count_mismatch` | TC 2 inputs vs CSV 4 inputs |
| 5 | `apply_rope` | _triton_rope | `(1,1,16,128),(1,4,16,128)` | `(336,4,128;336,1,128;...)` | `data_format_gap` | shape normalize 姝ｇ‘浣?CSV dtype 涓嶅尮閰?(FLOAT vs BF16) + decode M=16 鏃?CSV |
| 6 | `reshape_and_cache` | ReshapeAndCacheNdKernel | `(16,128),(16,128),...` | `(333,1,128;...)` | `tc_decomposition_mismatch` | TC KV 2D 缂?head_dim锛孋SV 3D 鍚?head_dim=1 |
| 7 | `attention` | FusedInferAttentionScore | (鐗规畩鏌ヨ) | CSV 缂?microbench 鍒?| `data_format_gap` | FIA CSV 闈?microbench 鏍煎紡 |
| 8 | `add.Tensor` | Add | `(1,16,5120),(16,5120)` | 鏃犲尮閰?| `input_count_mismatch` | TC 2 inputs锛宻hape 璇箟涓嶅尮閰?|
| 9 | `index.Tensor` | Index | `(1,16,5120),(16,)` | `(16,5120;1;2;16)` | `input_count_mismatch` | TC 2 inputs vs CSV 4 inputs |
| 10 | `copy_` | TensorMove | `(2,513,128,1,128)` 5D | `(336,5120)` 2D | `structural_miss` | KV cache copy vs activation copy |

---

### C.2 Qwen3 Prefill (M1=78.6%, M2=63.3%, M3=31.2%)

涓?Decode 鍏辨湁鐨?MISS 涓嶉噸澶嶏紝浠呭垪 Prefill 鐗规湁椤广€?

| # | TC Op | NPU Kernel | TC Shape | CSV 鏈€杩戜技 | 鏍瑰洜 | 鍒嗘瀽 |
|---|-------|-----------|----------|-----------|------|------|
| 1 | `rms_norm` | RmsNorm | `(1,41040,5120),(5120,)` | `(2565,5120;5120)` | `sp_modeling_gap` | M=41040 vs CSV M=2565=41040/16(TP)锛孴C 鏈櫎浠?TP |
| 2 | `apply_rope` | _triton_rope | `(1,1,41040,128),(1,4,41040,128)` | `(41040,4,128;41040,1,128;...)` | `data_format_gap` | shape 鍖归厤姝ｇ‘锛屼絾 CSV 绗簩 input dtype 涓?FLOAT (NPU FP32)锛孴C 涓?BF16 |
| 3 | `add_rms_norm2` | AddRmsNormBias | `(1,41040,5120),(41040,5120),(5120,)` | `(2565,5120;...)` | `sp_modeling_gap` | 鍚?rms_norm锛孧=41040 vs M/TP=2565 |

---

### C.3 DSv3 Decode (M1=59.6%, M2=40.9%, M3=16.1%)

DSv3 MISS 鏁伴噺鏄捐憲楂樹簬 Qwen3锛屼富瑕佸洜 MoE routing 杈呭姪 ops 鍜?DFC 铻嶅悎 gap銆?

#### C.3.1 Embedding & Shared Attention

| # | TC Op | NPU Kernel | TC Shape | CSV 鏈€杩戜技 | 鏍瑰洜 | 鍒嗘瀽 |
|---|-------|-----------|----------|-----------|------|------|
| 1 | `where.self` | SelectV2 | `(1,8)脳3` | 鏃?CSV | `structural_miss` | Embedding TP row 杈呭姪 op |
| 2 | `mul.Tensor` | Mul | `(1,8,7168),(1,8,1)` | `"8;"` | `input_count_mismatch` | TC 2 inputs vs CSV 1 input |
| 3 | `index.Tensor` | Index | `(163840,128),(8,)` | `(163840,64;1;2;4)` | `input_count_mismatch` | TC 2 vs CSV 4 inputs |
| 4 | `rms_norm` | RmsNorm | `(1,8,7168),(7168,)` | `(256,7168;7168)` | `shape_coverage_gap` | M=8 鏃?CSV锛屾渶杩?M=256 |
| 5 | `reshape_and_cache` | ReshapeAndCacheNdKernel | `(8,512),(8,64)` | `(333,1,128;...)` | `tc_decomposition_mismatch` | MLA KV 2D (kv_lora_rank=512) vs 鏍囧噯 KV cache 3D |

#### C.3.2 MLA/MLAPO Composite & Quantize

| # | TC Op | NPU Kernel | TC Shape | CSV 鏈€杩戜技 | 鏍瑰洜 | 鍒嗘瀽 |
|---|-------|-----------|----------|-----------|------|------|
| 6 | `quantize` (MLA) | AscendQuantV2 | `(8,16,128)` 3D | `(8,2048)` **TC 淇鍚庡彲鍖归厤** | `tc_decomposition_mismatch` | TC quantize 淇濈暀 3D per-head锛孨PU reshape 鍒?2D 鍚?quantize |
| 7 | `quantize` (shared FFN) | AscendQuantV2 | `(1,8,2304)` | 鏃?D=2304 | `shape_coverage_gap` | D=2304=18432/8(TP)锛屽叡浜笓瀹?FFN |
| 8 | `quantize` (MoE) | AscendQuantV2 | `(256,2048)` | `(8,2048)` | `shape_coverage_gap` | M=256 (鍏?expert)锛孋SV 浠?M=8 |

#### C.3.3 Shared Expert Path

| # | TC Op | NPU Kernel | TC Shape | CSV 鏈€杩戜技 | 鏍瑰洜 | 鍒嗘瀽 |
|---|-------|-----------|----------|-----------|------|------|
| 9 | `add_rms_norm2` | AddRmsNormQuant | `(1,8,7168),(8,7168),(7168,)` | 鏃?CSV | `shape_coverage_gap` | W8A8 浜х敓 AddRmsNormQuant锛岃 kernel 鏃?CSV |
| 10 | `add_rms_norm2` | AddRmsNormBias | `(1,8,7168),(8,7168),(7168,)` | `(1,7168;...)` | `shape_coverage_gap` | M=8 鏃?CSV |
| 11 | `copy_` | TensorMove | `(8,7168)` | `(256,7168)` | `shape_coverage_gap` | M=8 鏃?CSV |

#### C.3.4 MoE Routing 杈呭姪 Ops (12 涓?MISS)

`sigmoid`, `topk`脳3, `sum`脳3, `scatter`, `bitwise_not`, `where`, `gather`, `div` 鈥?鍧囦负 MoE expert routing 閫昏緫浜х敓鐨勮緟鍔?op銆傚ぇ閮ㄥ垎鏃?CSV锛坄shape_coverage_gap`锛夛紝灏戞暟涓?`input_count_mismatch`锛坱c_input_count 鏈厤缃級銆俿hape 鍧囧惈 `n_experts=256` 缁村害銆?

#### C.3.5 DFC 铻嶅悎缁?(5 涓?MISS)

`permute_tokens`, `grouped_matmul`, `cat`(256脳expert), `unpermute_tokens` 鈥?鍧囦负 `fused_kernel_gap`銆侼PU 灏嗘暣涓?MoE dispatch+compute+combine 铻嶅悎涓?DispatchFFNCombine kernel锛孴C 閫?op 妯℃嫙銆?

#### C.3.6 Post-MoE & Embedding

`add.Tensor`脳3, `index.Tensor`, `mm`(lm_head M=8), `copy_`(MLA KV 3D) 鈥?涓昏涓?`input_count_mismatch` 鍜?`shape_coverage_gap`锛坉ecode M=8 鏃?CSV锛夈€?

---

### C.4 DSv3 Prefill (M1=60.2%, M2=38.6%, M3=12.9%)

涓?Decode 澶ч儴鍒嗙浉鍚岋紙MoE routing銆丏FC 瀹屽叏涓€鑷达級銆侾refill 鐗规湁宸紓:

- `embedding` GatherV2: indices M=256 鏃?CSV锛圖ecode M=8 涔熸棤锛夆啋 `shape_coverage_gap`
- `rms_norm`: M=256 鍦?CSV 涓瓨鍦?鈫?**HIT**锛圖ecode M=8 MISS锛?
- `quantize` (MLA): shape `(256,16,128)` 鈫?鍚?Decode 鏍瑰洜
- `static_quant_linear` (shared expert): N=4608 vs CSV N=4096 鈫?`shape_coverage_gap`
- `swiglu` (shared expert): concat鈫抈(256,4608)` vs CSV `(256,4096)` 鈫?`shape_coverage_gap`
- `add_rms_norm2` (FFN 灞?: M=256 鍖归厤浣?input 鏁颁负 4 vs TC 3 鈫?`input_count_mismatch`
- `mm` (lm_head): M=1 鈫?**HIT**锛圖ecode M=8 MISS锛?

### C.5 璺ㄥ満鏅?MISS 鏍瑰洜姹囨€?

| 鏍瑰洜 | Qwen3 Decode | Qwen3 Prefill | DSv3 Decode | DSv3 Prefill |
|------|-------------|--------------|------------|-------------|
| `data_format_gap` | 2 (FIA+RoPE) | 2 (FIA+RoPE) | 0 | 0 |
| `shape_coverage_gap` | 0 | 1 | 18 | 17 |
| `tc_decomposition_mismatch` | 1 | 1 | 2 | 3 |
| `input_count_mismatch` | 4 | 4 | 8 | 8 |
| `fused_kernel_gap` | 0 | 0 | 5 | 5 |
| `sp_modeling_gap` | 0 | 2 | 0 | 0 |
| `structural_miss` | 2 | 2 | 2 | 2 |

### C.6 瑙ｅ喅浼樺厛绾х煩闃?

| 浼樺厛绾?| 鏍瑰洜 | 褰卞搷鍦烘櫙 | 棰勬湡 M3 鎻愬崌 | 宸ヤ綔閲?|
|--------|------|---------|-------------|--------|
| **P0** | `fused_kernel_gap` (DFC) | DSv3 鍏ㄥ満鏅?| +15-30pp | TC fusion pass (1-2 鍛? |
| **P0** | `data_format_gap` (FIA) | Qwen3 鍏ㄥ満鏅?| +5-10pp | FIA microbench CSV |
| **P1** | `shape_coverage_gap` (routing ops) | DSv3 鍏ㄥ満鏅?| +3-5pp | Microbench 閲囬泦 |
| **P1** | `shape_coverage_gap` (quantize/norm M脳D) | DSv3 鍏ㄥ満鏅?| +3-5pp | Microbench 缃戞牸鎵╁睍 |
| **P1** | `data_format_gap` (RoPE dtype) | Qwen3 Prefill | +1-2pp | dtype 瀹芥澗鍖归厤 |
| **P1** | `input_count_mismatch` | 鍏ㄥ満鏅?| +1-3pp | op_mapping 閰嶇疆 |
| **P1** | `sp_modeling_gap` | Qwen3 Prefill | +1-2pp | TC 涓诲垎鏀慨澶?|
| **P2** | `tc_decomposition_mismatch` (KV cache) | 鍏ㄥ満鏅?| +1pp | TC 寤烘ā灞?|
| **P3** | `structural_miss` | 鍏ㄥ満鏅?| <0.5pp | zero_cost 鏍囪 |

