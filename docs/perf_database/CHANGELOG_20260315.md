# CHANGELOG 2026-03-15

璁捐鏂囨。 v1.3.1 鈫?v1.4 + 宸ヤ綔璁″垝 v3.1 鈫?v3.2

鍩轰簬 Phase 1 E2E 闆嗘垚楠岃瘉缁撴灉锛? 鍦烘櫙锛孮wen3-32B + DeepSeek-V3锛夋洿鏂般€?

---

## 璁捐鏂囨。鍙樻洿 (v1.3.1 鈫?v1.4)

### 鏂板

- **搂4.2**: 閫氫俊 alpha-beta 妯″瀷鎻掑€硷紙least-squares fit锛岀簿纭尮閰嶄紭鍏?+ 鎻掑€?fallback锛岄粯璁ゅ紑鍚級
- **搂4.2**: 3D鈫?D Flatten Batch 鍖归厤瑙勫垯 鈥?quantize/norm 绫?kernel 鐨?`(B,M,D)鈫?B*M,D)` 灞曞钩鍖归厤锛屽尯鍒簬閫氱敤 `_strip_batch_dim`锛堜粎 B=1锛?
- **搂4.9**: FRACTAL_NZ 鎭㈠鍚庢潈閲嶈浆缃尮閰?鈥?瀵?`_MATMUL_KERNELS` 鍏ㄩ泦锛堝惈 `QuantBatchMatmulV3` 绛夐噺鍖栧彉浣擄級鐢熸晥锛屼笉闄愪簬 ND 鏍煎紡
- **搂7.5**: M1-M5 浜斿眰璇勪及鎸囨爣浣撶郴 + 鎮茶瑙勫垯 + 铻嶅悎鍒嗙粍锛圥hase 1 E2E 寤虹珛锛?

### 鍙樻洿

- **搂4.2**: 閫氫俊鏌ヨ绛栫暐浠?绮剧‘鍖归厤 `(num_devices, topology_tier)`"鏀逛负"绮剧‘鍖归厤浼樺厛 + alpha-beta 鎻掑€?fallback"

### 鏋舵瀯鍐崇瓥

1. **閫氫俊鎻掑€奸粯璁ゅ紑鍚?*: `message_bytes` 涓鸿繛缁€硷紝绮剧‘鍖归厤鍛戒腑鐜囨瀬浣庯紝alpha-beta 妯″瀷锛坄latency = 伪 + 尾 脳 message_bytes`锛夊绾挎€?bandwidth-dominated 閫氫俊鎷熷悎鍑嗙‘
2. **Flatten Batch 涓?kernel-specific 瑙勫垯**: 浠呭 quantize/norm 绫?kernel 鐢熸晥锛屼笉瀵?matmul 绛夌畻瀛愮敓鏁堛€傛壒娆″睍骞?`(B,M,D)鈫?B*M,D)` 浠呭湪 element-wise/row-wise 绠楀瓙涓婅涔夋纭?
3. **璇勪及鎸囨爣閲囩敤鎮茶瑙勫垯**: 鍚屼竴 op 浠讳竴 shape MISS 鈫?鏁翠釜 op 璁?MISS銆傝瀺鍚堝垎缁勶紙DFC銆丮LAPO銆丮LA銆丮C2锛夊弽鏄?NPU 绔疄闄?kernel 绮掑害

---

## 宸ヤ綔璁″垝鍙樻洿 (v3.1 鈫?v3.2)

### Phase 1 瀹屾垚鐘舵€?

Phase 1 E2E 闆嗘垚楠岃瘉浜?3.15 瀹屾垚锛孏O/NO-GO: **GO**銆?

| 鐘舵€?| 浠诲姟 |
|------|------|
| 鉁?鏂板畬鎴?| B2 (composite lookup), C4 (Decode E2E 楠岃瘉), D2 (attention lookup), D3 (InterpolatingDataSource), D4 (discover_operators) |
| 鉁?宸插畬鎴?| A1-A3, B1, C1-C3, C6-C10, D1 |
| 鈿?閮ㄥ垎瀹屾垚 | C5 (discover tool 瀛樺湪锛岃嚜鍔ㄧ敓鎴愭湭瀹炵幇), C11-1 (DFC 閰嶇疆 composite 鍒嗚В鏇夸唬鐙珛娓呭崟) |

### Phase 2 鎻愬墠瀹屾垚鐨勪换鍔?

浠ヤ笅 Phase 2 浠诲姟鍦?Phase 1 涓凡鎻愬墠瀹屾垚:

| 浠诲姟 | 璇存槑 |
|------|------|
| E1 generate_shape_grid.py | GEMM + attention shape 缃戞牸鐢熸垚 |
| E2 microbenchmark generation tool | torch.mm + ATB kernel 鑴氭湰鐢熸垚 |
| E5 Attention sqrt 鎻掑€?| InterpolatingDataSource 鍚?sqrt 鍙樻崲 |
| G1 MoeGatingTopK | op + CSV + op_mapping 鍏ㄩ摼璺?|

### Phase 2 鏂板浠诲姟 (E2E 鍙戠幇)

| 浠诲姟 | 璐熻矗浜?| 璇存槑 |
|------|--------|------|
| P-E2E-1 RoPE dtype 瀹芥澗鍖归厤 | 寰呭畾 | _triton_rope CSV FLOAT vs TC BF16锛寏10 琛?|
| P-E2E-2 quantize/norm Microbenchmark 缃戞牸 | TCX | DSv3 M脳D 缂哄け缁勫悎琛ュ厖 |
| P-E2E-3 MoE routing 杈呭姪 ops CSV | TCX | TopKV2, ReduceSum, Sigmoid 绛?|

### Phase 2 浼樺厛绾ц皟鏁?

鍩轰簬 E2E 鍙戠幇閲嶆柊鎺掑簭:
1. **P0**: DFC TC fusion pass 鈥?DSv3 ~40% 寤惰繜 (LJW)
2. **P0**: FIA microbench CSV 鏍煎紡 鈥?Qwen3 ~9% 寤惰繜 (ZZY)
3. **P0**: MLA/MLAPO composite shape 淇 鈥?DSv3 ~3-5% (ZH)

### TC 涓诲垎鏀緷璧?(鏂板)

| Issue | 鎻忚堪 | 褰卞搷 |
|-------|------|------|
| add_rms_norm2 SP 缁村害 | TC 鏈 seq dim 闄や互 TP | Qwen3 Prefill norm MISS |
| MLA output quantize shape | TC 3D per-head vs NPU 2D hidden | DSv3 W8A8 quantize MISS |

### 鏂板椋庨櫓

| # | 椋庨櫓 | 姒傜巼 | 缂撹В |
|---|------|------|------|
| R13 | _triton_rope CSV dtype gap | 浣?| Phase 2 dtype 瀹芥澗鍖归厤 |
| R14 | TC 涓诲垎鏀?SP/MLA 淇渚濊禆 | 涓?| 宸叉彁 Issue |

### E2E 鍏抽敭鍙戠幇

- `_FLATTEN_BATCH_KERNELS` 瑙勫垯宸插疄鐜颁絾褰撳墠涓嶆敼鍙樻寚鏍囷紙闇€鏁版嵁琛ュ厖閰嶅悎锛?
- `_ROPE_KERNELS` 宸叉墿灞?+ normalize 鏀寔 tc_input_count=2锛屼絾 dtype gap 闃绘 HIT
- quantize `(8,16,128)` MISS 涓?TC MLA output view/quantize 椤哄簭闂锛岄潪 3D鈫?D 鍙樻崲
- Qwen3 Prefill rms_norm/add_rms_norm2 MISS 涓?SP 寤烘ā闂锛圡 vs M/TP锛?

