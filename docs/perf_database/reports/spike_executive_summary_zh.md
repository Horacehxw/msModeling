# 绠楀瓙鎬ц兘鏁版嵁搴撶┛鍒哄疄楠屾€荤粨

**鏃ユ湡**: 2026-03-05
**浣滆€?*: Claude (AI 杈呭姪寮€鍙?
**鍏宠仈璁捐鏂囨。**: `docs/perf_database/OPERATOR_PERF_DATABASE_DESIGN_zh_v1.2.md`

---

## Executive Summary

鏈┛鍒哄疄楠岄獙璇佷簡璁捐鏂囨。 v1.2 涓?`EmpiricalPerformanceModel + DataSource` 鏋舵瀯鐨勫彲琛屾€с€備互 Qwen3-32B BF16 Prefill锛圱P=16锛変负娴嬭瘯鍦烘櫙锛屽熀浜?Qwen3-30B 鐨勭湡瀹?Profiling 鏁版嵁锛?*缁忚繃 5 杞凯浠ｄ紭鍖栵紝绠楀瓙鍖归厤鐜囦粠 6.5% 鎻愬崌鑷?87.0%锛?0/46锛夛紝鍏朵腑鍏ㄩ儴 12 涓绠楃畻瀛愬疄鐜?100% 鍖归厤**銆?

鏍稿績鍙戠幇锛歍C dispatch trace 涓?NPU Profiling 涔嬮棿瀛樺湪 8 绫荤郴缁熸€х殑 shape 宸紓锛坆atch 缁村害銆佹潈閲嶆牸寮忋€佽瀺鍚堢畻瀛愭媶鍒嗐€丷oPE 甯冨眬绛夛級锛屽潎鍙€氳繃 `ProfilingDataSource` 涓殑閫氱敤瑙勫垯鍖栧鐞嗚В鍐筹紝鏃犻渶淇敼 TC 鏍稿績浠ｇ爜銆傚墿浣?6 涓湭鍖归厤绠楀瓙灞炰簬缁撴瀯鎬у樊寮傦紙attention 鐗规畩妯″紡銆侀€氫俊绠楀瓙銆並V Cache 鎺ュ彛宸紓锛夛紝闇€鍚庣画涓撻」寮€鍙戙€?

绌垮埡杩囩▼涓彂鐜?**17 澶勭畝鍖栧疄鐜?*锛堣瑙?搂5锛夛紝娑夊強 shape 鍖归厤銆乷p_mapping 鏄犲皠銆佹煡璇㈤€昏緫銆乨type銆佹暟鎹瓑浜斾釜灞傞潰銆傝繖浜涚畝鍖栧湪绌垮埡闃舵瓒冲楠岃瘉鍙鎬э紝浣嗕骇鍝佸寲鍓嶉渶閫愰」璇勪及鍜屽崌绾с€?

**鍏抽敭缁撹锛氬熀浜?Profiling 鏁版嵁鐨勬€ц兘浼扮畻璺嚎鍙锛屽€煎緱鎶曞叆浜у搧鍖栥€?*

---

## 1. 绌垮埡鑼冨洿涓庣洰鏍?

| 椤圭洰 | 鍐呭 |
|------|------|
| **楠岃瘉鐩爣** | 璁捐鏂囨。 搂4.1-4.3锛欴ataSource ABC 鈫?ProfilingDataSource 鈫?EmpiricalPerformanceModel |
| **娴嬭瘯妯″瀷** | Qwen/Qwen3-32B (BF16, Prefill, TP=16) |
| **娴嬭瘯纭欢** | ATLAS_800_A3_752T_128G_DIE |
| **Profiling 鏉ユ簮** | Qwen3-30B Prefill锛堝悓鏋舵瀯缁村害锛孴P=16锛宻eq=136锛?|
| **鏁版嵁瑙勬ā** | 45 涓?kernel CSV 鏂囦欢锛?0+ 鏉?op_mapping 鏄犲皠 |

---

## 2. 杩唬杩囩▼涓庣粨鏋?

### 2.1 鍖归厤鐜囨紨杩?

| 杞 | 鍖归厤鐜?| 鏂板鏂规 | 鏂板 HIT |
|------|--------|---------|---------|
| v1 (鍩虹嚎) | 3/46 (6.5%) | `--compile` 鍚敤铻嶅悎绠楀瓙 | 鈥?|
| v2 | 9/46 (19.6%) | batch 缁村害鍓ョ + SwiGlu 杈撳叆鍚堝苟 | +6 |
| v3 | 36/46 (78.3%) | zero-cost 绠楀瓙娉ㄥ唽 + 澶囬€?kernel_type | +27 |
| v4 | 38/46 (82.6%) | RoPE shape 褰掍竴鍖?+ 瀵圭О batch 鍓ョ | +2 |
| v5 | 40/46 (87.0%) | 澶嶅悎绠楀瓙鍒嗚В锛坢atmul_all_reduce锛?| +2 |

### 2.2 鎸夌被鍒尮閰嶆儏鍐?

| 绠楀瓙绫诲埆 | 鎬绘暟 | 鍖归厤 | 鍖归厤鐜?| 璇存槑 |
|---------|------|------|--------|------|
| 璁＄畻绠楀瓙锛圡atMul銆丯orm銆佹縺娲汇€丷oPE锛?| 10 | 10 | **100%** | 鍏ㄩ儴鍛戒腑 |
| 澶嶅悎璁＄畻锛坢atmul_all_reduce锛?| 2 | 2 | **100%** | 鍒嗚В鍚庡尮閰?MatMulV2 |
| 闆朵唬浠风畻瀛愶紙view銆乸ermute銆乻plit 绛夛級 | 28 | 28 | **100%** | 鏍囪涓?zero_cost |
| KV Cache (reshape_and_cache) | 1 | 0 | 0% | TC 涓?NPU 鎺ュ彛缁撴瀯宸紓 |
| Embedding (GatherV2) | 1 | 0 | 0% | 璇嶈〃 TP 鍒嗙墖宸紓 |
| 閫氫俊 (all_gather) | 1 | 0 | 0% | 闇€ CommGrid 甯﹀妯″瀷 |
| Attention (鐗规畩妯″紡) | 1 | 0 | 0% | 闇€涓撶敤鍖归厤閫昏緫 |
| 鍏朵粬 (index) | 2 | 0 | 0% | 鏃犲搴?Profiling 鏁版嵁 |

### 2.3 宸插尮閰嶈绠楃畻瀛愯鎯?

| TC 绠楀瓙 | NPU Kernel | 寤惰繜 (us) | 鍖归厤鏂规硶 |
|---------|-----------|-----------|---------|
| `aten.mm` (QKV 鎶曞奖) | MatMulV2 | 19.6 | FRACTAL_NZ 杩樺師 + padding |
| `aten.mm` (gate_up 鎶曞奖) | MatMulV2 | 59.7 | FRACTAL_NZ 杩樺師 + padding |
| `aten.mm` (lm_head) | MatMulV2 | 91.8 | ND 杞疆鍖归厤 |
| `matmul_all_reduce` (o_proj) | MatMulV2 (澶嶅悎) | 14.2 | 澶嶅悎鍒嗚В + FRACTAL_NZ |
| `matmul_all_reduce` (down_proj) | MatMulV2 (澶嶅悎) | 25.1 | 澶嶅悎鍒嗚В + FRACTAL_NZ |
| `rms_norm` x3 | RmsNorm | 21.7 / 20.1 / 7.7 | batch 鍓ョ + padding |
| `add_rms_norm2` | AddRmsNorm | 12.5 | batch 鍓ョ + padding |
| `swiglu` | SwiGlu | 14.9 | 杈撳叆鍚堝苟 + batch 鍓ョ |
| `apply_rope` | ApplyRotaryPosEmb | 12.5 | RoPE 褰掍竴鍖?+ 澶囬€?kernel |
| `aten.add` | Add | 16.2 | batch 鍓ョ + padding |

**姣忓眰鎬昏绠楀欢杩燂細316.0 us**锛堝熀浜庡疄娴?Profiling 鏁版嵁锛?

---

## 3. 鍏抽敭鍙戠幇

### 3.1 TC 涓?NPU Profiling 鐨?8 绫?Shape 宸紓

| # | 宸紓绫诲瀷 | TC 琛屼负 | Profiling 琛屼负 | 瑙ｅ喅鏂规 |
|---|---------|--------|---------------|---------|
| 1 | Batch 缁村害 | 淇濈暀 `(1, seq, dim)` | 鎵佸钩鍖?`(seq, dim)` | `_strip_batch_dim()` |
| 2 | Seq padding | `ceil(seq/16)*16` = 144 | 鍘熷 seq = 136 | block-padding 瀹瑰樊 |
| 3 | FRACTAL_NZ | TC 鐢?ND `(K, N)` | 鏉冮噸 `[H,W,bh,bw]` | `fractal_nz_to_nd()` |
| 4 | ND 杞疆 | `F.linear` 杞疆鍚?`(K,N)` | 瀛樺偍 `(N,K)` | MatMul 涓撶敤杞疆妫€鏌?|
| 5 | SwiGlu 杈撳叆 | 2 涓嫭绔嬭緭鍏?`(S,D/2)` | 1 涓瀺鍚堣緭鍏?`(S,D)` | 鎸夋湯缁存嫾鎺?|
| 6 | RoPE 甯冨眬 | `(B,H,S,D)` + Q/K 椤哄簭 | `(B,S,H,D)` + K/Q 椤哄簭 | `_normalize_rope_inputs()` |
| 7 | RoPE kernel | 鍚屼竴 TC op 瀵瑰簲涓嶅悓 kernel | neox鈫扐pplyRotaryPosEmb | `alternate_kernel_types` |
| 8 | 澶嶅悎绠楀瓙 | matmul+allReduce 铻嶅悎 | 鍙兘鐙珛鎴?MC2 铻嶅悎 | `_lookup_composite()` |

### 3.2 op_mapping.yaml 鎵╁睍鏈哄埗

绌垮埡杩囩▼涓负 op_mapping.yaml 寮曞叆浜?3 涓柊瀛楁锛?

```yaml
# 澶囬€?kernel 绫诲瀷锛堝悓涓€ TC op 鍦ㄤ笉鍚屾ā鍨?閰嶇疆涓嬫槧灏勪笉鍚?kernel锛?
alternate_kernel_types: [ApplyRotaryPosEmb]

# 闆朵唬浠锋爣璁帮紙绾?shape 鍙樻崲锛屾棤纭欢鎵ц锛?
zero_cost: true

# 澶嶅悎绠楀瓙鍒嗚В
composite: true
sub_kernels: [MatMulV2, hcom_allReduce_]
```

### 3.3 `--compile` 瀵圭畻瀛愬尮閰嶇殑鍏抽敭褰卞搷

涓嶅姞 `--compile` 鏃讹紝TC 灏嗚瀺鍚堢畻瀛愶紙RmsNorm銆丼wiGlu銆丷oPE锛夊垎瑙ｄ负 72+ 涓?aten 鍘熻锛屽鑷存棤娉曞尮閰嶅埌 Profiling 涓殑铻嶅悎 kernel銆?*`--compile` 涓庨噺鍖栨棤鍏筹紙BF16 涔熼渶瑕侊級锛屾槸姝ｇ‘浣跨敤 `--performance-model profiling` 鐨勫墠鎻愭潯浠?*銆?

---

## 4. 鏈В鍐抽棶棰樺垎鏋?

### 4.1 Attention 鐗规畩妯″紡锛圥1锛屽奖鍝嶆渶澶э級

`FusedInferAttentionScore` 鏄?Prefill 涓崟娆″欢杩熸渶楂樼殑绠楀瓙锛垀100us 绾у埆锛夛紝浣嗗叾杈撳叆缁撴瀯澶嶆潅锛圦銆並 cache銆乂 cache銆乵ask銆乻eq_lens 绛?7 涓緭鍏ワ級锛岄渶瑕佷笓鐢ㄧ殑 shape 鍖归厤閫昏緫锛屽寘鎷細
- seq 缁村害鍔ㄦ€侊細渚濊禆 num_queries x query_length
- block 缁村害锛氫緷璧?KV cache block_size锛?28锛?
- head 缁村害锛氶渶瑕佹寜 TP 鍒嗙墖鍚庣殑 num_heads 鍖归厤

### 4.2 閫氫俊绠楀瓙锛圥2锛屾灦鏋勮璁￠棶棰橈級

`hcom_allReduce_` 鍜?`hcom_allGather_` 鐨勫欢杩熶笉鍙栧喅浜?tensor shape锛岃€屽彇鍐充簬锛?
- 娑堟伅澶у皬锛坆ytes锛?
- 鎷撴墤缁撴瀯锛坮ing/mesh/full-mesh锛?
- 閫氫俊缁勫ぇ灏忥紙world_size锛?

璁捐鏂囨。 搂4.4 宸茶鍒?`CommDataSource`锛岄渶缁撳悎 `CommGrid` 鐨勫甫瀹藉弬鏁板拰 HCCL 瀹炴祴鏁版嵁銆?

### 4.3 reshape_and_cache锛圥3锛孴C 鎺ュ彛宸紓锛?

TC 鐨?`reshape_and_cache` op 鎺ュ彛涓?NPU kernel锛坄ReshapeAndCacheNdKernel`锛夊樊寮傝繃澶э細
- 杈撳叆鏁伴噺涓嶅悓锛? vs 5锛?
- KV tensor 缂哄皯 head 缁村害
- cache 缁撴瀯涓嶅悓

**鏍规湰鍘熷洜**锛歍C 涓轰簡閫氱敤鎬т娇鐢ㄧ畝鍖栨帴鍙ｏ紝鑰?NPU kernel 闇€瑕佸畬鏁寸殑 paged attention cache 淇℃伅銆?

### 4.4 Embedding 璇嶈〃鍒嗙墖锛圥4锛岃緝鏄撹В鍐筹級

TC 鍙戦€佸叏閲忚瘝琛?`(151936, 5120)`锛孭rofiling 瀛樺偍 TP 鍒嗙墖鍚庣殑 `(9496, 5120)` = 151936/16銆傝В鍐虫€濊矾锛氬湪 `_inputs_match` 涓姞鍏?embedding-aware 鐨勮瘝琛ㄧ淮搴︾缉鏀鹃€昏緫锛堥渶瑕佷紶鍏?TP size锛夈€?

---

## 5. 绌垮埡涓殑绠€鍖栧疄鐜版竻鍗?

绌垮埡杩囩▼涓叡鏈?17 澶勭畝鍖栧疄鐜帮紝鎸夊眰闈㈠垎绫诲涓嬨€傛瘡椤规爣娉ㄤ骇鍝佸寲鎵€闇€鐨勫崌绾ф柟鍚戝拰瀵瑰簲鐨勮璁℃枃妗ｇ珷鑺傘€?

### 5.1 Shape 鍖归厤灞傞潰

#### S-1: Block-Padding 瀹瑰樊 鈥?纭紪鐮佸榻愬€?

**鐜扮姸**: `_BLOCK_SIZES = (16, 32, 64)` 纭紪鐮佷笁绉嶅榻愬€硷紝TC 缁村害鍙鏄?CSV 缁村害鎸夎繖浜涘€?ceil 瀵归綈鐨勭粨鏋滃氨绠楀尮閰嶃€?

**椋庨櫓**: 瀹為檯 NPU tile 瀵归綈绛栫暐鏇村鏉傦紙涓嶅悓绠楀瓙銆佷笉鍚?dtype 鍙兘鏈変笉鍚?tile size锛夛紝鍙兘浜х敓璇尮閰嶆垨婕忓尮閰嶃€?

**浜у搧鍖栨柟鍚?*: 鏍规嵁 kernel_type + dtype 纭畾鍑嗙‘鐨?tile size锛圖a Vinci Cube: BF16=16x16, INT8=16x32锛夈€傚彲鍦?op_mapping.yaml 涓坊鍔?`tile_alignment` 瀛楁銆?

#### S-2: Batch 缁村害鍓ョ 鈥?鏃犲樊鍒墺绂?leading dim=1

**鐜扮姸**: `_strip_batch_dim()` 瀵规墍鏈夌畻瀛愭棤宸埆鍦板墺绂?leading dim=1锛屼笉鍖哄垎璇ョ淮搴︽槸鐪熸鐨?batch 杩樻槸鍏朵粬璇箟銆?

**椋庨櫓**: 濡傛灉鏌愮畻瀛愮殑 leading dim=1 鏈夎涔夊惈涔夛紙涓嶆槸 batch锛夛紝浼氬鑷磋鍖归厤銆傜洰鍓嶅鎵€鏈?TC 鍜?CSV 杈撳叆閮藉仛瀵圭О鍓ョ銆?

**浜у搧鍖栨柟鍚戯紙璁捐鏂囨。 搂4.2锛?*: 搴斿湪 TC 灞傞潰鎴?EmpiricalPerformanceModel 缁熶竴 shape 褰掍竴鍖栵紝鑰岄潪鍦?DataSource 鍐呴儴閫愪釜澶勭悊銆?

#### S-3: RoPE 褰掍竴鍖?鈥?纭紪鐮?Q/K 閲嶆帓 + 杞疆瑙勫垯

**鐜扮姸**: `_normalize_rope_inputs()` 鍋囪 TC 姘歌繙鍙戦€?`[Q(B,H,S,D), K(B,H,S,D), cos, sin]`锛孋SV 姘歌繙鏄?`[K(B,S,H,D), Q(B,S,H,D), cos, sin]`銆傝緭鍏ラ『搴忓拰缁村害鎺掑垪瑙勫垯鏄唬鐮佺‖缂栫爜鐨勩€?

**椋庨櫓**: 涓嶅悓鐗堟湰 vLLM-ascend 鎴栦笉鍚?RoPE 妯″紡锛堝 GLM4 鐨?rotary_dim != head_dim锛夊彲鑳芥敼鍙樿緭鍏ョ粨鏋勩€?

**浜у搧鍖栨柟鍚?*: 鍦?op_mapping.yaml 涓敤澹版槑寮忚鍒欐弿杩?shape 鍙樻崲锛堝 `input_transform: [{permute: [0,2,1,3]}, {swap: [0,1]}]`锛夛紝鑰岄潪鍦?Python 浠ｇ爜涓‖缂栫爜銆?

#### S-4: SwiGlu 杈撳叆鍚堝苟 鈥?鍋囪 2鈫? 鍚堝苟

**鐜扮姸**: 鍋囪 TC 姘歌繙鍙?2 涓瓑褰㈢姸杈撳叆锛孋SV 姘歌繙瀛?1 涓部鏈淮鎷兼帴鐨勮緭鍏ャ€?

**椋庨櫓**: W8A8 DequantSwigluQuant 绛夊彉浣撶殑杈撳叆缁撴瀯涓嶅悓锛屾瑙勫垯涓嶉€傜敤銆?

**浜у搧鍖栨柟鍚?*: 涓?S-3 绫讳技锛岀敤澹版槑寮忚瀺鍚堟ā寮忔弿杩般€?

#### S-5: ND 杞疆鍖归厤 鈥?浠呭绗?2+ 涓緭鍏ュ皾璇?

**鐜扮姸**: `i >= 1 and fmt == "ND"` 鈥?鍙闈炵涓€涓?ND 鏍煎紡杈撳叆鍋?`(K,N) <-> (N,K)` 杞疆銆?

**椋庨櫓**: 鍋囪绗竴涓緭鍏ユ案杩滄槸 activation锛堜笉闇€杞疆锛夛紝闈炴爣鍑?matmul pattern 浼氬け鏁堛€?

### 5.2 op_mapping.yaml 鏄犲皠灞傞潰

#### S-6: `alternate_kernel_types` 鈥?鏆村姏鍥為€€鑰岄潪鏉′欢鍒嗘淳

**鐜扮姸**: RoPE 鐨?`alternate_kernel_types: [ApplyRotaryPosEmb]` 鏄?璇曞畬涓荤被鍨嬪啀璇曞閫?锛屼笉鐪?`is_neox` 绛夎繍琛屾椂鍙傛暟銆?

**浜у搧鍖栨柟鍚戯紙璁捐鏂囨。 搂4.5锛?*: 瀹炵幇 `kernel_type_variants` 鏉′欢鏄犲皠锛?
```yaml
kernel_type_variants:
  - condition: {is_neox: true}
    kernel_type: ApplyRotaryPosEmb
  - condition: {is_neox: false}
    kernel_type: InterleaveRope
```

#### S-7: `zero_cost: true` 鈥?绗肩粺鏍囪锛岄儴鍒嗗瓨鐤?

**鐜扮姸**: 14 涓?op 鏍囪涓?zero_cost锛屼絾鍏朵腑閮ㄥ垎瀛樼枒锛?
- `aten.copy_.default`锛氬疄闄呮湁鏁版嵁鎼Щ锛圞V cache update锛夛紝Profiling 涓彲鑳借〃鐜颁负 TensorMove kernel
- `aten.slice.Tensor`锛氳法姝ュ垏鐗囨湁瀹為檯寮€閿€
- `aten.arange.start`锛氭湁寰噺璁＄畻

**浜у搧鍖栨柟鍚?*: 鍖哄垎"鐪熸闆朵唬浠?锛坴iew銆乸ermute锛夊拰"杩戜技闆朵唬浠?锛坈opy_銆乻lice锛夛紝鍚庤€呭簲鏈変及绠楅€昏緫鎴栨煡璇?TensorMove.csv銆?

#### S-8: 閲忓寲鍙樹綋鏄犲皠鏈粡 Profiling 楠岃瘉

**鐜扮姸**: 浠ヤ笅鏄犲皠鍩轰簬 op-plugin 浠ｇ爜鍒嗘瀽鎺ㄥ锛屾棤瀹為檯 Profiling 鏁版嵁楠岃瘉锛?
- `fp8_linear` 鈫?`QuantBatchMatmulV3`锛團P8 涓撶敤 API 鍙兘涓嶅瓨鍦級
- `mxfp4_linear` 鈫?`QuantBatchMatmulV3`锛坧laceholder锛?
- `grouped_matmul_fp8_swiglu` 鈫?`DequantSwigluQuant`锛堣矾寰勪笉纭畾锛?
- 鎵€鏈?`*_all_reduce` 澶嶅悎绠楀瓙鐨?sub_kernels 鍒嗚В锛堝疄闄呭彲鑳借蛋 MC2 鍗?kernel锛?

**浜у搧鍖栨柟鍚?*: 闇€閲囬泦 FP8/MXFP4/W4A8 鍦烘櫙鐨?Profiling 鏁版嵁锛岄獙璇佹垨淇鏄犲皠銆?

#### S-9: MoE 璺敱绠楀瓙鏄犲皠渚濊禆鍦烘櫙鍋囪

**鐜扮姸**: `permute_tokens` 鏄犲皠鍒?`MoeDistributeDispatchV2`锛圗P 鍦烘櫙锛夛紝闈?EP 鍦烘櫙搴旀槧灏勫埌 `MoeInitRouting`銆傚綋鍓嶆棤鏉′欢鏄犲皠銆?

**浜у搧鍖栨柟鍚?*: 涓?S-6 鐩稿悓锛岄渶 `kernel_type_variants` 鎸?EP/闈?EP 鏉′欢閫夋嫨銆?

### 5.3 鏌ヨ閫昏緫灞傞潰

#### S-10: 澶嶅悎绠楀瓙鍒嗚В 鈥?鍙彇璁＄畻閮ㄥ垎锛屽拷鐣ラ€氫俊寤惰繜

**鐜扮姸**: `_lookup_composite()` 璺宠繃 `hcom_*` sub_kernel锛屽彧杩斿洖 MatMulV2 寤惰繜锛宑onfidence=0.8銆傞€氫俊閮ㄥ垎瀹屽叏浜ょ粰 analytic model銆?

**椋庨櫓**: 瀹為檯 MC2 鏄祦姘寸嚎铻嶅悎锛坢atmul 鍜?allReduce 閲嶅彔鎵ц锛夛紝latency != matmul + allReduce锛屽垎寮€浼扮畻浼?*楂樹及**鎬诲欢杩熴€?

**浜у搧鍖栨柟鍚?*: MC2 kernel 搴旀湁鐙珛鐨?Profiling CSV锛圱ype = MC2 涓撶敤 kernel锛夛紝鑰岄潪鍒嗚В銆傛垨鍦?`_lookup_composite` 涓缓妯℃祦姘寸嚎閲嶅彔銆?

#### S-11: Attention 瀹屽叏璺宠繃锛堣璁℃枃妗?搂4.2 query_mode锛?

**鐜扮姸**: `query_mode: attention_special` 鈫?鐩存帴杩斿洖 None銆俙FusedInferAttentionScore` 鏄欢杩熸渶楂樼殑鍗曠畻瀛愶紙~100us+锛夛紝瀵圭鍒扮绮惧害褰卞搷鏈€澶с€?

**浜у搧鍖栨柟鍚戯紙璁捐鏂囨。 搂4.2锛?*: 瀹炵幇 attention_special 鏌ヨ妯″紡锛氭彁鍙?(seq_len, num_heads, head_dim, block_size) 绛夊叧閿淮搴︼紝缁撳悎 FusedInferAttentionScore.csv 鍖归厤銆傞渶鍚屾椂澶勭悊 PA锛圥agedAttention锛夊拰 FA锛團lashAttention锛変袱绉嶆ā寮忋€?

#### S-12: 閫氫俊绠楀瓙瀹屽叏璺宠繃锛堣璁℃枃妗?搂4.4 CommDataSource锛?

**鐜扮姸**: `category: communication` 鈫?鐩存帴杩斿洖 None銆俙hcom_allReduce_.csv` 鍙瓨涓€涓钩鍧囧€?690us锛屾棤 shape 渚濊禆銆?

**浜у搧鍖栨柟鍚戯紙璁捐鏂囨。 搂4.4锛?*: 瀹炵幇 `CommDataSource`锛屽熀浜庢秷鎭ぇ灏?+ 鎷撴墤 + 閫氫俊缁勭殑甯﹀妯″瀷銆傞渶 HCCL benchmark 鏁版嵁锛坄hccl/{cann_version}/`锛夈€?

#### S-13: CSV 閫愯閬嶅巻鍖归厤锛屾棤绱㈠紩

**鐜扮姸**: `_inputs_match` 瀵?CSV DataFrame 閫愯 `iterrows()`锛孫(N) 鏆村姏鍖归厤銆傚綋鍓?CSV 鍙湁鍑犺锛屼笉褰卞搷鎬ц兘銆?

**浜у搧鍖栨柟鍚?*: 鏁版嵁閲忓ぇ鏃讹紙Microbenchmark 缃戞牸鍙揪鏁板崈琛岋級闇€寤虹珛 shape hash 绱㈠紩鎴栭鎺掑簭缁撴瀯銆?

### 5.4 dtype 鏄犲皠灞傞潰

#### S-14: FP16 = BF16 绛変环澶勭悊

**鐜扮姸**: `torch.float16: "DT_BF16"` 鈥?灏?FP16 瑙嗗悓 BF16銆侫scend A3 涓?BF16 鍜?FP16 鍏辩敤鐩稿悓 kernel 璺緞銆?

**椋庨櫓**: 濡傛灉鏈潵纭欢鎴?CANN 鍖哄垎 FP16/BF16 kernel 璺緞锛屼細瀵艰嚧 dtype 涓嶅尮閰嶃€?

#### S-15: 涓嶆鏌?output dtype/shape

**鐜扮姸**: `_inputs_match` 鍙鏌ヨ緭鍏?shape + dtype锛屽畬鍏ㄤ笉鐪嬭緭鍑恒€?

**椋庨櫓**: 鍚屼竴杈撳叆涓嶅悓杈撳嚭閰嶇疆锛堝 in-place vs out-of-place銆佷笉鍚岃緭鍑?dtype锛夊彲鑳芥湁涓嶅悓鎬ц兘銆?

### 5.5 鏁版嵁灞傞潰

#### S-16: 鍗曞満鏅暟鎹?+ 璺ㄦā鍨嬪鐢?

**鐜扮姸**: 鐢?Qwen3-30B Prefill锛圱P=16, seq=136锛夌殑 Profiling 鏁版嵁楠岃瘉 Qwen3-32B銆備袱鑰呭悓鏋舵瀯浣嗕笉鍚屽弬鏁帮紝闅愬惈鍋囪"鍚屾灦鏋?鈫?kernel 琛屼负鐩稿悓"銆?

**浜у搧鍖栨柟鍚?*: 闇€瑕佸 seq length銆佸 batch size 鐨?Profiling 鏁版嵁 + 鎻掑€笺€傚搴旇璁℃枃妗?搂4.8 InterpolatingDataSource銆?

#### S-17: 鏃犳彃鍊?鈥?涓ユ牸绮剧‘鍖归厤

**鐜扮姸**: 涓ユ牸绮剧‘鍖归厤锛堝厑璁?padding 瀹瑰樊锛夛紝涓嶆敮鎸佸鏈 shape 杩涜鎻掑€间及绠椼€俿eq=200 灏辨棤娉曞尮閰嶃€?

**浜у搧鍖栨柟鍚戯紙璁捐鏂囨。 搂4.8锛?*: 瀹炵幇 `InterpolatingDataSource`锛屽弬鑰?AI Configurator 鐨?2D+1D 娣峰悎鎻掑€?+ sqrt 鍙樻崲锛圓ttention O(n^2) 绠楀瓙锛夈€?

---

## 6. 涓嬩竴姝ヨ鍔ㄥ缓璁?

缁撳悎绌垮埡缁撹銆佺畝鍖栧疄鐜版竻鍗曘€佷互鍙婅璁℃枃妗?v1.2 鐨勬暣浣撹鍒掞紝寤鸿鎸変互涓嬩笁涓樁娈垫帹杩涖€?

### 绗竴闃舵锛氭秷闄ゅ叧閿洸鍖猴紙1-2 鍛紝瀵瑰簲璁捐鏂囨。 搂4.2-4.8锛?

鐩爣锛氳В鍐崇┛鍒轰腑璺宠繃鐨?3 绫荤畻瀛?+ 瀹炵幇鎻掑€硷紝浣跨鍒扮浠跨湡鍙敤銆?

| 浼樺厛绾?| 浠诲姟 | 娑夊強绠€鍖栭」 | 璁捐鏂囨。 | 棰勮鏀剁泭 |
|--------|------|-----------|---------|---------|
| **P0** | **Attention 鐗规畩妯″紡鍖归厤** | S-11 | 搂4.2 query_mode | +1 HIT锛岃鐩栨渶楂樺欢杩熺畻瀛?|
| **P0** | **CommDataSource 閫氫俊甯﹀妯″瀷** | S-12 | 搂4.4 | +1 HIT锛岃В鍐?allReduce/allGather |
| **P0** | **InterpolatingDataSource 鎻掑€?* | S-17 | 搂4.8 | 鏀寔浠绘剰 seq length锛屼笉鍐嶄緷璧栫簿纭尮閰?|
| P1 | Embedding TP 鍒嗙墖 | 鈥?| 鈥?| +1 HIT |
| P1 | 鏉′欢鏄犲皠锛坘ernel_type_variants锛?| S-6, S-9 | 搂4.5 | 鏇夸唬 alternate_kernel_types 鏆村姏鍥為€€ |
| P2 | 绮剧粏鍖?zero_cost 鍒嗙被 | S-7 | 鈥?| 鍖哄垎鐪熼浂浠ｄ环 vs 杩戜技闆朵唬浠?|

**Attention 鍖归厤鍏蜂綋鏂规**锛?
1. 浠?`OpInvokeInfo` 鎻愬彇 `(seq_len, num_heads, head_dim, block_size, num_blocks)` 鍏抽敭缁村害
2. `FusedInferAttentionScore.csv` 宸叉湁鏁版嵁锛圦wen3 Prefill 67x锛夛紝寤虹珛澶氱淮鍖归厤
3. 鍖哄垎 PA锛圥agedAttention, decode锛夊拰 FA锛團lashAttention, prefill锛変袱绉嶆ā寮忕殑杈撳叆缁撴瀯

**InterpolatingDataSource 鍏蜂綋鏂规**锛堝弬鑰?AI Configurator锛夛細
1. Wrapper 妯″紡鍖呰 ProfilingDataSource锛氱簿纭懡涓?鈫?鐩存帴杩斿洖锛屾湭鍛戒腑 鈫?鎻掑€?
2. 婵€娲荤淮搴︼紙seq_len, batch锛夊仛绾挎€?鍙岀嚎鎬ф彃鍊?
3. Attention 绠楀瓙鍋?sqrt 鍙樻崲鍚庡啀鎻掑€硷紙O(n^2) 澶嶆潅搴︼級
4. 鍙 `interpolatable` 鏍囪鐨勭淮搴﹀仛鎻掑€硷紙op_mapping.yaml 宸叉湁 `interpolation_policy` 瀛楁锛?

### 绗簩闃舵锛欴SV3 Decode 鏀寔 + 绔埌绔獙璇侊紙2-3 鍛紝瀵瑰簲璁捐鏂囨。 搂5.2锛?

鐩爣锛氳鐩栫浜屼釜鐩爣妯″瀷 DeepSeekV3 Decode 鍦烘櫙锛岄獙璇佺鍒扮绮惧害 <15%銆?

| 浠诲姟 | 娑夊強绠€鍖栭」 | 璇存槑 |
|------|-----------|------|
| **DSV3 Decode 鏁版嵁闆嗘垚** | S-16 | `v0.14.0_dsv3_decode/` 鏁版嵁宸插氨缁紝闇€ W8A8 dtype 鏀寔 |
| **MoE 绠楀瓙鏄犲皠楠岃瘉** | S-8, S-9 | GroupedMatmul銆丮oeGatingTopK銆丏istributeDispatch/Combine 瀹為檯楠岃瘉 |
| **MC2 铻嶅悎 kernel** | S-10 | 纭 MC2 鍦?Profiling 涓殑瀹為檯琛ㄧ幇锛堝崟 kernel vs 鍒嗙锛夛紝璋冩暣澶嶅悎鍒嗚В閫昏緫 |
| **reshape_and_cache 缁撴瀯閫傞厤** | 鈥?| 鍒嗘瀽 TC 涓?NPU 鐨?KV cache 鎺ュ彛宸紓锛岄€夋嫨鏀?TC op 杩樻槸鍔犻€傞厤灞?|
| **绔埌绔簿搴﹂獙璇?* | 鈥?| 瀵规瘮 TC 浠跨湡缁撴灉 vs 瀹為檯 vLLM Profiling 绔埌绔欢杩燂紝鐩爣 <15% |
| **涓?develop 鍒嗘敮闆嗘垚** | 鈥?| 鍚堝苟 gitcode/develop 鐨?SwiGlu 铻嶅悎銆丟MM+SwiGlu 铻嶅悎绛夋柊 pass |
| **澹版槑寮?shape 鍙樻崲** | S-3, S-4 | 灏?RoPE/SwiGlu 鐨勭‖缂栫爜褰掍竴鍖栨敼涓?op_mapping.yaml 涓殑澹版槑寮忚鍒?|

**DSV3 Decode 鏂板绠楀瓙鏄犲皠娓呭崟**锛?
- `QuantBatchMatmulV3`锛?5006x锛夆€?W8A8 matmul锛岄渶 INT8 dtype 鏀寔
- `AscendQuantV2`锛?0004x锛夆€?static quantization
- `DequantSwigluQuant`锛?879x锛夆€?GMM+SwiGlu+Quant 涓夊悎涓€
- `GroupedMatmul`锛?756x锛夆€?MoE expert computation
- `InplaceAddRmsNorm`锛?002x锛夆€?AddRmsNorm in-place 鍙樹綋
- `TransposeBatchMatMul`锛?002x锛夆€?MLA absorb projections
- `InterleaveRope`锛?501x锛夆€?DeepSeek interleave RoPE
- `KvRmsNormRopeCache`锛?501x锛夆€?KV Norm+RoPE+Cache 铻嶅悎
- `MoeGatingTopK`锛?378x锛夆€?MoE routing
- `MoeDistributeDispatch/CombineV2`锛?378x each锛夆€?MoE token routing

### 绗笁闃舵锛氫骇鍝佸寲锛?-4 鍛紝瀵瑰簲璁捐鏂囨。 搂5.3-5.5锛?

鐩爣锛氳揪鍒板彲缁存姢銆佸彲鎵╁睍鐨勪骇鍝佽川閲忋€?

| 浠诲姟 | 娑夊強绠€鍖栭」 | 璁捐鏂囨。 | 璇存槑 |
|------|-----------|---------|------|
| **鏁版嵁閲囬泦鑷姩鍖?* | S-16 | 搂5.4 | Profiling 鈫?CSV 鈫?楠岃瘉鐨?CI 娴佹按绾?|
| **Microbenchmark 缃戞牸** | S-13 | 搂5.5 | 鐢熸垚璁＄畻绠楀瓙 microbench 鑴氭湰锛屾墿鍏?CSV 瑕嗙洊鑼冨洿 |
| **CSV 绱㈠紩浼樺寲** | S-13 | 鈥?| shape hash 绱㈠紩锛屾敮鎸佹暟鍗冭蹇€熸煡璇?|
| **澶氱増鏈鐞?* | 鈥?| 搂2.3 | CANN/vLLM-Ascend 鐗堟湰鏁版嵁鐩綍闅旂 |
| **op_mapping 鍒嗗眰** | 鈥?| 鈥?| `op_mapping_base.yaml` + 妯″瀷/鍦烘櫙 overlay |
| **Shape 褰掍竴鍖栧眰绾т笂绉?* | S-1, S-2 | 鈥?| 鍦?TC/EmpiricalPerformanceModel 灞傜粺涓€褰掍竴鍖?|
| **鐩存帴 vLLM op graph 鎶撳彇** | 鈥?| 搂璁捐鍘熷垯 | 缁曡繃 TC dispatch锛岀洿鎺ヤ粠 vLLM 瀹炶窇鎶撳彇绠楀瓙鍥?|
| **CompositePerformanceModel** | 鈥?| 搂9.3 | 澶?PerformanceModel 绾ц仈璋冨害鍣?|

### 鏋舵瀯寤鸿

1. **op_mapping.yaml 搴旀寜妯″瀷/鍦烘櫙鎷嗗垎**: 褰撳墠鍗曟枃浠?60+ 鏉℃槧灏勶紝闅忔ā鍨嬪澶氬皢鑶ㄨ儉銆傚缓璁?`op_mapping_base.yaml` + `op_mapping_qwen3.yaml` overlay 妯″紡銆?

2. **Shape 鍖归厤绠＄嚎闇€瑕佹洿濂界殑鍙娴嬫€?*: 褰撳墠 MISS 鍙湁 DEBUG 鏃ュ織锛屽缓璁鍔犵粨鏋勫寲鐨?match report 杈撳嚭锛堢被浼兼湰鎶ュ憡 搂2.3 琛ㄦ牸锛夛紝鏂逛究蹇€熷畾浣嶆柊妯″瀷鐨勫尮閰嶉棶棰樸€?

3. **ProfilingDataSource 涓嶅簲鎰熺煡 TC 鐨?batch/padding 琛屼负**: 褰撳墠 `_strip_batch_dim` 鍜?padding 瀹瑰樊鏄负浜嗗讥琛?TC 涓?Profiling 鐨勫樊寮傘€傞暱鏈熸柟鍚戝簲鍦?TC 鎴?EmpiricalPerformanceModel 灞傞潰缁熶竴 shape 褰掍竴鍖栵紝鑰岄潪鍦?DataSource 鍐呴儴閫愪釜澶勭悊銆?

4. **浼樺厛瀹炵幇 InterpolatingDataSource**: 绌垮埡渚濊禆绮剧‘ shape 鍖归厤 + padding 瀹瑰樊锛屼絾鐢熶骇鐜 seq length 鍙樺寲棰戠箒銆傛彃鍊兼槸瀹炵敤鎬х殑鍏抽敭鐡堕锛屽簲浼樺厛浜庡叾浠栦紭鍖栭」銆?

---

## 7. 浜や粯鐗╂竻鍗?

| 绫诲埆 | 鏂囦欢/璺緞 | 璇存槑 |
|------|----------|------|
| **鏍稿績浠ｇ爜** | `tensor_cast/performance_model/perf_database/data_source.py` | DataSource ABC |
| | `tensor_cast/performance_model/perf_database/profiling_data_source.py` | CSV 鏌ヨ + 8 绉?shape 鍖归厤瑙勫垯 |
| | `tensor_cast/performance_model/empirical.py` | EmpiricalPerformanceModel 閲嶆瀯 |
| | `tensor_cast/core/model_runner.py` | `--performance-model profiling` CLI |
| **鏁版嵁** | `.../v0.14.0/op_mapping.yaml` | 60+ 鏉＄畻瀛愭槧灏?|
| | `.../v0.14.0/*.csv` (45 涓? | Qwen3-30B Profiling 鏁版嵁 |
| **娴嬭瘯** | `tests/perf_database/` (34 涓祴璇? | 鍏ㄩ儴閫氳繃 |
| **鏂囨。** | `docs/perf_database/reports/qwen3_32b_alignment_report.md` | 璇︾粏瀵归綈鎶ュ憡 (v5) |
| | `docs/perf_database/reports/spike_executive_summary_zh.md` | 鏈枃妗?|
| | `docs/plans/2026-03-04-perf-database-spike.md` | 绌垮埡璁″垝 |
| | `docs/plans/2026-03-05-fix-shape-matching.md` | Shape 鍖归厤淇璁″垝 |

