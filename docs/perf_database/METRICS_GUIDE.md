# 璇勪及鎸囨爣浣跨敤鎸囧崡 (M1鈥揗6)

鏈枃妗ｈ鏄庡浣曡繍琛屽拰瑙ｈ绠楀瓙鎬ц兘鏁版嵁搴撶殑鍏眰璇勪及鎸囨爣銆?

## 蹇€熷弬鑰?

| 鎸囨爣 | 涓€鍙ヨ瘽 | 闇€瑕佷粈涔?| 鐪嬩粈涔?|
|------|--------|---------|--------|
| M1 | 澶氬皯娆¤皟鐢ㄥ懡涓簡锛?| TC profiling 杩愯 | Debug 鐢紝琚?zero_cost 铏氬 |
| M2 | 澶氬皯涓€昏緫绠楀瓙鍛戒腑浜嗭紵 | TC profiling 杩愯 | GO/NO-GO 鍒ゅ畾 |
| M3 | 澶氬皯涓绠楃畻瀛愬懡涓簡锛?| TC profiling 杩愯 | **鏍稿績杩涘害鎸囨爣** |
| M4 | 澶氬皯涓?shape 鍙樹綋鍛戒腑浜嗭紵 | TC profiling 杩愯 | 瀹氫綅缂哄け shape锛屾寚瀵兼暟鎹噰闆?|
| M5 | 浠跨湡寤惰繜涓灏戞湁瀹炴祴鏁版嵁锛?| TC profiling 杩愯 | 寤惰繜鍔犳潈瑕嗙洊鐜?(浠跨湡瑙嗚) |
| M6 | Empirical 棰勬祴 vs 鐪熷疄鍗曟 forward pass | TC profiling 杩愯 + step_trace + kernel_details | **楠屾敹鏍囧噯锛?.85鈥?.15** |

## 杩愯 M1鈥揗5 (鍦ㄧ嚎鎸囨爣)

M1鈥揗5 鍦?TC profiling 妯″紡杩愯鏃惰嚜鍔ㄨ緭鍑哄埌鏃ュ織銆?

### 鍛戒护

```bash
# Qwen3-32B Prefill (BF16, TP=16)
python3.10 -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 10 --query-length 4104 \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --quantize-linear-action DISABLED \
  --performance-model profiling --compile \
  --perf-database tensor_cast/performance_model/perf_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5 \
  --log-level info

# Qwen3-32B Decode (BF16, TP=16)
python3.10 -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 16 --query-length 1 --context-length 4096 \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --quantize-linear-action DISABLED \
  --performance-model profiling --compile \
  --perf-database tensor_cast/performance_model/perf_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5 \
  --log-level info

# DSv3 Prefill (W8A8, TP=8, DP=2, EP=16)
python3.10 -m tensor_cast.scripts.text_generate deepseek-ai/DeepSeek-V3 \
  --num-queries 1 --query-length 256 \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 8 --dp-size 2 --ep-size 16 \
  --word-embedding-tp row \
  --quantize-linear-action W8A8_STATIC \
  --performance-model profiling --compile \
  --perf-database tensor_cast/performance_model/perf_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5 \
  --log-level info

# DSv3 Decode (W8A8, TP=8, DP=2, EP=16)
python3.10 -m tensor_cast.scripts.text_generate deepseek-ai/DeepSeek-V3 \
  --num-queries 16 --query-length 1 --context-length 4096 \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 8 --dp-size 2 --ep-size 16 \
  --word-embedding-tp row \
  --quantize-linear-action W8A8_STATIC \
  --performance-model profiling --compile \
  --perf-database tensor_cast/performance_model/perf_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5 \
  --log-level info
```

### 杈撳嚭绀轰緥

```
EmpiricalPerformanceModel: 35/46 ops matched (76.1%)              鈫?M1
  HITs (20 unique):
    aten.mm.default->MatMulV2 (x128)
    tensor_cast.all_reduce.default->hcom_allReduce_ (x64)
    ...
  MISSes (3 unique reasons):
    [shape_mismatch] kernel found, no matching shape in CSV: ...
    [unmapped] not in op_mapping.yaml: ...
Fused Op Match Rate: 10/20 (50.0%) [GO/NO-GO]                    鈫?M2
Fused Op Match Rate (excl zero_cost): 3/13 (23.1%) [Reference]   鈫?M3
Per-Shape Match Rate: 8/19 (42.1%)                                鈫?M4
  MISS shapes (11):
    aten.mm.default ((4096, 5120), (5120, 5120))
    tensor_cast.swiglu.default ((4096, 6912),)
    ...
Simulated Latency Coverage: 50.8% (12.723ms / 25.037ms)          鈫?M5
```

### 濡備綍瑙ｈ

- **M3 浣庝絾 M5 楂?*: 灏戦噺楂樺欢杩熺畻瀛?(MatMul, Attention) 宸插尮閰嶏紝浣嗗緢澶氫綆寤惰繜杈呭姪绠楀瓙鏈尮閰嶃€傝鐩栫巼鎸夊欢杩熺畻鍏跺疄涓嶅樊銆?
- **M4 鐨?MISS shapes 鍒楄〃**: 鐩存帴鍛婅瘔浣犻渶瑕佷负鍝簺 (绠楀瓙, shape) 缁勫悎閲囬泦 microbenchmark 鏁版嵁銆?
- **M5 鎺ヨ繎 M3**: analytic 妯″瀷璁や负鎵€鏈夌畻瀛愬欢杩熷樊涓嶅锛屾病鏈夌壒鍒獊鍑虹殑楂樺欢杩熺畻瀛愩€?
- **M5 杩滈珮浜?M3**: 宸插尮閰嶇殑灏戞暟绠楀瓙鎭板ソ鏄欢杩熷ぇ鎴枫€?

## 杩愯 M6 (鍗婄绾挎寚鏍?

```
M6 = Empirical_HIT_total / Real_per_forward_pass
```

M6 = 1.0 琛ㄧず瀹岀編棰勬祴銆侻6 > 1 = 楂樹及锛孧6 < 1 = 浣庝及銆侾hase 3 鐩爣锛?.85 鈮?M6 鈮?1.15銆?

**鍙惈 empirical HIT**锛堜笉鍚?analytic fallback 鐨?MISS ops锛夛紝鐩存帴琛￠噺宸叉湁 microbench 鏁版嵁鐨勮川閲忋€?

涓ゆ锛?
1. 璺?TC profiling 骞跺鍑?JSON锛歚--export-metrics report.json`
2. 鐢?`compute_m6.py` 璁＄畻 M6锛堥渶瑕?ASCEND_PROFILER_OUTPUT 鐩綍 + `--model` 鍙傛暟锛?

### 鍓嶇疆鏉′欢

1. 宸叉湁瀵瑰簲鍦烘櫙鐨?profiling 鏁版嵁锛堝寘鍚?`step_trace_time.csv` + `kernel_details.csv`锛?
2. 宸茶窇杩囧搴旂殑 TC profiling 鍛戒护骞跺鍑轰簡 metrics JSON

### 璁＄畻 M6

```bash
# Step 1: TC profiling + 瀵煎嚭 JSON (鍚?M1-M5 鍛戒护锛屽姞 --export-metrics)
python3.10 -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 10 --query-length 4104 \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --quantize-linear-action DISABLED \
  --performance-model profiling --compile \
  --perf-database $DATA_DIR \
  --export-metrics results/qwen3_prefill_metrics.json

# Step 2: 璁＄畻 M6 (--model 鐢ㄤ簬浼扮畻 forward pass 鏁伴噺)
python3.10 tools/perf_data_collection/compute_m6.py \
  --tc-report results/qwen3_prefill_metrics.json \
  --profiler-output /path/to/ASCEND_PROFILER_OUTPUT \
  --model qwen3
```

### 杈撳嚭绀轰緥

```
============================================================
M6: Empirical E2E Prediction Ratio
============================================================

Empirical HIT total:    609,642.0 us (609.6 ms)
Real per-fwd:         1,146,669.8 us (1,146.7 ms)
TC full prediction:   1,407,989.9 us (1,408.0 ms)  [for reference]
  Step total:         5,733,348.9 us (Computing: 3,242,347.5 + Comm: 2,491,001.4)
  Forward passes:               5   (anchor: FusedInferAttentionScore [320])

M6 = 0.532  (TC / Real)
     underestimate by 47%
Phase 3 target: 0.85 鈮?M6 鈮?1.15 [FAIL]
```

### 濡備綍瑙ｈ

- **M6 鈮?1.0**: empirical 鏁版嵁绮惧噯锛屽凡鍖归厤绠楀瓙鐨?microbench 寤惰繜涓庣湡瀹炶繍琛屼竴鑷?
- **M6 < 1 (濡?0.5)**: 浣庝及锛岃鏄?MISS ops 璐＄尞浜嗗ぇ閲忕湡瀹炲欢杩熶絾娌℃湁 empirical 鏁版嵁瑕嗙洊
- **M6 > 1 (濡?2.0)**: 楂樹及锛宮icrobench 鏁版嵁鍋忛珮锛坕solation vs real workload 宸紓锛夛紝鎴栭€氫俊 microbench 涓庣湡瀹?serving 宸窛澶?
- **M6 杩滃皬浜?1 浣?M5 楂?*: M5 (analytic鍔犳潈) 璁や负宸插尮閰嶇畻瀛愰噸瑕侊紝浣嗗疄闄呭畠浠湪鐪熷疄寤惰繜涓崰姣斿皬
- **unmatched 鍒楄〃**: 璇婃柇鍙傝€冿紝鏄剧ず鏈 empirical 瑕嗙洊鐨?kernel 鎸夋椂闂存帓搴?

### Forward Pass 浼扮畻璇存槑

`step_trace_time.csv` 鐨?`Step` 鍒椾负绌烘椂锛岃仛鍚堜簡鏁翠釜 profiling 绐楀彛锛堝彲鑳藉寘鍚暟鍗佸埌鏁扮櫨娆?forward pass锛夈€?
`compute_m6.py` 閫氳繃 `kernel_details.csv` 涓殑绠楀瓙璋冪敤璁℃暟鑷姩浼扮畻 forward pass 鏁伴噺锛?

| 妯″瀷 | Anchor Kernel | 姣忓眰姣?fwd pass 璋冪敤鏁?| 灞傛暟 |
|------|--------------|:---:|:---:|
| Qwen3 | FusedInferAttentionScore | 1 | 64 |
| DSv3 | DispatchFFNCombine | 1 | 58 (MoE layers) |

濡傛灉鑷姩浼扮畻涓嶅噯纭紝鍙娇鐢?`--n-forward-passes N` 鎵嬪姩鎸囧畾銆?

## 鎸囨爣鍏崇郴鍥?

```
绮楃矑搴?                                         缁嗙矑搴?
(绠楀瓙鏁伴噺)                                      (寤惰繜鍔犳潈)

M1 鈹€鈹€鈫?M2 鈹€鈹€鈫?M3 鈹€鈹€鈫?M4
 鈹?      鈹?      鈹?      鈹?
 鈹?   +鎮茶   -zero   +shape
 鈹?   +铻嶅悎    cost    鐙珛璁?
 鈹?
 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈫?M5 (analytic 寤惰繜鍔犳潈, 鍦ㄧ嚎)
                          鈹?
                          鈹斺啋 M6 (empirical-only ratio vs 鐪熷疄 per-fwd, 鍗婄绾?
                              鈫?
                              闇€瑕?step_trace_time.csv + kernel_details.csv
                              (浼扮畻 forward pass 鏁伴噺)
```

## Phase 鐩爣

| Phase | M3 鐩爣 | M5 鐩爣 | M6 鐩爣 |
|-------|:---:|:---:|:---:|
| Phase 1 (鉁? | 寤虹珛鎸囨爣 | 鈥?| 鈥?|
| Phase 2 (鉁? | > 50% | 寤虹珛鎸囨爣 | 寤虹珛鎸囨爣 |
| Phase 3 | 鈥?| > 80% | **0.85 鈮?M6 鈮?1.15** |

