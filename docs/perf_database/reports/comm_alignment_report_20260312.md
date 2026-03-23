# 閫氫俊绠楀瓙瀵归綈鎶ュ憡

**鍒涘缓鏃ユ湡**锛?026-03-12
**鏈€鍚庢洿鏂?*锛?026-03-19
**鐗堟湰**锛歷3.0锛坅lternating 妯″紡 + 鍥哄畾寮€閿€缁堢増锛?
**璐熻矗浜?*锛欻DY
**鏁版嵁鏉ユ簮**锛歈wen3-32B 澶?ISL profiling + DSV3 澶?ISL profiling锛坴LLM 0.15.0 + CANN 8.5 + AIV 妯″紡锛屽潎鍏抽棴 stack锛?
**Microbench 鐗堟湰**锛歨ccl/v8.5锛圕10-1 閲囬泦锛屽崟 session锛屽惈 tier=1/2锛宎lternating + kernel + event 涓夋ā寮忥級

---

## Executive Summary

| # | 鏍稿績缁撹 | 鏁版嵁鏀拺 |
|---|---------|---------|
| C1 | **alternating 妯″紡娑堥櫎 1-5MB 棰勭儹鍋忛珮** | Qwen3 1.3-5MB 浠庡亸楂?19-63% 闄嶅埌 卤3%锛汥SV3 3.5MB 浠庡亸楂?45-51% 闄嶅埌 卤3% |
| C2 | **鍥哄畾寮€閿€妯″瀷锛氫豢鐪熷€?= bench + 鍥哄畾寮€閿€** | Qwen3 AR +7.7us, AG +14.6us锛汥SV3 AG +1.2us, RS +2.0us锛涗笌 msg_bytes 鏃犲叧 |
| C3 | **HCCL 鍗忚鍒囨崲鐐?768KB nd=8** | per_device 鈮?60KB 澶?5x 璺冲彉锛沚ench 鏃犳硶澶嶇幇鐢熶骇鍊硷紙+107-147%锛夛紱鐢?profiler P50 |
| C4 | **bench CSV 绾€氫俊鏃堕棿鍑嗙‘** | kernel/bench ratio 0.59-0.82x锛汚IV 妯″紡瀵?HCCL 寰熀鍑嗘棤鏄捐憲褰卞搷 |
| C5 | **E2E 棰勬祴绛栫暐** | Prefill 鈮?MB: 鐩存帴鐢?alternating bench锛汥ecode 灏忔秷鎭? kernel bench + 鍥哄畾寮€閿€锛?68KB: profiler P50 |

---

## 2. 閲囬泦閰嶇疆

### 2.1 鐜閰嶇疆

| 妯″瀷 | TP | DP | 閲忓寲 | 鍏抽敭閰嶇疆 |
|------|----|----|------|---------|
| Qwen3-32B | 16 | 1 | BF16 | async-scheduling, CUDAGraph FULL_DECODE_ONLY, max-num-batched-tokens=65536 |
| DSV3 | 8 | 2 | W8A8 | async-scheduling, CUDAGraph FULL_DECODE_ONLY, max-num-seqs=8, max_num_batched_tokens=2048, EP |

鍏抽敭鐜鍙橀噺锛堜袱涓ā鍨嬪潎寮€鍚級锛?

```bash
HCCL_OP_EXPANSION_MODE="AIV"      # AIV 鍔犻€熼€氫俊
TASK_QUEUE_ENABLE=1                # 浠诲姟闃熷垪浼樺寲
VLLM_ASCEND_ENABLE_FUSED_MC2=1    # MC2 铻嶅悎锛圖SV3 TP 閫氫俊锛?
VLLM_ASCEND_ENABLE_FLASHCOMM1=1   # 鍚敤 Sequence Parallelism
```

### 2.2 璐熻浇鐭╅樀

**Qwen3-32B**锛? 缁勫満鏅級锛?

| 鍦烘櫙 | input_len | output_len | concurrency | request_rate | 璇存槑 |
|------|-----------|------------|-------------|-------------|------|
| Prefill | 1024 | 1 | 16 | 10 | 鐭?ISL prefill |
| Prefill | 2048 | 1 | 16 | 10 | 涓?ISL prefill |
| Prefill | 4096 | 1 | 16 | 10 | 鏍囧噯 ISL prefill |
| Prefill | 8192 | 1 | 16 | 10 | 闀?ISL prefill |
| Decode | 4096 | 1536 | 1 | 1 | 浣庡苟鍙?decode |
| Decode | 4096 | 1536 | 4 | 2 | 涓綆骞跺彂 decode |
| Decode | 4096 | 1536 | 16 | 2 | 涓苟鍙?decode |
| Decode | 4096 | 1536 | 32 | 4 | 楂樺苟鍙?decode |

**DSV3**锛?0 缁勫満鏅級锛?

| # | 鍦烘櫙 | input_len | output_len | concurrency | request_rate | 璇存槑 |
|---|------|-----------|------------|-------------|-------------|------|
| 1 | Prefill | 512 | 1 | 8 | 10 | 鐭?ISL锛宐atch 4 鏉?|
| 2 | Prefill | 1024 | 1 | 8 | 10 | batch 2 鏉?|
| 3 | Prefill | 2048 | 1 | 8 | 10 | batch 1 鏉?|
| 4 | Prefill | 4096 | 1 | 8 | 10 | 瑙﹀彂 chunked prefill |
| 5 | Prefill | 4096 | 1 | 1 | 1 | chunked prefill 涓茶鍩虹嚎 |
| 6 | Decode | 4096 | 1536 | 1 | 1 | 涓茶鍩虹嚎 |
| 7 | Decode | 4096 | 1536 | 2 | 1 | 浣庡苟鍙?|
| 8 | Decode | 4096 | 1536 | 4 | 2 | 涓苟鍙?|
| 9 | Decode | 4096 | 1536 | 8 | 4 | 婊¤浇 |
| 10 | Decode | 2048 | 1536 | 8 | 4 | 鐭?ISL 婊¤浇 |

---

## 3. Step Trace 姒傝

`Factor = Stage / (Computing + Communication_Not_Overlapped)`锛屾潵鑷?step_trace_time.csv銆?

| 妯″瀷 | 鍦烘櫙 | Stage(ms) | Computing(ms) | Comm_NO(ms) | Free(ms) | Factor | Free% |
|------|------|-----------|--------------|-------------|----------|--------|-------|
| Qwen3 | Prefill (ISL=8192) | 4375 | 2372 | 1594 | 409 | 1.10x | 9.3% |
| Qwen3 | Decode (c32r4) | 3196 | 2190 | 849 | 156 | 1.05x | 4.9% |
| DSV3 | Prefill (ISL=4096 c8) | 3540 | 3055 | 353 | 132 | 1.039x | 3.7% |
| DSV3 | Decode (c1) | 3407 | 1454 | 1798 | 156 | 1.048x | 4.6% |

**缁撹**锛氱敓浜х幆澧?overhead factor 鏁翠綋 1.04x-1.10x锛岃皟搴﹀紑閿€鍙帶銆?

### 3.1 Qwen3-32B 澶?ISL Step Trace

| 鍦烘櫙 | Stage(ms) | Computing(ms) | Comm_NO(ms) | Free(ms) | Factor | Free% |
|------|-----------|--------------|-------------|----------|--------|-------|
| Prefill ISL=1024 | 8200 | 3546 | 2690 | 1965 | 1.31x | 24.0% |
| Prefill ISL=2048 | 8304 | 3596 | 2696 | 2012 | 1.32x | 24.2% |
| Prefill ISL=4096 (rank3) | 3072 | 1096 | 1467 | 509 | 1.20x | 16.6% |
| Prefill ISL=8192 | 4375 | 2372 | 1594 | 409 | 1.10x | 9.3% |
| Decode c1r1 | 3115 | 2235 | 668 | 212 | 1.07x | 6.8% |
| Decode c4r2 | 3076 | 2109 | 775 | 193 | 1.07x | 6.3% |
| Decode c16r2 | 3083 | 2055 | 826 | 203 | 1.07x | 6.6% |
| Decode c32r4 | 3196 | 2190 | 849 | 156 | 1.05x | 4.9% |

### 3.2 DSV3 澶?ISL/澶?Concurrency Step Trace

| 鍦烘櫙 | Stage(ms) | Computing(ms) | Comm_NO(ms) | Free(ms) | Factor | Free% |
|------|-----------|--------------|-------------|----------|--------|-------|
| Prefill ISL=512 | 3352 | 2697 | 481 | 173 | 1.055x | 5.2% |
| Prefill ISL=1024 | 3241 | 2512 | 638 | 91 | 1.029x | 2.8% |
| Prefill ISL=2048 | 3464 | 2945 | 422 | 97 | 1.029x | 2.8% |
| Prefill ISL=4096 c8 | 3540 | 3055 | 353 | 132 | 1.039x | 3.7% |
| Prefill ISL=4096 c1 | 3437 | 2276 | 335 | 826 | 1.317x | 24.0% |
| Decode c1 | 3407 | 1454 | 1798 | 156 | 1.048x | 4.6% |
| Decode c2 | 3332 | 2063 | 917 | 352 | 1.118x | 10.6% |
| Decode c4 | 3487 | 2269 | 1028 | 191 | 1.058x | 5.5% |
| Decode c8 | 3466 | 1883 | 652 | 931 | 1.367x | 26.9% |
| Decode ISL=2048 c8 | 3432 | 2308 | 980 | 144 | 1.044x | 4.2% |

---

## 4. 閫氫俊绠楀瓙 Profiling 鑰楁椂

鍙?p10-p90 stable median锛宬ernel_details.csv 涓?`hcom_*` 绠楀瓙銆?

### 4.1 Qwen3-32B锛圱P=16, Dense, BF16, num_devices=16锛?

**Prefill 澶?ISL kernel_details**锛?

| ISL | 绠楀瓙 | Count | Stable Median(us) | P10(us) | P90(us) |
|-----|------|-------|-------------------|---------|---------|
| 1024 | allGather | 390 | 3,141 | 3,125 | 3,165 |
| 1024 | reduceScatter | 387 | 3,796 | 3,775 | 3,822 |
| 4096 (rank3) | allGather | 2,080 | 186 | 176 | 368 |
| 4096 (rank3) | reduceScatter | 2,064 | 433 | 223 | 703 |
| 8192 | allGather | 390 | 1,657 | 708 | 3,150 |
| 8192 | reduceScatter | 387 | 2,115 | 877 | 3,832 |

**Decode 澶?Concurrency kernel_details**锛?

| Concurrency | 绠楀瓙 | Count | Stable Median(us) |
|-------------|------|-------|-------------------|
| c1r1 | allReduce | 25,671 | 19.8 |
| c4r2 | allReduce | 25,800 | 21.7 |
| c16r2 | allReduce | 20,511 | 20.2 |
| c32r4 | allReduce | 14,190 | 24.1 |

### 4.2 DSV3锛圱P=8, DP=2, EP, W8A8, num_devices=8锛?

**Prefill kernel_details**锛堜富鍔?ISL锛夛細

| ISL | 绠楀瓙 | Count | Stable Median(us) | P10(us) | P90(us) |
|-----|------|-------|-------------------|---------|---------|
| 2048 | allGather | 1820 | 63.9 | 28.4 | 90.1 |
| 2048 | reduceScatter | 910 | 208.0 | 197.9 | 644.6 |
| 4096 c8 | allGather | 1820 | 63.1 | 28.2 | 88.5 |
| 4096 c8 | reduceScatter | 910 | 207.5 | 198.7 | 226.4 |

**Decode kernel_details**锛坈8 婊¤浇锛夛細

| 鍦烘櫙 | 绠楀瓙 | Count | Stable Median(us) | P10(us) | P90(us) |
|------|------|-------|-------------------|---------|---------|
| c8 | allGather | 2600 | 8.8 | 6.4 | 385.8 |
| c8 | reduceScatter | 1300 | 17.3 | 8.1 | 795.2 |

---

## 5. Bench vs 鐢熶骇 Profiler 鏁翠綋瀵规瘮

浠ヤ笅姹囨€?bench 瀹炴祴缁撴灉涓庣敓浜х幆澧?profiler trace view 涓€氫俊绠楀瓙 Duration 鐨勯€愬満鏅姣斻€俠ench 鏁版嵁鏉ヨ嚜涓夌閲囬泦妯″紡锛坅lternating / kernel / event锛夛紝profiler 鏁版嵁鏉ヨ嚜 operator_details 鐨?`Device Total Duration`锛堝榻?Comm_NO 璇箟锛夈€?

### 5.1 Qwen3-32B锛圱P=16, nd=16锛?

**Prefill**锛坆ench 妯″紡锛歛lternating锛夛細

| ISL | 绠楀瓙 | msg_bytes | bench(us) | profiler(us) | 鍋忓樊 | 鍒ゅ畾 |
|-----|------|-----------|-----------|-------------|------|------|
| 4096 | allGather | 1.3MB | 176.7 | 182.5 | -3% | PASS |
| 4096 | allGather | 3.8MB | 366.6 | 366.6 | 0% | PASS |
| 4096 | allGather | 5.0MB | 435.1 | 432.3 | +1% | PASS |
| 8192 | allGather | 8.8MB | 679.0 | 722.4 | -6% | PASS |
| 8192 | allGather | 21.4MB | 1,628.0 | 1,657.2 | -2% | PASS |
| 1024 | allGather | 40.0MB | 3,080.2 | 3,141.1 | -2% | PASS |
| 4096 | reduceScatter | 1.3MB | 216.3 | 221.2 | -2% | PASS |
| 4096 | reduceScatter | 3.8MB | 440.7 | 442.1 | 0% | PASS |
| 8192 | reduceScatter | 8.8MB | 808.5 | 887.2 | -9% | PASS |
| 8192 | reduceScatter | 21.4MB | 2,044.3 | 2,115.1 | -3% | PASS |
| 1024 | reduceScatter | 40.0MB | 3,664.6 | 3,795.8 | -3% | PASS |

**Decode**锛坆ench 妯″紡锛歬ernel / alternating profiler-fallback锛夛細

| Concurrency | 绠楀瓙 | msg_bytes | bench(us) | profiler(us) | diff(us) | 鍒ゅ畾 |
|-------------|------|-----------|-----------|-------------|----------|------|
| c=1 | allReduce | 160KB | 12.1 | 19.8 | +7.7 | 鍥哄畾寮€閿€ |
| c=4 | allGather | 74KB | 12.5 | 25.9 | +13.3 | 鍥哄畾寮€閿€ |
| c=16 | allGather | 278KB | 31.3 | 46.5 | +15.3 | 鍥哄畾寮€閿€ |
| c=16 | allGather | 297KB | 33.0 | 48.2 | +15.2 | 鍥哄畾寮€閿€ |

### 5.2 DSV3锛圱P=8, nd=8锛?

**Prefill**锛坆ench 妯″紡锛歛lternating锛夛細

| ISL | 绠楀瓙 | msg_bytes | bench(us) | profiler(us) | 鍋忓樊 | 鍒ゅ畾 |
|-----|------|-----------|-----------|-------------|------|------|
| 2048 | allGather | 3.5MB | 166.9 | 172.0 | -3% | PASS |
| 2048 | reduceScatter | 3.5MB | 202.1 | 197.9 | +2% | PASS |

**Decode**锛坆ench 妯″紡锛歬ernel锛宑=8 婊¤浇锛夛細

| 绠楀瓙 | msg_bytes | bench(us) | profiler(us) | diff(us) | 鍒ゅ畾 |
|------|-----------|-----------|-------------|----------|------|
| allGather | 1KB | 5.4 | 6.4 | +1.0 | 鍥哄畾寮€閿€ |
| allGather | 7KB | 5.5 | 6.8 | +1.4 | 鍥哄畾寮€閿€ |
| allGather | 14KB | 5.6 | 7.0 | +1.3 | 鍥哄畾寮€閿€ |
| reduceScatter | 14KB | 6.1 | 8.1 | +2.0 | 鍥哄畾寮€閿€ |

### 5.3 瀵规瘮灏忕粨

鏁翠綋瀵规瘮鍛堢幇涓ょ娓呮櫚妯″紡锛?

| 妯″紡 | 鍦烘櫙 | 鐗瑰緛 | 澶勭悊鏂瑰紡 |
|------|------|------|---------|
| **姣斾緥涓€鑷?* | Prefill 鈮?MB | bench 鈮?profiler锛堝亸宸?卤6%锛?| 鐩存帴鐢?bench |
| **鍥哄畾鍋忕Щ** | Decode 灏忔秷鎭?| profiler = bench + 鍥哄畾鍊?| bench + 鍥哄畾寮€閿€ |

渚嬪锛欴SV3 allGather 768KB nd=8锛宐ench 鍋忛珮 107-147%锛圚CCL 鍗忚鍒囨崲鐐癸紝瑙?搂8锛夈€?

---

## 6. Bench 娴嬭瘯楠岃瘉

涓婅妭鏁翠綋瀵规瘮琛ㄦ槑 bench 鍦?prefill 澶ф秷鎭満鏅彲鐩存帴浣跨敤銆傛湰鑺傝仛鐒﹂獙璇?alternating 妯″紡鐩告瘮鏃?event 妯″紡鐨勬敼鍠勬晥鏋溾€斺€旀棫 event 妯″紡鍦?1-5MB 鍖洪棿绯荤粺鎬у亸楂?19-63%锛宎lternating 妯″紡灏嗗亸宸帇缂╁埌 卤6%銆?

### 6.1 Qwen3 prefill allGather锛圱P=16锛?

| 鍦烘櫙 | msg_bytes | bench(us) | prof(us) | 鍋忓樊 | 鏃?event 鍋忓樊 |
|------|-----------|-----------|---------|------|-------------|
| ISL=4096 | 1.3MB | 176.7 | 182.5 | -3% | +63% |
| ISL=4096 | 3.8MB | 366.6 | 366.6 | 0% | +30% |
| ISL=4096 | 5.0MB | 435.1 | 432.3 | +1% | +23% |
| ISL=8192 | 8.8MB | 679.0 | 722.4 | -6% | +8% |
| ISL=8192 | 21.4MB | 1628.0 | 1657.2 | -2% | +4% |
| ISL=1024 | 40.0MB | 3080.2 | 3141.1 | -2% | 0% |

### 6.2 Qwen3 prefill reduceScatter锛圱P=16锛?

| 鍦烘櫙 | msg_bytes | bench(us) | prof(us) | 鍋忓樊 | 鏃?event 鍋忓樊 |
|------|-----------|-----------|---------|------|-------------|
| ISL=4096 | 1.3MB | 216.3 | 221.2 | -2% | +37% |
| ISL=4096 | 3.8MB | 440.7 | 442.1 | 0% | +19% |
| ISL=8192 | 8.8MB | 808.5 | 887.2 | -9% | +2% |
| ISL=8192 | 21.4MB | 2044.3 | 2115.1 | -3% | -4% |
| ISL=1024 | 40.0MB | 3664.6 | 3795.8 | -3% | -1% |

### 6.3 DSV3 prefill锛圱P=8锛?

| 鍦烘櫙 | 绠楀瓙 | msg_bytes | bench(us) | prof(us) | 鍋忓樊 | 鏃?event 鍋忓樊 |
|------|------|-----------|-----------|---------|------|-------------|
| ISL=2048 | allGather | 3.5MB | 166.9 | 172.0 | -3% | +51% |
| ISL=2048 | reduceScatter | 3.5MB | 202.1 | 197.9 | +2% | +45% |

### 6.4 灏忕粨

- alternating 妯″紡鍦?1.3-5MB 鍖洪棿灏嗗亸宸粠 +19%~+63% 鍘嬬缉鍒?卤3%
- 澶ф秷鎭紙鈮?.8MB锛夊亸宸?卤6%锛屼笌鏃?event 妯″紡宸紓涓嶅ぇ锛堝ぇ娑堟伅鏈韩棰勭儹褰卞搷灏忥級
- DSV3 3.5MB 浠?+45-51% 闄嶅埌 卤3%锛屾敼鍠勬渶涓烘樉钁?
- **缁撹锛歛lternating 妯″紡搴斾綔涓?prefill 澶ф秷鎭?bench 鐨勯粯璁ら噰闆嗘ā寮?*

---

## 7. 鍥哄畾寮€閿€鍒嗘瀽

Decode 灏忔秷鎭満鏅紝bench 涓?profiler 涔嬮棿瀛樺湪涓?msg_bytes 鏃犲叧鐨勫浐瀹氬樊鍊硷紝浠跨湡鏃跺簲鍦?bench 鍊间笂鍙犲姞銆?

| 绠楀瓙 | TP | 鍥哄畾寮€閿€(us) | 鏁版嵁鐐?| 鏉ユ簮 |
|------|----|-------------|--------|------|
| allReduce | 16 | +7.7 | 4锛圦wen3 c1-c32锛?| profiler - bench 宸€?|
| allGather | 16 | +14.6 | 3锛圦wen3 c4/c16锛?| profiler - bench 宸€?|
| allGather | 8 | +1.2 | 4锛圖SV3 c8, 1-14KB锛?| profiler - bench 宸€?|
| reduceScatter | 8 | +2.0 | 1锛圖SV3 c8, 14KB锛?| profiler - bench 宸€?|

鏍稿績缁撹锛?

1. **鍥哄畾寮€閿€涓?TP 鏁伴噺姝ｇ浉鍏筹紝涓庡叿浣撴ā鍨嬫棤鍏?*锛歍P=16 鐨勫紑閿€锛?.7-14.6us锛夋樉钁楅珮浜?TP=8锛?.2-2.0us锛夛紝鍥犱负鏇村 rank 鍙備笌閫氫俊鏃?group lookup 鍜?barrier 鍚屾閾捐矾鏇撮暱
2. 寮€閿€涓?msg_bytes 鏃犲叧锛屾槸鐢熶骇鐜 vLLM scheduler dispatch 鈫?c10d 鍖呰 鈫?HCCL group lookup 鈫?stream sync 绛夎皟搴﹂摼璺殑鍥哄畾寤惰繜
3. 浠跨湡鍏紡锛歚浠跨湡鍊?= bench(msg_bytes) + 鍥哄畾寮€閿€(TP)`

---

## 8. HCCL 鍗忚鍒囨崲鐐?

> DSV3 nd=8 allGather 768KB 鍦ㄦ墍鏈?bench 妯″紡涓嬮兘杩滈珮浜庣敓浜?profiler锛屽睘浜?HCCL 鍐呴儴鍗忚鍒囨崲瀵艰嚧鐨勪笉鍙鐜板尯闂淬€?

### 8.1 768KB 涓夋ā寮?bench vs 鐢熶骇 profiler

| bench 妯″紡 | 768KB nd=8 (us) | 鐢熶骇 profiler (us) | 鍋忛珮 |
|-----------|----------------|-------------------|------|
| alternating | 186.4 | 75.4 | +147% |
| kernel (profiler) | 156.4 | 75.4 | +107% |
| event | 160.0 | 75.4 | +112% |

涓夌妯″紡鍧囧亸楂?107-147%锛岃鏄庤繖涓嶆槸 bench 妯″紡闂锛岃€屾槸 HCCL 鍦ㄧ嫭绔?microbench 涓庣敓浜х幆澧冧腑閫夋嫨浜嗕笉鍚岀殑浼犺緭鍗忚銆?

### 8.2 nd=8 allGather kernel Duration 璺冲彉

| msg_bytes | per_device | kernel Duration(us) |
|-----------|-----------|---------------------|
| 465KB | 58KB | 30.4 |
| 596KB | 75KB | 155.3 鈫?5x 璺冲彉 |
| 768KB | 96KB | 156.4 |
| 946KB | 118KB | 171.0 |

per_device 鈮?60KB 澶勫彂鐢?5x 璺冲彉锛屽搴?HCCL 浠庡皬娑堟伅鍗忚锛堝 recursive halving-doubling锛夊垏鎹㈠埌澶ф秷鎭崗璁紙濡?ring锛夈€傜敓浜х幆澧冧腑 768KB 鍙兘浠嶈蛋灏忔秷鎭崗璁紙鍥?pipeline 涓婁笅鏂囦笉鍚岋級锛屽鑷?bench 鏃犳硶澶嶇幇銆?

### 8.3 澶勭悊绛栫暐

768KB 灞炰簬鍗忚鍒囨崲涓嶅彲澶嶇幇鍖洪棿锛?*涓嶄娇鐢?bench 鍊硷紝鏀圭敤 profiler P50锛?5.4us锛?*銆?

---

## 9. 浠跨湡绛栫暐鎬昏〃

| 妯″瀷 | 闃舵 | 绠楀瓙 | msg_bytes 鑼冨洿 | 绛栫暐 | 鏁版嵁鏉ユ簮 |
|------|------|------|--------------|------|---------|
| Qwen3 | decode | allReduce | 鎵€鏈?| bench + 鍥哄畾寮€閿€ | alternating (profiler fallback) + 7.7us |
| Qwen3 | decode | allGather | 鎵€鏈?| bench + 鍥哄畾寮€閿€ | kernel (profiler) + 14.6us |
| Qwen3 | prefill | allGather | 鈮?.26MB | 鐩存帴鐢?bench | alternating锛堣宸?卤6%锛墊
| Qwen3 | prefill | reduceScatter | 鈮?.26MB | 鐩存帴鐢?bench | alternating锛堣宸?卤3%锛墊
| DSV3 | decode | allGather | 鈮?26KB (c=8) | bench + 鍥哄畾寮€閿€ | kernel (profiler) + 1.2us |
| DSV3 | decode | reduceScatter | 14KB (c=8) | bench + 鍥哄畾寮€閿€ | kernel (profiler) + 2.0us |
| DSV3 | prefill | allGather | 768KB | 鐢?profiler P50 | HCCL 鍗忚鍒囨崲鐐癸紝bench 鏃犳硶澶嶇幇 |
| DSV3 | prefill | allGather | 鈮?.5MB | 鐩存帴鐢?bench | alternating锛堣宸?卤3%锛墊
| DSV3 | prefill | reduceScatter | 鈮?.5MB | 鐩存帴鐢?bench | alternating锛堣宸?卤2%锛墊

**绛栫暐璇存槑**锛?

1. **鐩存帴鐢?bench**锛歛lternating 妯″紡 bench 鍊肩洿鎺ヤ綔涓轰豢鐪熼娴嬪€硷紝閫傜敤浜?prefill 澶ф秷鎭?
2. **bench + 鍥哄畾寮€閿€**锛歜ench 鍊?+ 妯″瀷/绠楀瓙鐗瑰畾鐨勫浐瀹氬紑閿€锛岄€傜敤浜?decode 灏忔秷鎭?
3. **profiler P50**锛氱洿鎺ヤ娇鐢ㄧ敓浜?profiling 鐨?P50 鍊硷紝浠呯敤浜?HCCL 鍗忚鍒囨崲涓嶅彲澶嶇幇鍖洪棿

---

## 10. 閬楃暀闂

| # | 闂 | 浼樺厛绾?| 琛屽姩 | 鐘舵€?|
|---|------|--------|------|------|
| P8 | DSV3 allToAll 灏佽鍦?DispatchFFNCombine | 涓?| 鎷嗗垎瀛?kernel | Open |

宸插叧闂細P1锛坈ommunication_profiled 鍐呰仈鍊?鈫?bench 鏂规宸蹭綔涓烘渶缁堟柟妗堬紝鍥哄畾寮€閿€涓?TP 鐩稿叧銆佷笌妯″瀷鏃犲叧锛岄€傜敤浜庡叾浠栨ā鍨嬶級銆丳2锛坈orrection_factor 鈫?宸茶鍥哄畾寮€閿€妯″瀷鏇夸唬锛夈€丳3/P4锛堟暟鎹啑浣?rank 宸紓锛孨oted锛夈€丳5锛圖ecode c8 Free 寮傚父锛夈€丳6锛?6x 鍙樺寲涓虹粺璁￠噺鍙樺寲锛夈€丳7锛坅llGather 2.2MB pipeline 鍖栵級銆丳9锛坆ench CSV 鎻掑€煎亸宸?3%锛夈€丳10锛?68KB 鍗忚鍒囨崲 鈫?build_comm_csv.py 宸插湪鍚庡鐞嗛樁娈靛皢 CSV 涓鍊兼浛鎹负 profiler P50锛孌ataSource 鏃犻渶棰濆 fallback锛夈€丳11锛堝浐瀹氬紑閿€鍙傛暟 鈫?宸茬‘璁や笌 TP 鏁伴噺鐩稿叧銆佷笌鍏蜂綋妯″瀷鏃犲叧锛屾棤闇€閫愭ā鍨嬮獙璇侊級銆?

**澶囨敞**锛欴SV3 浣庡苟鍙戯紙concurrency < max-num-seqs锛変笅瀛樺湪 chunked prefill 鎺掗槦鏁堝簲锛岄€氫俊绠楀瓙瀹為檯寤惰繜鍙楄皟搴︽帓闃熷奖鍝嶏紝bench 鍗曠畻瀛愰殧绂绘墽琛屾棤娉曞鐜般€傛湰鎶ュ憡鎵€鏈夋暟鎹拰绛栫暐鍧囧熀浜庢弧杞藉満鏅€傝鐜拌薄鍚庣画鍙敤浜庢寚瀵兼ā鍨嬭皟搴︿紭鍖栥€?

---

## 闄勫綍

### A. AIV 妯″紡楠岃瘉

AIV 妯″紡寰熀鍑?vs 鍘熼潪 AIV 寰熀鍑嗭紙num_devices=16, tier=1锛夛細

| 绠楀瓙 | 灏忔秷鎭噺(鈮?MB) | 澶ф秷鎭噺(鈮?6MB) | 缁撹 |
|------|---------------|----------------|------|
| allReduce | 鍙樺寲 <12% | 鍙樺寲 <1% | 鏃犳樉钁楀樊寮?|
| allGather | 鍙樺寲 <23% | 鍙樺寲 <1% | 灏忔秷鎭噺鏈夋尝鍔紝澶ф秷鎭噺涓€鑷?|
| reduceScatter | 鍙樺寲 <10% | 鍙樺寲 <1% | 鏃犳樉钁楀樊寮?|
| alltoallv | 鍙樺寲 <27% | 鍙樺寲 <18% | alltoallv 灏忔秷鎭噺 AIV 鐣ュ揩 |

**缁撹锛欰IV 妯″紡瀵圭嫭绔?HCCL 寰熀鍑嗘棤鏄捐憲鍔犻€熸晥鏋溿€?*

### B. bench CSV 鎻掑€奸獙璇?

| 绠楀瓙 | msg_bytes | 瀹炴祴(us) | 鎻掑€?us) | 鍋忓樊 |
|------|-----------|---------|---------|------|
| reduceScatter TP=8 | 3.5MB | 255.1 | 262.3 | 3% |
| allGather TP=8 | 3.5MB | 252.1 | 249.0 | 1% |

### C. 涓夊眰 Duration 妯″瀷鍙傝€?

閫氫俊绠楀瓙鑰楁椂瀛樺湪涓変釜娴嬮噺灞傜骇锛歬ernel_details < bench < operator_details銆?

| 灞傜骇 | 娴嬮噺鏂瑰紡 | 鍖呭惈鍐呭 |
|------|---------|---------|
| kernel_details Duration | Ascend Profiler NPU timeline | 浠?NPU 涓?HCCL kernel 鎵ц鐗囨 |
| bench Duration | host 绔?perf_counter + synchronize | HCCL dispatch + NPU 鎵ц + host-device sync |
| operator_details Duration | Ascend Profiler operator timeline | 鍚?HCCL 鍚屾绛夊緟銆乻tream 绠＄悊銆乺ank barrier |

Prefill kernel/bench ratio 0.59-0.82x锛岀鍚?kernel < bench 棰勬湡銆?

### D. DSV3 鐗规湁娉ㄦ剰浜嬮」

1. DSV3 閫氫俊绠楀瓙涓?`hcom_allGather_` / `hcom_reduceScatter_`锛屾棤 `allgatherAicpuKernel` 璺緞
2. MC2 铻嶅悎锛氭墍鏈夊満鏅棤鐙珛 allReduce锛孴P 閫氫俊璧?MC2 铻嶅悎绠楀瓙
3. DispatchFFNCombine 灏佽 EP 閫氫俊锛歛lltoall 浠嶈灏佽锛屾棤娉曠洿鎺ュ姣?
4. Decode 閫氫俊闅忓苟鍙戝彉鍖栧墽鐑堬細reduceScatter 14KB 浠?c1=789us 鍒?c8=17us锛堝樊 46x锛?
5. Prefill ISL=2048 鍜?ISL=4096 閫氫俊鑰楁椂涓€鑷达細鍥?max-num-batched-tokens=2048 闄愬埗

