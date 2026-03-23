# Phase 1 Profiling 鍒嗘瀽: Kernel Duration 涓庣鍒扮鏃堕棿鍏崇郴 (aclgraph)

**杞欢鏍?*: CANN 8.5 / vLLM 0.15.0 / PyTorch 2.9.0 / **aclgraph** (FULL_DECODE_ONLY cudagraph)
**纭欢**: Atlas 800 A3 (Ascend 910B)
**妯″瀷**: DeepSeek-V3 (W8A8, TP=8/DP=2/EP) + Qwen3-32B (BF16, TP=16)
**鏁版嵁**: `/mnt/d/Data/Profiling/Profiling-0313-phase1-e2e-test/`

---

## 鏍稿績缁撹

1. **e2e = Computing + Comm(Not Overlapped) + Free 鍦ㄦ墍鏈夊満鏅簿纭垚绔?* (step_trace 楠岃瘉)
2. **Compute/Comm Overlap 浠嶇劧鍙拷鐣?* 鈥?鏈€澶?13.7ms (Qwen3 Decode), 鍗?comm 浠?1%
3. **aclgraph 鐨?Free/CPU gap** 涓?eager 妯″紡涓嶅悓: PandD 鍦烘櫙鐨?Free 鍖呭惈 step 闂寸┖闂? 涓嶈兘鐩存帴瀵规瘮
4. **aclgraph 涓?HCCL 閫氫俊浠?AivKernel 褰㈠紡鍑虹幇鍦ㄧ嫭绔?stream 涓?*, kernel_details.csv 闇€鎸夋璇嗗埆

---

## 涓€銆乤clgraph 妯″紡鐨?Stream 缁撴瀯

aclgraph 妯″紡涓?kernel_details.csv 鐨?stream 鍒嗗竷涓?eager 妯″紡鏈夐噸瑕佸樊寮?

### 1.1 DSv3 (Prefill + Decode)

| Stream | 鍐呭 | Prefill 鏃堕棿 | Decode 鏃堕棿 |
|--------|------|-------------|-------------|
| **Stream 2** | 涓昏绠?(MatMul, Norm, Attention, DispatchFFNCombine) | 2,829 ms | 1,306 ms |
| **Stream 37** | AivKernel = **HCCL 閫氫俊鎵ц** | 100 ms | 3,425 ms |
| Stream 40 | Sampling (Sort, ArgMax, Neg) | 6 ms | 6 ms |
| Stream 69 | AICPU comm (reduce_scatter, allgather) | 275 ms | 鈥?|
| **NaN** | hcom_* 琛?(HCCL 瀹樻柟璁板綍, 涓?Stream 37 瀵瑰簲) | 1,361 ms | 3,425 ms |

**鍏抽敭鍙戠幇: Stream 37 涓婄殑 AivKernel = HCCL 閫氫俊**
- DSv3 Decode: Stream 37 (977 kernels, 3,425.1 ms) 鈮?NaN hcom (975 kernels, 3,425.1 ms) 鈥?**鏃堕棿瀹屽叏鍖归厤**
- DSv3 Prefill: Stream 37 浠?262 AivKernel (100ms), 鍥犱负 prefill 鐨勯€氫俊璧?AICPU 璺緞 (Stream 69)

### 1.2 Qwen3-32B (Prefill + Decode)

| Stream | 鍐呭 | Prefill 鏃堕棿 | Decode 鏃堕棿 |
|--------|------|-------------|-------------|
| **Stream 6/62** | 涓昏绠?(eager 璺緞) | 3,242 ms | 88 ms |
| **Stream 148** | Graph-compiled 璁＄畻 (decode 涓讳綋) | 鈥?| 1,429 ms |
| **Stream 149** | AivKernel = **HCCL 閫氫俊鎵ц** | 鈥?| 1,403 ms |
| Stream 67 | AivKernel (灏戦噺, prefill) | 0.2 ms | 10 ms |
| Stream 71 | Sampling | 3 ms | 47 ms |
| Stream 99 | AICPU comm (prefill only) | 5,572 ms | 鈥?|
| **NaN** | hcom_* 琛?| 2,491 ms | 1,413 ms |

**娉ㄦ剰:**
- Qwen3 Prefill: AICPU comm stream (5,572 ms) >> NaN hcom (2,491 ms). AICPU 鏃堕棿鍚瓑寰?HCCL 瀹屾垚鐨?idle銆?*step_trace 浣跨敤 NaN hcom 鏃堕棿 (2,491 ms) 浣滀负鏉冨▉ Communication 鍊笺€?*
- Qwen3 Decode: Stream 149 AivKernel (1,403 ms) 鈮?NaN hcom (1,413 ms)

### 1.3 涓?eager 妯″紡鐨?Stream 宸紓

| 鐗瑰緛 | Eager 妯″紡 | aclgraph 妯″紡 |
|------|-----------|--------------|
| HCCL 鎵ц stream | 鐙珛 HCCL stream (鏈?Stream ID) | AivKernel 鍦ㄧ嫭绔?stream (37/149) |
| hcom_* 鍙岄噸璁℃暟 | hcom (NaN) + AivKernel (HCCL stream) | hcom (NaN) + AivKernel (鏂?stream) |
| Graph-compiled kernel | 鏃?| Decode 鏈?hash 鍚庣紑 (Stream 148) |
| AICPU comm | Prefill 鏈?| 鍚?eager |

---

## 浜屻€乻tep_trace 瀹樻柟鍒嗚В

step_trace_time.csv 鐩存帴缁欏嚭 Computing / Communication / Overlapped / Free 鐨勬潈濞佸垎瑙?

| 鍦烘櫙 | Computing (ms) | Comm(Not Overlapped) (ms) | **Overlapped (ms)** | Free (ms) | **Stage/e2e (ms)** |
|------|-----------|-----------|-----------|------|------|
| DSv3 Prefill | 2,829 (62.1%) | 1,361 (29.9%) | **0.07 (~0%)** | 365 (8.0%) | **4,555** |
| DSv3 Decode | 1,306 (26.8%) | 3,425 (70.2%) | **0.27 (~0%)** | 149 (3.1%) | **4,880** |
| Qwen3 Prefill | 3,242 (55.7%) | 2,491 (42.8%) | **0.0 (0%)** | 83 (1.4%) | **5,816** |
| Qwen3 Decode | 1,532 (49.5%) | 1,399 (45.1%) | **13.7 (0.4%)** | 166 (5.4%) | **3,098** |

**楠岃瘉: e2e = Computing + Comm(Not Overlapped) + Free**
- DSv3 Prefill: 2829 + 1361 + 365 = 4,555 鉁?
- DSv3 Decode: 1306 + 3425 + 149 = 4,880 鉁?
- Qwen3 Prefill: 3242 + 2491 + 83 = 5,816 鉁?
- Qwen3 Decode: 1532 + 1399 + 166 = 3,098 鉁?(Overlap 13.7ms 琚垎鍒墸鍑?

---

## 涓夈€丆ompute / Comm Overlap 鍒嗘瀽

**鎵€鏈夊満鏅?Overlap 鍧囧彲蹇界暐:**

| 鍦烘櫙 | Overlapped | 鍗?Comm 姣斾緥 | 缁撹 |
|------|-----------|-------------|------|
| DSv3 Prefill | 0.07 ms | 0.005% | 鏃?overlap |
| DSv3 Decode | 0.27 ms | 0.008% | 鏃?overlap |
| Qwen3 Prefill | 0.0 ms | 0% | 鏃?overlap |
| Qwen3 Decode | 13.7 ms | **0.97%** | 寰噺 overlap, 鍙拷鐣?|

涓庝箣鍓?eager 鍒嗘瀽缁撹涓€鑷? **aclgraph 娌℃湁鍚敤 compute/comm pipeline**銆傛瘡灞備粛鐒舵槸 compute 鈫?allReduce 鈫?涓嬩竴灞?compute 鐨勪覆琛屾ā寮忋€?

鍘熷洜涓嶅彉: `multistream_overlap_shared_expert=false`, aclgraph 鍙墦鍖呬覆琛屽簭鍒? 鍑忓皯 CPU dispatch, 浣嗕笉鏀瑰彉 stream 闂翠緷璧栧叧绯汇€?

---

## 鍥涖€丆PU Gap / Free 鏃堕棿鍒嗘瀽

| 鍦烘櫙 | Free (ms) | Free% | 璇存槑 |
|------|----------|-------|------|
| DSv3 Prefill | 365 | 8.0% | 鍖呭惈 PandD step 闂寸┖闂?+ profiling overhead |
| DSv3 Decode | 149 | 3.1% | 姝ｅ父 CPU dispatch gap |
| Qwen3 Prefill | 83 | 1.4% | 鏋佷綆, 鍑犱箮鏃?gap |
| Qwen3 Decode | 166 | 5.4% | 鍖呭惈 PandD step 闂寸┖闂?|

涓庝箣鍓?eager 鏁版嵁瀵规瘮:

| 鍦烘櫙 | eager Free% | aclgraph Free% | 璇存槑 |
|------|-----------|---------------|------|
| DSv3 decode_b8 | 1.3% | 3.1% | aclgraph PandD 鍚?step 闂?idle |
| Qwen3 decode_b8 | 1.8% | 5.4% | 鍚屼笂 |
| Qwen3 prefill_4096 | 35.5% | 1.4% | eager 鐨?35% 鏄?profiling overhead |

> **娉ㄦ剰: aclgraph PandD 鍦烘櫙鐨?Free 涓嶈兘鐩存帴涓?eager 鍗曞満鏅姣?*, 鍥犱负 PandD 鐨?profiling trace 璺ㄥ涓?step, Free 鍖呭惈浜?step 闂寸殑璋冨害绌洪棽銆俀wen3 Prefill 鐨?1.4% 鏇磋兘浠ｈ〃瀹為檯 CPU gap 涓嬮檺銆?

---

## 浜斻€丳er-Stream 楠岃瘉

閫氳繃鎸?Stream ID 鍒嗙粍 kernel_details.csv 楠岃瘉 step_trace:

### 5.1 DSv3 Prefill

| 鏉ユ簮 | Compute (ms) | Comm (ms) |
|------|-------------|-----------|
| step_trace | 2,829 | 1,361 (NaN hcom) |
| Stream 鍒嗙粍 | 2,829 (Stream 2) | 1,361 (NaN hcom) |
| 鉁?**鍖归厤** | | |

鍒嗚В: Stream 2 = 璁＄畻 0.315s + DispatchFFNCombine 2.515s = 2.830s

### 5.2 DSv3 Decode

| 鏉ユ簮 | Compute (ms) | Comm (ms) |
|------|-------------|-----------|
| step_trace | 1,306 | 3,425 (NaN hcom) |
| Stream 鍒嗙粍 | 1,306 (Stream 2) + 6 (Stream 40) | 3,425 (Stream 37 AivKernel = NaN hcom) |
| 鉁?**鍖归厤** | | |

### 5.3 Qwen3 Prefill

| 鏉ユ簮 | Compute (ms) | Comm (ms) |
|------|-------------|-----------|
| step_trace | 3,242 | 2,491 (NaN hcom) |
| Stream 鍒嗙粍 | 3,242 (Stream 6) | **5,572 (Stream 99 AICPU)** 鈮?2,491 |

> AICPU stream 涓婄殑 allgatherAicpuKernel + reduce_scatterAicpuKernel 鎬绘椂闂?(5,572 ms) 杩滃ぇ浜庡疄闄?HCCL 鎵ц (2,491 ms). AICPU kernel 鐨?Duration 鍖呭惈浜嗙瓑寰?HCCL 瀹屾垚鐨勬椂闂淬€?*浣跨敤 NaN hcom 鏃堕棿 (= step_trace Communication) 鎵嶆槸姝ｇ‘鐨勯€氫俊鏃堕棿銆?*

### 5.4 Qwen3 Decode

| 鏉ユ簮 | Compute (ms) | Comm (ms) |
|------|-------------|-----------|
| step_trace | 1,532 | 1,413 (NaN hcom) |
| Stream 鍒嗙粍 | 1,429 (Stream 148) + 88 (Stream 62) + 47 (Stream 71) + 10 (Stream 67) = 1,574 | 1,403 (Stream 149 AivKernel) |

> Compute stream 鍚堣 (1,574ms) 鐣ュぇ浜?step_trace Computing (1,532ms), 宸紓鏉ヨ嚜 Stream 71 涓婄殑 sampling 绠楀瓙 (47ms) 鍙兘琚?step_trace 褰掑叆 Free銆?

---

## 鍏€佸 Perf-Database 浠跨湡鐨勬剰涔?

| 缁撹 | 璇存槑 |
|------|------|
| **浠跨湡鍏紡** | `e2e 鈮?危(compute_kernels) + 危(comm) + t_free` 鈥?aclgraph 涓嬩緷鐒舵垚绔?|
| **涓嶉渶瑕佸缓妯?overlap** | Overlap < 1% |
| **comm 鏃堕棿浣跨敤 NaN hcom** | Stream 涓婄殑 AICPU kernel 鍚瓑寰呮椂闂? 涓嶇瓑浜庡疄闄呴€氫俊 |
| **aclgraph Free 鍚?step 闂?idle** | 浠跨湡涓嶅簲鐩存帴鐢?PandD trace 鐨?Free 浣滀负 CPU gap |
| **Graph kernel 鍚嶇О闇€褰掍竴鍖?* | `MatMulV2_NDNZ_..._229955` 鈫?`MatMulV2` |
| **AivKernel = HCCL comm** | 鍦?kernel_details.csv 涓渶璇嗗埆骞舵纭垎绫?|
| **鍗曟 PandD trace 鍙彁渚?baseline** | 鏃犻渶澶氫釜鍗曞満鏅?profiling, PandD 鍗冲彲瑕嗙洊 prefill + decode |

---

## 闄勫綍

### A. 鏁版嵁鏂囦欢

姣忎釜鍦烘櫙鍖呭惈: `kernel_details.csv`, `step_trace_time.csv`, `op_statistic.csv`, `communication.json`, `communication_matrix.json`, `operator_details.csv`, `api_statistic.csv`, `trace_view.json`.

閰嶇疆鏂囦欢:
- `deepseekv3_vllm+bench閰嶇疆.txt`
- `qwen3-32b_vllm+bench閰嶇疆.txt`

### B. aclgraph Stream 璇嗗埆瑙勫垯

```
DSv3:
  Stream 2  鈫?Main compute (鍚?DispatchFFNCombine)
  Stream 37 鈫?AivKernel = HCCL comm
  Stream 40 鈫?Sampling
  Stream 69 鈫?AICPU comm (prefill only)
  NaN       鈫?hcom_* (authoritative comm timing)

Qwen3:
  Stream 6/62  鈫?Main compute (eager path)
  Stream 148   鈫?Graph-compiled compute (decode main)
  Stream 149   鈫?AivKernel = HCCL comm (decode)
  Stream 67    鈫?AivKernel (minimal, prefill)
  Stream 71    鈫?Sampling
  Stream 99    鈫?AICPU comm (prefill only)
  NaN          鈫?hcom_* (authoritative comm timing)
```

### C. 鍒嗘瀽鑴氭湰

```bash
python3.10 docs/perf_database/reports/profiling_analysis_aclgraph/analyze_phase1.py    # Stream 鍒嗙粍 + 绠楀瓙鍒嗙被
```

### D. 鐩稿叧鎶ュ憡

- 濮婂鎶ュ憡: [绠楀瓙绋冲畾鎬т笌鎺ュ叆鍙鎬(report_op_stability_zh.md)
- 鍓嶅簭鎶ュ憡 (eager): [kernel duration 涓?e2e](../profiling_analysis_kernel_vs_e2e_zh.md)

