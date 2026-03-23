# NPU Profiling 鍒嗘瀽鎶ュ憡 (浜?: Kernel Duration 涓庣鍒扮鏃堕棿鍏崇郴

**杞欢鏍?*: CANN 8.5 / vLLM 0.15.0 / PyTorch 2.9.0 / Eager + aclgraph 妯″紡
**纭欢**: Atlas 800 A3 (Ascend 910B)
**妯″瀷**: DeepSeek-V3 (W8A8, TP=8/DP=2/EP, 16 NPU) + Qwen3-32B (BF16, TP=16)
**鍒嗘瀽鑴氭湰**: `docs/perf_database/reports/profiling_analysis_eager/`

---

## 鏍稿績闂

Perf-database 浠跨湡鐨勬牳蹇冨亣璁炬槸: **灏嗘墍鏈夌畻瀛愮殑 device kernel 鎵ц鏃堕棿鍔犳€? 鍗冲彲杩戜技绔埌绔帹鐞嗘椂闂?*. 鏈姤鍛婇獙璇佽繖涓€鍋囪, 閲忓寲 kernel duration 鍔犳€讳笌 e2e 鐨勫樊璺濆強鍏舵潵婧? 骞跺垎鏋?compute/comm 鏄惁瀛樺湪 overlap.

---

## 涓€銆佸垎鏋愬師鐞?

### 1.1 NPU 澶?Stream 鎵ц妯″瀷

NPU 涓婃湁澶氫釜纭欢 stream 鍙互骞惰鎵ц kernel:
- **Compute stream** (Stream 2): 鎵ц璁＄畻 kernel (MatMul, Norm, Attention 绛? 鍜?DispatchFFNCombine
- **HCCL stream** (DSv3 Stream 8 / Qwen3 Stream 6): 鎵ц閫氫俊 kernel (allReduce, reduceScatter 绛?

### 1.2 kernel_details.csv 鐨勫弻閲嶈鏁伴棶棰?

`kernel_details.csv` 璁板綍浜嗘墍鏈?stream 涓婃瘡涓?kernel 鐨?`Task Duration(us)`. 瀛樺湪鍏抽敭闂:

**鍙岄噸璁℃暟**: HCCL 閫氫俊 kernel 鍦?CSV 涓璁板綍浜嗕袱娆?鈥?涓€娆′綔涓?`hcom_*` 琛?(Stream ID = NaN), 涓€娆′綔涓?`AivKernel` 琛?(鍦?HCCL stream 涓?. 鏃堕棿鎴冲拰 duration 瀹屽叏鐩稿悓. 濡傛灉涓嶅幓閲? kernel_sum/e2e 浼氳揪鍒?1.5-2.0x, 瀹规槗璇垽涓?"stream 骞惰瀵艰嚧 kernel 鎬绘椂闂?> e2e".

### 1.3 鐞嗚妯″瀷

鍋囪涓や釜 stream 涓茶浜ゆ浛鎵ц (鏃?overlap):

```
e2e = 危(compute_stream kernels) + 危(comm_stream kernels) + 危(CPU dispatch gaps)
    = t1_compute + t1_comm + t_cpu
```

濡傛灉瀛樺湪 overlap, 鍒?`t1_compute + t1_comm > e2e - t_cpu`.

---

## 浜屻€佸垎鏋愭柟娉?

1. 鎸?`Stream ID` 鍒嗙粍 kernel_details.csv, **鍘婚櫎 Stream ID = NaN 鐨勯噸澶?HCCL 琛?*
2. 鍒╃敤 `Start Time(us)` 鍜?`Duration(us)` 璁＄畻姣忎釜 stream 鐨?span: `stream_span = max(start + duration) - min(start)`
3. 瀵规瘮: `CPU gap = stream_span - 危(kernel_duration)` (璇?stream 涓?kernel 涔嬮棿鐨勭┖闂?
4. 瀵规瘮: `e2e - (t1_compute + t1_comm)` = 绾?CPU 璋冨害寮€閿€

---

## 涓夈€丒ager 妯″紡缁撴灉

### 3.1 DSv3

| 鍦烘櫙 | t1_comp (ms) | t1_comm (ms) | t1_sum (ms) | e2e (ms) | **e2e - t1_sum** | **CPU gap%** | comp_span (ms) |
|------|---------|---------|---------|---------|----------|----------|---------|
| decode_b1 | 869 | 1488 | 2357 | 2652 | **+295ms** | **11.1%** | 2650 |
| decode_b4 | 4245 | 1274 | 5519 | 5761 | **+242ms** | **4.2%** | 5759 |
| decode_b8 | 4420 | 935 | 5355 | 5425 | **+70ms** | **1.3%** | 5424 |
| decode_b16 | 3998 | 2107 | 6105 | 6173 | **+68ms** | **1.1%** | 6171 |
| decode_b32 | 3604 | 4525 | 8129 | 8183 | **+54ms** | **0.7%** | 8182 |
| prefill_256 | 2070 | 1705 | 3775 | 3822 | **+47ms** | **1.2%** | 3821 |
| prefill_1024 | 2099 | 1308 | 3407 | 3856 | **+449ms** | **11.6%** | 3855 |
| prefill_4096 | 2200 | 1041 | 3241 | 4809 | **+1568ms** | **32.6%** | 4807 |

### 3.2 Qwen3-32B

| 鍦烘櫙 | t1_comp (ms) | t1_comm (ms) | t1_sum (ms) | e2e (ms) | **e2e - t1_sum** | **CPU gap%** | comp_span (ms) |
|------|---------|---------|---------|---------|----------|----------|---------|
| decode_b1 | 29 | 1103 | 1132 | 1463 | **+331ms** | **22.6%** | 1453 |
| decode_b4 | 58 | 2029 | 2087 | 2218 | **+131ms** | **5.9%** | 2215 |
| decode_b8 | 88 | 3211 | 3299 | 3361 | **+62ms** | **1.8%** | 3358 |
| decode_b16 | 129 | 4822 | 4951 | 4981 | **+30ms** | **0.6%** | 4977 |
| decode_b32 | 210 | 7488 | 7698 | 7822 | **+124ms** | **1.6%** | 7818 |
| prefill_256 | 37 | 1064 | 1101 | 1127 | **+26ms** | **2.3%** | 1125 |
| prefill_1024 | 49 | 1056 | 1105 | 1125 | **+20ms** | **1.8%** | 1117 |
| prefill_4096 | 92 | 710 | 802 | 1244 | **+442ms** | **35.5%** | 1241 |

---

## 鍥涖€佸叧閿彂鐜?

### 4.1 鍘婚噸鍚?t1_sum < e2e, 宸€煎叏閮ㄤ负姝?

绗﹀悎涓茶鎵ц鐨勭悊璁烘ā鍨?
```
e2e = t1_compute + t1_comm + t_cpu_dispatch   (瀹岀編鎴愮珛)
```

鍘婚噸鍓嶇殑 `kernel_sum / e2e` 杈惧埌 1.5-2.0x, 鏄?HCCL 鍙岄噸璁℃暟瀵艰嚧鐨勫亣璞?

### 4.2 Compute/Comm 鍑犱箮鏃?overlap

Eager 妯″紡涓?step_trace 鐨?Overlapped 椤瑰湪鎵€鏈夊満鏅?<0.1%. 涓や釜 stream 浜ゆ浛涓茶鎵ц:
```
Compute stream: |--kernel--|  wait  |--kernel--|  wait  |--kernel--|
Comm stream:        idle   |--comm--|   idle   |--comm--|   idle
```

### 4.3 CPU 璋冨害寮€閿€: batch 瓒婂ぇ瓒婂皬

| 鍦烘櫙 | DSv3 CPU gap% | Qwen3 CPU gap% | 璇存槑 |
|------|--------|--------|------|
| decode_b1 | 11.1% | 22.6% | kernel 鏁板皯浣嗘瘡涓煭, 涓嬪彂 overhead 鏆撮湶 |
| decode_b8 | 1.3% | 1.8% | 姝ｅ父姘村钩 |
| decode_b32 | 0.7% | 1.6% | kernel 瀵嗛泦, CPU gap 鍙拷鐣?|
| prefill_4096 | 32.6% | 35.5% | 寮傚父楂?鈥?profiling 閲囬泦鑷韩寮€閿€ (tracing + disk I/O) |

prefill_4096 鐨?CPU gap 寮傚父楂?(>30%), 鍘熷洜鏄?profiling 宸ュ叿鑷韩鐨?tracing/鍐欑洏寮€閿€鍦ㄨ绠楀瘑闆嗗満鏅笅鏆撮湶. 姝ｅ父鎺ㄧ悊 (涓嶅紑 profiling) 鏃惰寮€閿€涓嶅瓨鍦?

### 4.4 Compute stream span 鈮?e2e

compute stream 鏄富鎺?stream, 瀹冪殑鏃堕棿璺ㄥ害鍑犱箮绛変簬 e2e. 杩欐剰鍛崇潃 compute stream 鍦ㄧ瓑閫氫俊瀹屾垚鏃跺浜?idle 鐘舵€?(浣撶幇涓?CPU gap 鐨勪竴閮ㄥ垎).

---

## 浜斻€乤clgraph vs Eager: 璁＄畻閫氫俊鎺╃洊瀵规瘮

### 5.1 鍒嗘瀽鐩殑

aclgraph (cudagraph) 灏嗘暣涓绠楀浘涓€娆℃€ф彁浜ょ粰 device, 鐞嗚涓婂彲浠ュ噺灏?CPU dispatch 寮€閿€, 骞跺彲鑳藉惎鐢?compute/comm pipeline 鎺╃洊.

### 5.2 鏁版嵁鏉ユ簮

- DSv3 aclgraph: `deepseekv3_torch2.9.0_vllm0.15.0_cann8.5_aclgraph_PandD/` (W8A8, TP=8/DP=2/EP, `FULL_DECODE_ONLY` cudagraph)
- Qwen3 aclgraph: `qwen3-32b_torch2.9.0_vllm0.15.0_cann8.5_aclgraph_PandD/` (BF16, TP=16, `FULL_DECODE_ONLY` cudagraph)
- 涓よ€呭潎涓?PandD 娣峰悎鍦烘櫙 (闈炲崟涓€ decode/prefill), 鍖呭惈澶氫釜 step

### 5.3 Qwen3-32B step_trace 瀹樻柟鍒嗚В

| 椤圭洰 | aclgraph | eager (decode_b8 鍙傝€? |
|------|---------|---------|
| E2E (Stage) | 3295 ms | 3361 ms |
| Computing | 635 ms (19.3%) | 89 ms (2.6%) |
| Comm(Not Overlapped) | 1872 ms (56.8%) | 3210 ms (95.5%) |
| **Overlapped** | **33.5 ms (1.0%)** | **~0 ms (<0.1%)** |
| Free | 788 ms (23.9%) | 62 ms (1.8%) |

### 5.4 DSv3 鎸?stream 璁＄畻 (鏃?step_trace)

| 椤圭洰 | aclgraph | eager (decode_b8 鍙傝€? |
|------|---------|---------|
| E2E span | 3258 ms | 5425 ms |
| t1_compute | 1149 ms | 4420 ms |
| t1_comm | 624 ms | 935 ms |
| **Overlap** | **0.3 ms (~0%)** | **~0 ms** |
| Free/CPU gap | 1487 ms (45.6%) | 70 ms (1.3%) |

> 娉? aclgraph 鏄?PandD 娣峰悎鍦烘櫙, Computing 鏃堕棿 (635ms) 姣?eager 鍗?decode_b8 (89ms) 澶у緱澶? 鏄洜涓鸿法浜嗗涓?step (鍚?prefill). Free 涓寘鍚?step 闂寸┖闂?

### 5.5 缁撹: aclgraph 妯″紡涓嬩粛鐒跺嚑涔庢病鏈夎绠楅€氫俊鎺╃洊

- Qwen3 Overlapped = 33.5 ms, 浠呭崰 Communication 鎬婚噺鐨?**1.8%**
- DSv3 Overlap = 0.3 ms, 鍙拷鐣?
- 涓よ€呬笌 eager 妯″紡涓€鏍? compute/comm 鏈川涓婃槸涓茶浜ゆ浛鎵ц

### 5.6 鍘熷洜鍒嗘瀽

vllm-ascend 鐨?aclgraph (`FULL_DECODE_ONLY` cudagraph) 鍑忓皯浜?CPU dispatch 寮€閿€, 浣嗗苟娌℃湁瀹炵幇 compute/comm 鐨?pipeline 璋冨害:

```
Layer N:  |--compute--|--allReduce--|
Layer N+1:                          |--compute--|--allReduce--|
                                     鈫?蹇呴』绛?N 鐨?allReduce 瀹屾垚
```

姣忓眰鐨?compute 瀹屾垚鍚庢墠鍚姩璇ュ眰鐨?allReduce, allReduce 瀹屾垚鍚庢墠鍚姩涓嬩竴灞傜殑 compute. 杩欑**灞傚唴涓茶**妯″紡鏃犺 eager 杩樻槸 aclgraph 閮戒竴鏍?鈥?aclgraph 鍙槸鎶婁覆琛屽簭鍒楁墦鍖呮垚鍥? 鍑忓皯 CPU 鍙備笌.

鐪熸瀹炵幇 compute/comm overlap 闇€瑕?*璺ㄥ眰 pipeline** (濡?layer N 鐨?allReduce 涓?layer N+1 鐨?compute 骞惰), 杩欓渶瑕佹鏋跺眰闈㈢殑鏄惧紡璁捐 (濡?vllm-ascend 鐨?`multistream_overlap_shared_expert`, 浣嗗湪姝ゆ profiling 涓璁句负 `false`).

---

## 鍏€佸 Perf-Database 浠跨湡鐨勬剰涔?

缁煎悎 eager 鍜?aclgraph 鐨勫垎鏋愮粨鏋?

| 缁撹 | 璇存槑 |
|------|------|
| **浠跨湡鍏紡** | `e2e 鈮?危(compute_kernels) + 危(comm_kernels) + t_cpu` 鈥?鏃犻渶寤烘ā overlap |
| **compute kernel duration 鍙俊** | 鍗?stream 涓婃棤骞惰, 姣忎釜 kernel 鐨?Task Duration 鏄湡瀹炵殑 device 鎵ц鏃堕棿 |
| **CPU gap 鍙拷鐣?* | batch鈮? 鏃?<2%, 浠跨湡鍙笉寤烘ā CPU dispatch; batch=1 鏃堕渶鍔?~10-20% 淇 |
| **閫氫俊 kernel duration 鍚瓑寰?* | comm kernel 鐨?Task Duration 鍖呭惈浜嗙瓑 compute stream ready 鐨勬椂闂? 涓嶇瓑浜庣函缃戠粶浼犺緭鏃跺欢. 浠跨湡閫氫俊搴斾娇鐢ㄥ甫瀹芥ā鍨嬭€岄潪鏌ヨ〃 |
| **prefill_4096 鐨?Free 涓嶄唬琛ㄧ湡瀹?* | 32% 鐨?CPU gap 鏄?profiling 宸ュ叿 overhead, 涓嶅簲浣滀负浠跨湡鍩哄噯 |
| **kernel_details.csv 闇€鍘婚噸** | HCCL kernel 琚褰曚袱娆?(hcom_* 琛?+ AivKernel 琛?, 浣跨敤鏃跺繀椤绘寜 Stream ID 鍘婚噸 |

---

## 闄勫綍

### A. 鏁版嵁鏂囦欢

**Eager 妯″紡** (`dsv3_qwen3_full_torch2.9.0_vllm0.15.0_cann8.5_eager/`):
姣忎釜鍦烘櫙鍖呭惈: `kernel_details.csv` (閫?kernel 璇︽儏), `op_statistic.csv` (鑱氬悎缁熻), `step_trace_time.csv` (compute/comm/free 鍒嗚В), `communication.json`, `trace_view.json`.

**aclgraph 妯″紡**:
- DSv3: `deepseekv3_torch2.9.0_vllm0.15.0_cann8.5_aclgraph_PandD/kernel_details_deepseekv3-cann85.csv` (浠?kernel_details)
- Qwen3: `qwen3-32b_torch2.9.0_vllm0.15.0_cann8.5_aclgraph_PandD/.../kernel_details.csv` + `step_trace_time.csv`

### B. 鍒嗘瀽鑴氭湰

```bash
python3.10 docs/perf_database/reports/profiling_analysis_eager/profiling_analysis.py          # CV/outlier 娣卞害鍒嗘瀽, 鍚?per-stream 鏃堕棿璁＄畻
python3.10 docs/perf_database/reports/profiling_analysis_eager/analyze_aclgraph_overlap.py   # aclgraph compute/comm overlap 鍒嗘瀽
```

### C. 鐩稿叧鎶ュ憡

- 濮婂鎶ュ憡: [绠楀瓙绋冲畾鎬т笌鎺ュ叆鍙鎬(profiling_analysis_op_stability_zh.md) 鈥?鍒嗘瀽姣忎釜绠楀瓙鍦ㄧ浉鍚?shape 涓嬬殑 CV, 璇勪及鏌ヨ〃鍙鎬?

