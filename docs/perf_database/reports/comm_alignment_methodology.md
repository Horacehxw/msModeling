# 閫氫俊绠楀瓙鑰楁椂寤烘ā鏂规硶璁?

**鐗堟湰**锛歷6.0
**鏃ユ湡**锛?026-03-19
**浣滆€?*锛欻DY
**閫傜敤鍦烘櫙**锛欻CCL 閫氫俊绠楀瓙鑰楁椂棰勬祴寤烘ā锛宐ench 瀵归綈 Comm_NO
**鏁版嵁鏉ユ簮**锛歱rofiler-qwen3-0314 + profiler-dsv3-0316 + hccl_bench_v8.5锛坅lternating/kernel/event/pipeline/profiler 浜旀ā寮忥級

> **v6.0 鏇存柊**锛氬熀浜?0318 bench 浜旀ā寮忓叏闈㈤獙璇侊紝纭珛 alternating 妯″紡涓洪粯璁ゆ帹鑽愩€?
> alternating 妯″紡娑堥櫎 1-5MB 棰勭儹鍋忛珮锛堜粠 19-63% 楂樹及闄嶈嚦 卤3%锛夛紱
> kernel 妯″紡鐢ㄤ簬 <1MB 灏忔秷鎭紙event 搴曞櫔 ~60us 娣规病鐪熷疄鍊硷級锛?
> 鏂板鍥哄畾寮€閿€淇鐢ㄤ簬 decode 灏忔秷鎭満鏅€?

## 1. E2E 鏃堕棿妯″瀷

### 1.1 TensorCast 涓茶姹傚拰

TensorCast 閫氳繃 `Runtime` 鎷︽埅鎵€鏈夌畻瀛愶紝閫愪釜鏌ヨ `PerformanceModel` 鑾峰彇鑰楁椂锛岀函涓茶姹傚拰锛?

```
T_total = 危 T_compute + 危 T_comm
```

鍏朵腑 `T_comm` 鐢?`DataSource.lookup()` 鏌?bench CSV 鑾峰緱銆?

### 1.2 瀹為檯 Stage 鏃堕棿

step_trace 璁板綍鐨勫疄闄?Stage 鏃堕棿锛?

```
Stage = Computing + Comm_NO + Free
      = T_total 脳 overhead_factor

overhead_factor = 1 + Free / (Computing + Comm_NO)
```

| 鍦烘櫙 | overhead_factor | Free 鍗犳瘮 | 鐗瑰緛 |
|------|----------------|----------|------|
| Qwen3 Prefill ISL=1024 | 1.315x | 24.0% | Dense, 鐭?ISL, 璋冨害寮€閿€澶?|
| Qwen3 Prefill ISL=4096 | 1.198x | 16.6% | Dense, 闀?ISL |
| Qwen3 Decode c1 | 1.073x | 6.8% | 鍥炬ā寮? 璋冨害寮€閿€灏?|
| DSV3 Prefill ISL=1024 | 1.029x | 2.8% | MoE, 璁＄畻瀵嗛泦 |
| DSV3 Prefill ISL=4096 | 1.039x | 3.7% | MoE, 璁＄畻瀵嗛泦 |
| DSV3 Decode c1 | 1.048x | 4.6% | MoE, 鍥炬ā寮?|

瑙勫緥锛欴ecode < Prefill锛孧oE < Dense銆俹verhead_factor 鍦?ModelRunner 灞傚簲鐢紝涓嶈繘鍏ュ崟绠楀瓙寤烘ā銆?

---

## 2. 涓夊眰 Duration 妯″瀷

閫氫俊绠楀瓙鐨勮€楁椂瀛樺湪涓変釜娴嬮噺灞傜骇锛?

| 灞傜骇 | 娴嬮噺鏂瑰紡 | 鍖呭惈鍐呭 | 鏁版嵁婧?|
|------|---------|---------|--------|
| kernel_details | NPU timeline | hcom_kernel锛圚CCL 鏁版嵁浼犺緭锛?| `kernel_details.csv` |
| operator_details | operator timeline | AicpuKernel + hcom_kernel | `operator_details.csv` c10d::* |
| bench (alternating) | host perf_counter | peer 娴佹按鎵ц锛屾秷闄ら鐑亸楂?| microbench CSV |

### 2.1 operator_details 鐨勭墿鐞嗗垎瑙?

```
operator_details (Device Total Duration) = AicpuKernel + hcom_kernel
```

| 妯″瀷 | AicpuKernel | 鍏崇郴 |
|------|------------|------|
| DSV3 (CANN 8.5) | = 0 | operator = kernel |
| Qwen3 (CANN 8.5) | > 0 (reduceScatter avg 319us, allGather avg 618us) | operator > kernel (1.4x~6.7x) |

AicpuKernel 鏄惁瀛樺湪鍙栧喅浜庢ā鍨?CANN 鐗堟湰/閫氫俊绠楀瓙瀹炵幇锛屼笉鑳藉亣璁句负 0銆?

---

## 3. 鎭掔瓑鍏崇郴楠岃瘉

### 3.1 鎭掔瓑鍏崇郴 1锛欳ommunication = 危 kernel_details hcom_*锛堝幓 AivKernel锛夆€?绮剧‘鎴愮珛

鍏ㄩ儴 18 涓満鏅紙Qwen3 脳 8 + DSV3 脳 10锛塰com/Comm = 1.0000锛屾棤涓€渚嬪銆?

AivKernel 鏉＄洰鐨?Type 鍒椾篃鏍囪涓?hcom_*锛屾湸绱犳眰鍜屼細鍙岄噸璁℃暟銆傚幓閲嶅悗绮剧‘绛変簬 Communication銆?

### 3.2 鎭掔瓑鍏崇郴 2锛欳omm_NO = 危 operator_details 鈥?浠呴儴鍒嗘垚绔?

operator_details 瀛樺湪涓夊眰宓屽锛歚c10d::_allgather_base_` 鈫?`HcclAllGatherBase` 鈫?`HcclAllGather`锛屾瘡灞傝褰曞嚑涔庣浉鍚岀殑 Device Total Duration銆傚彇鏈€搴曞眰 `Hccl*(涓嶅惈Base)` 鍘婚噸鍚庯細

| 鍦烘櫙 | Hccl*(鍘婚噸) | COMM_NO | ratio | 鍘熷洜 |
|------|------------|---------|------:|------|
| DSV3 Decode conc=8 | 652.6ms | 652.1ms | 1.001x | Overlap鈮?锛岀簿纭垚绔?|
| DSV3 Decode conc=1 | 1,799ms | 1,798ms | 2.001x | 宓屽 double counting |
| Qwen3 Prefill input4096 | 3.40s | 1.47s | 2.319x | operator Duration 鍚 compute overlap 閬洊鐨勯儴鍒?|
| Qwen3 Prefill input1024 | 17.81s | 2.69s | 6.623x | 鍚屼笂锛宱verlap 姣斾緥鏇撮珮 |
| Qwen3 Decode conc=1 | 3.5ms | 667.6ms | 0.005x | allReduce 璧?CUDAGraph 涓嶅湪 operator_details 涓?|

**鏍瑰洜**锛歰perator_details Device Total Duration 鏄瘡娆¤皟鐢ㄧ殑瀹屾暣 wall-clock Duration锛堝惈琚?overlap 閬洊鐨勯儴鍒嗭級锛岃€?COMM_NO 鏄?step_trace 绾у埆鐨勬湭琚?overlap 閫氫俊鏃堕棿銆備袱鑰呰涔変笉鍚岋紝浠呭湪 Overlap鈮? 鏃剁浉绛夈€?

### 3.3 姝ｇ‘鐨勫叧绯婚摼

```
Communication (step_trace) = 危 kernel_details hcom_* (鍘?AivKernel)  [绮剧‘锛屽叏閮?18 鍦烘櫙]
COMM_NO = Communication - Overlapped                                  [绮剧‘锛宻tep_trace 瀹氫箟]
operator_details 鈮?COMM_NO                                            [浠?Overlap鈮? 鏃剁浉绛塢
```

---

## 4. Bench 閲囬泦妯″紡鍒嗘瀽锛坴6.0 鏂板锛?

### 4.1 浜旂 bench 妯″紡姒傝堪

| 妯″紡 | 璁℃椂鏂瑰紡 | 鐗圭偣 |
|------|---------|------|
| event | NPU Event 鍖呭洿鍗曟璋冪敤 | 鍚悓姝ュ紑閿€锛屽皬娑堟伅搴曞櫔 ~60us |
| kernel | profiler 閲囬泦 hcom_kernel Duration | 绾?HCCL 浼犺緭鏃堕棿锛屾棤鍚屾寮€閿€ |
| pipeline | host perf_counter 100 娆℃棤閫愭 sync | 绋虫€佸悶鍚愶紝棰勭儹鍋忛珮 |
| profiler | profiler 閲囬泦 operator_details Duration | 鍚?AicpuKernel |
| alternating | peer 绠楀瓙浜ゆ浛娴佹按鎵ц | 娑堥櫎棰勭儹鍋忛珮锛屾帹鑽愰粯璁ゆā寮?|

### 4.2 alternating 妯″紡鏍稿績浼樺娍

alternating 妯″紡璁╃洰鏍囩畻瀛愪笌 peer 绠楀瓙浜ゆ浛鎵ц锛堝 allGather + reduceScatter 娴佹按锛夛紝娑堥櫎浜?pipeline 妯″紡涓?1-5MB 娑堟伅鐨勯鐑亸楂橀棶棰橈細

- pipeline 妯″紡鍦?1-5MB 鑼冨洿楂樹及 19-63%
- alternating 妯″紡鍦ㄧ浉鍚岃寖鍥磋宸?卤3%
- 澶ф秷鎭紙鈮?.5MB锛変袱绉嶆ā寮忚秼鍚?

### 4.3 kernel 妯″紡鐢ㄤ簬灏忔秷鎭?

瀵逛簬 <1MB 鐨勫皬娑堟伅锛宔vent 妯″紡搴曞櫔 ~60us 杩滃ぇ浜庡疄闄?kernel 鏃堕棿锛堝 allReduce nd=16 kernel 浠?~13us锛夛紝鍥犳灏忔秷鎭繀椤讳娇鐢?kernel 妯″紡鑾峰彇绾?HCCL 浼犺緭鏃堕棿銆?

### 4.4 鍥哄畾寮€閿€淇

decode 鍦烘櫙灏忔秷鎭殑 profiling P50 = bench kernel + 鍥哄畾寮€閿€锛堣皟搴?鍚屾/AicpuKernel锛夛細

| 妯″瀷 | 绠楀瓙 | nd | 鍥哄畾寮€閿€ |
|------|------|---:|--------:|
| Qwen3 | allReduce | 16 | +7.7us |
| Qwen3 | allGather | 16 | +14.6us |
| DSV3 | allGather | 8 | +1.2us |
| DSV3 | reduceScatter | 8 | +2.0us |

### 4.5 HCCL 鍗忚鍒囨崲寮傚父

DSV3 prefill allGather 768KB锛坣d=8, per_device 鈮?60KB锛夊浜?HCCL 鍗忚鍒囨崲鐐癸紝bench 鏃犳硶澶嶇幇姝よ涓猴紝闇€鐩存帴浣跨敤 profiler P50銆?

---

## 5. 浠跨湡绛栫暐鎬昏〃锛坴6.0 鏂板锛?

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

---

## 6. Bench 妯″紡閫夋嫨鎸囧崡锛坴6.0 鏂板锛?

| 绠楀瓙 | 鎺ㄨ崘妯″紡 | 鍘熷洜 |
|------|---------|------|
| allGather | alternating | peer=reduceScatter 娴佹按鎵ц锛屾秷闄ら鐑亸楂?|
| reduceScatter | alternating | peer=allGather 娴佹按鎵ц锛屾秷闄ら鐑亸楂?|
| allReduce | alternating (鑷姩 profiler fallback) | 鏃?peer锛孨PU Event 搴曞櫔 ~270us 杩滃ぇ浜?kernel 鏃堕棿 ~13us |
| all_to_all | kernel 鎴?event | 鏃?peer锛屾寜闇€閫夋嫨 |

鍏抽敭缁撹锛?

1. alternating 妯″紡娑堥櫎 1-5MB 棰勭儹鍋忛珮锛堜粠 19-63% 楂樹及闄嶈嚦 卤3%锛?
2. kernel 妯″紡鐢ㄤ簬 <1MB 灏忔秷鎭紙event 搴曞櫔 ~60us 娣规病鐪熷疄鍊硷級
3. 鍥哄畾寮€閿€淇鐢ㄤ簬 decode 灏忔秷鎭紙bench kernel + offset = profiling P50锛?
4. HCCL 鍗忚鍒囨崲鐐癸紙768KB nd=8, per_device 鈮?60KB锛夐渶鐢?profiler P50
5. allReduce 鏃?peer 绠楀瓙锛宎lternating 鑷姩 fallback 鍒?profiler 妯″紡

---

## 7. 鐜涓€鑷存€ц姹?

```bash
export HCCL_OP_EXPANSION_MODE="AIV"   # 寤鸿
export TASK_QUEUE_ENABLE=1             # 寤鸿
```

---

## 8. 瀵规瘮 Checklist锛坴6.0 鏇存柊锛?

- [ ] 鎭掔瓑鍏崇郴 1锛氶獙璇?Communication = 危 kd hcom锛堝幓 AivKernel锛夛紝搴旂簿纭?1.0000
- [ ] 鎭掔瓑鍏崇郴 2锛氭鏌?Overlap 鏄惁鈮?锛屼粎姝ゆ椂 operator_details 鈮?COMM_NO
- [ ] bench 妯″紡閫夋嫨锛歛llGather/reduceScatter 鐢?alternating锛宎llReduce 鐢?alternating (profiler fallback)
- [ ] 灏忔秷鎭紙<1MB锛夛細浣跨敤 kernel 妯″紡 + 鍥哄畾寮€閿€淇
- [ ] 澶ф秷鎭紙鈮?.5MB锛夛細鐩存帴鐢?alternating bench 鍊?
- [ ] DSV3 768KB allGather锛氫娇鐢?profiler P50锛圚CCL 鍗忚鍒囨崲寮傚父锛?
- [ ] 纭 operator_details 宓屽鍘婚噸锛堝彇 Hccl* 涓嶅惈 Base锛?
- [ ] Qwen3 Decode锛歛llReduce 鍙兘涓嶅湪 operator_details 涓紙CUDAGraph锛?
- [ ] DSV3锛氭敞鎰?MoE 閫氫俊鏂瑰樊澶э紙P10 vs P90 宸?10-100x锛?
- [ ] 璇︾粏鏁版嵁瑙?[bench_vs_profiler_comm_20260318.md](bench_vs_profiler_comm_20260318.md)


