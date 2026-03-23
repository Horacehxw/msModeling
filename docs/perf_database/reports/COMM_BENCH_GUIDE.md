# 閫氫俊绠楀瓙 Bench 宸ュ叿閾炬寚鍗?

**鏃ユ湡**锛?026-03-19
**浣滆€?*锛欻DY

---

## 1. 姒傝堪

閫氫俊绠楀瓙 bench 宸ュ叿閾剧敱涓変釜鑴氭湰缁勬垚锛岀敤浜庨噰闆?HCCL 閫氫俊绠楀瓙鐨勬€ц兘鏁版嵁骞剁敓鎴愭渶缁?CSV锛?

| 鑴氭湰 | 鑱岃矗 |
|------|------|
| `generate_comm_microbench.py` | 5 妯″紡閲囬泦寮曟搸锛岀敓鎴愬苟鎵ц HCCL 閫氫俊寰熀鍑?|
| `run_comm_bench.sh` | 缁熶竴閲囬泦鑴氭湰锛岀紪鎺掍袱杞噰闆嗭紙alternating + kernel锛?|
| `build_comm_csv.py` | 鍚庡鐞嗗悎骞惰剼鏈紝鍥哄畾寮€閿€淇 + profiler P50 鏇挎崲 + bandwidth 閲嶇畻 |

宸ヤ綔娴侊細`run_comm_bench.sh` 鈫?璋冪敤 `generate_comm_microbench.py` 涓よ疆閲囬泦 鈫?`build_comm_csv.py` 鍚堝苟杈撳嚭鏈€缁?CSV銆?

---

## 2. generate_comm_microbench.py 浜旂 Bench 妯″紡

### 2.1 profiler 妯″紡

- **娴嬮噺鏂瑰紡**锛歚torch_npu.profiler` 鈫?`operator_details.csv` 涓?`c10d::*` 绠楀瓙鐨?`Device Total Duration`
- **鍖呭惈鍐呭**锛欰icpuKernel + hcom_kernel锛堝榻?Comm_NO 璇箟锛?
- **鍙傛暟**锛歐ARMUP=5, ACTIVE=10锛屽彇 10 娆?Duration 鐨?median
- **鐗圭偣**锛欳ANN profiler 涓嶅彲閲嶅惎锛屾墍鏈?message size 鎵归噺鏀惧叆涓€涓?profiler session
- **Leader/Follower**锛氫粎 group_ranks[0] 寮€鍚?profiler锛屽叾浣?rank 鎵ц鐩稿悓璋冪敤浣嗕笉閲囬泦

### 2.2 kernel 妯″紡

- **娴嬮噺鏂瑰紡**锛歚torch_npu.profiler` 鈫?`kernel_details.csv` 涓?`hcom_*` 绠楀瓙鐨?Duration锛堝幓 AivKernel锛?
- **鍖呭惈鍐呭**锛氱函 HCCL kernel 鏃堕棿锛屼笉鍚?AicpuKernel 璋冨害寮€閿€
- **閫傜敤鍦烘櫙**锛氬皬娑堟伅锛?1MB锛夛紝NPU Event 搴曞櫔 ~60us 浼氭饭娌＄湡瀹炲€?
- **鍚?profiler 妯″紡鐨?batching 鍜?Leader/Follower 绛栫暐**

### 2.3 alternating 妯″紡锛堟帹鑽愶級

- **娴嬮噺鏂瑰紡**锛歱eer鈫抰arget 鏃?sync 娴佹按鎵ц锛孨PU Event 浠呮祴 target op
- **鍘熺悊**锛氭ā鎷熺敓浜?MC2 浜ゆ浛鎵ц妯″紡锛坅llGather鈫攔educeScatter 鍦ㄥ悓涓€ HCCL stream 涓婅儗闈犺儗鎵ц锛?
- **peer 鏄犲皠**锛歛llGather鈫攔educeScatter锛堝弻鍚戯級锛沘llReduce/alltoall 鏃?peer 鈫?fallback 鍒?event 妯″紡
- **鍙傛暟**锛?00 娆¤凯浠ｏ紝鍙?median
- **鏍稿績浼樺娍**锛氭秷闄?1-5MB 鍖洪棿鐨勯鐑亸楂橈紙浠?19-63% 闄嶅埌 卤3%锛?

### 2.4 event 妯″紡

- **娴嬮噺鏂瑰紡**锛氶€愭 NPU Event 璁℃椂锛?00 娆″彇 median
- **鐗圭偣**锛氬揩閫熶絾鏈?~60us 搴曞櫔锛屼笉閫傚悎灏忔秷鎭紙<1MB锛?
- **鐢ㄩ€?*锛氬揩閫?sanity check锛沘lternating 妯″紡涓?allReduce 鐨?fallback

### 2.5 pipeline 妯″紡

- **娴嬮噺鏂瑰紡**锛?00 娆℃棤閫愭 sync锛屽彇骞冲潎
- **鐗圭偣**锛氭祴閲?pipeline 绋虫€佸悶鍚愶紝涓嶅榻?Comm_NO
- **鐢ㄩ€?*锛氬悜鍚庡吋瀹癸紝纭欢閫氫俊鑳藉姏涓婄晫鍙傝€?

---

## 3. 妯″紡閫夋嫨鎸囧崡

### 3.1 鎸夌畻瀛愭帹鑽?

| 绠楀瓙 | 鎺ㄨ崘妯″紡 | 鍘熷洜 |
|------|---------|------|
| allGather | alternating | peer=reduceScatter 娴佹按鎵ц锛屾秷闄ら鐑亸楂?|
| reduceScatter | alternating | peer=allGather 娴佹按鎵ц锛屾秷闄ら鐑亸楂?|
| allReduce | alternating锛堣嚜鍔?profiler fallback锛?| 鏃?peer锛孨PU Event 搴曞櫔 ~270us 杩滃ぇ浜?kernel 鏃堕棿 ~13us |
| all_to_all | kernel 鎴?event | 鏃?peer锛屾寜闇€閫夋嫨 |

### 3.2 鎸夋秷鎭ぇ灏忔帹鑽?

| 娑堟伅澶у皬 | 鎺ㄨ崘妯″紡 | 鍘熷洜 |
|---------|---------|------|
| <1MB | kernel | NPU Event 搴曞櫔 ~60us 娣规病鐪熷疄鍊硷紱profiler kernel_details 鏃犳闂 |
| 鈮?MB | alternating | peer 娴佹按娑堥櫎棰勭儹鍋忛珮锛岃宸?卤3% |
| 768KB nd=8 | profiler P50 | HCCL 鍗忚鍒囨崲鐐瑰紓甯革紝bench 鏃犳硶澶嶇幇鐢熶骇鍊?|

### 3.3 鍏抽敭缁撹

1. **alternating 妯″紡鏄?allGather/reduceScatter 鐨勯閫?*锛氶€氳繃 peer鈫抰arget 娴佹按鎵ц妯℃嫙鐢熶骇 MC2 浜ゆ浛妯″紡锛屾秷闄?1-5MB 棰勭儹鍋忛珮
2. **灏忔秷鎭繀椤荤敤 kernel 妯″紡**锛歂PU Event 搴曞櫔 ~60us 瀵?<30us 鐨?kernel 鏃堕棿褰卞搷宸ㄥぇ
3. **allReduce 鐢?alternating 鐨?profiler fallback**锛氭棤 peer op锛岃嚜鍔ㄥ洖閫€鍒?profiler batch 閲囬泦 kernel_details
4. **鍥哄畾寮€閿€闇€鍚庡鐞嗕慨姝?*锛歬ernel 妯″紡鏁版嵁闇€鍔犲浐瀹氬紑閿€鎵嶈兘瀵归綈鐢熶骇 Comm_NO
5. **HCCL 鍗忚鍒囨崲鐐归渶鐗规畩澶勭悊**锛?68KB nd=8 allGather 鐢?profiler P50 鏇夸唬 bench 鍊?

---

## 4. run_comm_bench.sh 浣跨敤璇存槑

### 4.1 涓よ疆閲囬泦绛栫暐

**Round 1: alternating 妯″紡**
- allReduce锛氭墍鏈?message size锛堟棤 peer 鈫?event fallback锛屽彲鎺ュ彈鍣０搴曪級
- allGather/reduceScatter锛氣墺1MB锛坧eer 娴佹按娑堥櫎棰勭儹鍋忛珮锛?
- 瑕嗙洊 nd=16, 8, 4, 2

**Round 2: kernel 妯″紡**
- allGather/reduceScatter锛?1MB锛坋vent 搴曞櫔娣规病鐪熷疄鍊硷級
- 瑕嗙洊 nd=16, 8, 4, 2

### 4.2 杩愯鍛戒护

```bash
bash tools/perf_data_collection/run_comm_bench.sh ./output_dir
```

### 4.3 杈撳嚭缁撴瀯

```
output_dir/
  alternating/
    hcom_allReduce_.csv      # 鎵€鏈?msg size
    hcom_allGather_.csv      # 鈮?MB
    hcom_reduceScatter_.csv  # 鈮?MB
  kernel/
    hcom_allGather_.csv      # <1MB
    hcom_reduceScatter_.csv  # <1MB
```

### 4.4 Message Grid

- 鏍囧噯 grid锛?KB~512MB锛宲owers of 2锛?0 涓偣锛?
- 鐢熶骇 msg_bytes锛歈wen3锛坣d=16, TP=16锛夊拰 DSV3锛坣d=8, TP=8锛夌殑瀹為檯閫氫俊娑堟伅澶у皬

---

## 5. build_comm_csv.py 鍚庡鐞?

### 5.1 鏁版嵁婧愰€夋嫨瑙勫垯

| 绠楀瓙 | 娑堟伅澶у皬 | 鏁版嵁鏉ユ簮 | 鍘熷洜 |
|------|---------|---------|------|
| allReduce | 鎵€鏈?| alternating | 鏃?peer锛宔vent fallback 鍙帴鍙?|
| allGather | <1MB | kernel + 鍥哄畾寮€閿€ | event 搴曞櫔娣规病鐪熷疄鍊?|
| allGather | 鈮?MB | alternating | peer 娴佹按娑堥櫎棰勭儹鍋忛珮 |
| allGather | 768KB nd=8 | profiler P50 | HCCL 鍗忚鍒囨崲鐐瑰紓甯?|
| reduceScatter | <1MB | kernel + 鍥哄畾寮€閿€ | event 搴曞櫔娣规病鐪熷疄鍊?|
| reduceScatter | 鈮?MB | alternating | peer 娴佹按娑堥櫎棰勭儹鍋忛珮 |
| alltoallv | 鈥?| 绌猴紙浠?header锛?| 鏃?bench 鏁版嵁 |

### 5.2 鍥哄畾寮€閿€淇

kernel 妯″紡鏁版嵁闇€鍔犲浐瀹氬紑閿€浠ュ榻愮敓浜?Comm_NO锛?

```python
OVERHEAD = {
    "allReduce": {16: 7.7},        # Qwen3 TP=16
    "allGather": {16: 14.6, 8: 1.2},  # Qwen3 TP=16, DSV3 TP=8
    "reduceScatter": {16: 14.6, 8: 2.0},  # Qwen3 TP=16, DSV3 TP=8
}
```

鏉ユ簮锛歜ench vs 鐢熶骇 profiler 鐨勫浐瀹氬樊鍊硷紙涓?msg_bytes 鏃犲叧锛屼粎涓庤皟搴﹂摼璺繁搴︾浉鍏筹級銆?

### 5.3 澶勭悊娴佺▼

1. 璇诲彇 alternating CSV锛坅llReduce 鍏ㄩ噺锛宎llGather/reduceScatter 鈮?MB锛?
2. 璇诲彇 kernel CSV锛坅llGather/reduceScatter <1MB锛?
3. 瑙ｆ瀽 profiler trace 鑾峰彇 DSV3 allGather 768KB P50锛堢壒娈婂鐞嗭級
4. 鎸?(message_bytes, num_devices) 鍚堝苟锛宻ource priority 鍘婚噸
5. 瀵?kernel 鏁版嵁搴旂敤鍥哄畾寮€閿€淇
6. 閲嶇畻 bandwidth_gbps = message_bytes / (duration_us 脳 1e-6) / 1e9
7. 杈撳嚭鏈€缁?CSV

### 5.4 杈撳嚭鏍煎紡

```csv
message_bytes,num_devices,dtype,topology_tier,Duration(us),bandwidth_gbps
```

鏈€缁堣緭鍑猴細
```
output_dir/
  hcom_allReduce_.csv       # 136 琛屾暟鎹?
  hcom_allGather_.csv       # 136 琛屾暟鎹?
  hcom_reduceScatter_.csv   # 136 琛屾暟鎹?
  hcom_alltoallv_.csv       # 浠?header
```

---

## 6. 鍙傝€?

- [閫氫俊绠楀瓙瀵归綈鎶ュ憡](./comm_alignment_report_20260312.md) 鈥?璇︾粏鏁版嵁楠岃瘉
- [閫氫俊绠楀瓙寤烘ā鏂规硶璁篯(./comm_alignment_methodology.md) 鈥?鐞嗚妗嗘灦

