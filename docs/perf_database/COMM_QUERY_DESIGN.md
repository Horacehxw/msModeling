# 閫氫俊鏌ヨ鎺ュ彛璁捐鏂囨。

**鐗堟湰**锛歷1.0
**浣滆€?*锛歓H
**鏃ユ湡**锛?026-03-11
**鍏宠仈鏂囨。**锛歚OPERATOR_PERF_DATABASE_DESIGN_zh_v1.2.md` 搂4.2/搂4.7銆乣COMM_DATA_SPEC.md`
**鍏宠仈浠诲姟**锛欱1锛堝凡瀹屾垚锛夈€丅2锛堣繘琛屼腑锛?

---

## 1. 姒傝堪

閫氫俊鏌ヨ鏄?`ProfilingDataSource` 鍥涙潯鏌ヨ璺緞涔嬩竴锛坈ompute / comm / attention / composite锛夈€傚綋 `op_mapping.yaml` 涓煇绠楀瓙鏍囪 `category: communication` 鏃讹紝`lookup()` 鍒嗘淳鍒?`_lookup_comm()`銆?

鏌ヨ缁村害锛歚(message_bytes, num_devices, topology_tier)`锛岀簿纭尮閰嶉€氫俊 CSV銆傛湭鍛戒腑鏃?fallback 鍒?`CommAnalyticModel`銆?

---

## 2. 鎺ュ彛瀹氫箟

### 2.1 鏌ヨ鍏ュ彛

```python
# profiling_data_source.py
def _lookup_comm(
    self, op_invoke_info: "OpInvokeInfo", mapping: dict
) -> Optional[QueryResult]:
```

**杈撳叆**锛?
- `op_invoke_info`锛歊untime 鎷︽埅鐨勭畻瀛愯皟鐢ㄤ俊鎭紝鍚?`func`銆乣args`
- `mapping`锛氭潵鑷?`op_mapping.yaml` 鐨勮绠楀瓙閰嶇疆锛岃嚦灏戝惈 `kernel_type`

**杈撳嚭**锛?
- 鍛戒腑 鈫?`QueryResult(latency_us=float)`
- 鏈懡涓?鈫?`None`锛坄self.last_miss_reason` 璁板綍鍘熷洜锛?

### 2.2 OpInvokeInfo args 甯冨眬

璁捐鏂囨。 搂4.7 瀹氫箟锛屾墍鏈?TC 閫氫俊绠楀瓙 rank_group 鍧囦负鏈€鍚庝竴涓?arg锛宺ank 涓哄€掓暟绗簩锛?

| Op | args[0] | args[1] | args[2] | args[3] | args[4] |
|----|---------|---------|---------|---------|---------|
| `all_reduce` | Tensor x | int rank | List rank_group | | |
| `all_gather` | Tensor x | int dim | int rank | List rank_group | |
| `reduce_scatter` | Tensor x | int dim | int rank | List rank_group | |
| `all_to_all` | Tensor x | List out_splits | List in_splits | int rank | List rank_group |

> 瀹炵幇涓粺涓€鐢?`args[-1]` 鍙?rank_group锛宍args[-2]` 鍙?rank锛屾棤闇€鎸夌畻瀛愮被鍨嬪垎鏀€?

---

## 3. 鏌ヨ閫昏緫

### 3.1 娴佺▼

```
_lookup_comm(op_invoke_info, mapping)
  鈹?
  鈹溾攢 1. 鍙?kernel_type锛堟潵鑷?mapping锛?
  鈹?     鏈厤缃?鈫?miss: "unmapped"
  鈹?
  鈹溾攢 2. 鍔犺浇 CSV锛坃load_csv(kernel_type)锛?
  鈹?     鏂囦欢涓嶅瓨鍦?鈫?miss: "csv_not_found"
  鈹?
  鈹溾攢 3. 妫€鏌?CSV 鏍煎紡锛坮equired_cols = {message_bytes, num_devices}锛?
  鈹?     缂哄垪 鈫?miss: "csv_format_raw"锛堝師濮?profiling CSV锛岄潪 microbenchmark 鏍煎紡锛?
  鈹?
  鈹溾攢 4. 璁＄畻 message_bytes = args[0].nelement() * args[0].element_size()
  鈹?
  鈹溾攢 5. 鍙?rank_group = args[-1]锛宯um_devices = len(rank_group)
  鈹?
  鈹溾攢 6. 瑙ｆ瀽 topology_tier锛坃resolve_topology_tier(rank_group)锛?
  鈹?     comm_grid 涓?None 鈫?topology_tier = None
  鈹?
  鈹溾攢 7. 鏋勫缓鍖归厤 mask
  鈹?     鍩虹锛歮essage_bytes == x AND num_devices == x
  鈹?     鑻?topology_tier 闈?None 涓?CSV 鏈?topology_tier 鍒楋細杩藉姞 topology_tier == x
  鈹?
  鈹斺攢 8. 鏌ユ壘鍖归厤琛?
         鏃犲尮閰?鈫?miss: "shape_mismatch"
         鏈夊尮閰?鈫?QueryResult(latency_us=matched.iloc[0]["Duration(us)"])
```

### 3.2 topology_tier 瑙ｆ瀽

```python
def _resolve_topology_tier(self, group: list) -> Optional[int]:
    """浠?rank_group 閫氳繃 CommGrid 鎺ㄥ topology_tier銆?

    杩斿洖 diff_dim锛堢涓€涓湁宸紓鐨勭淮搴︾储寮曪級锛屼笌 CommAnalyticModel 闀滃儚銆?
    comm_grid 涓?None 鏃惰繑鍥?None锛堥檷绾т负涓ゅ瓧娈靛尮閰嶏級銆?
    """
```

tier 璇箟锛圓TLAS_800_A3 涓夌淮缃戞牸 [48, 8, 2]锛夛細

| tier | 鍚嶇О | 鍚箟 | 鍏稿瀷鍦烘櫙 |
|------|------|------|---------|
| 0 | inter_pod | 璺?pod | DSV3 EP all_to_all锛?2 鍗?|
| 1 | intra_pod | pod 鍐呰法 node | Qwen3 TP=16锛?6 鍗?|
| 2 | die_level | 鍚?node 鍐?| DSV3 TP=4锛? 鍗?|

### 3.3 闄嶇骇绛栫暐

| 鏉′欢 | 琛屼负 |
|------|------|
| `comm_grid` 涓?None | 涓嶈繃婊?topology_tier锛屼粎鐢?message_bytes + num_devices 鍖归厤 |
| CSV 鏃?`topology_tier` 鍒?| 鍚屼笂锛堝吋瀹规棫鏍煎紡 CSV锛?|
| topology_tier 瑙ｆ瀽澶辫触 | 鍚屼笂锛堜笉鎶涘紓甯革級 |
| 鏁翠綋鏈懡涓?| return None 鈫?EmpiricalPerformanceModel fallback 鍒?CommAnalyticModel |

---

## 4. CSV 鏍煎紡绾﹀畾

### 4.1 鍒楀畾涔?

| 鍒楀悕 | 绫诲瀷 | 蹇呴』 | 璇存槑 |
|------|------|------|------|
| `message_bytes` | int | 鏄?| 鍗曡澶囧彂閫?鎺ユ敹瀛楄妭鏁?|
| `num_devices` | int | 鏄?| 鍙備笌閫氫俊鐨勮澶囨暟 |
| `topology_tier` | int | 鎺ㄨ崘 | 鎷撴墤灞傜骇锛?/1/2锛夛紝鏁存暟绫诲瀷 |
| `Duration(us)` | float | 鏄?| 骞冲潎鑰楁椂锛堝井绉掞級 |
| `dtype` | str | 鍙€?| 褰撳墠涓嶅弬涓庡尮閰嶏紝浠呬緵鍙傝€?|
| `bandwidth_gbps` | float | 鍙€?| 瀹炴祴甯﹀锛岀敤浜庨獙璇?|

> 鍒楀悕澶у皬鍐欐晱鎰燂細`Duration(us)` 涓嶈兘鍐欐垚 `duration_us` 鎴?`Duration_us`銆?

### 4.2 鏂囦欢鍛藉悕

鏂囦欢鍚嶅繀椤讳笌 `op_mapping.yaml` 涓?`kernel_type` 瀛楁瀹屽叏涓€鑷达細

| TC Op | kernel_type锛堟枃浠跺悕锛?|
|-------|----------------------|
| `tensor_cast.all_reduce.default` | `hcom_allReduce_.csv`锛堟湯灏炬湁涓嬪垝绾匡級 |
| `tensor_cast.all_gather.default` | `hcom_allGather_.csv` |
| `tensor_cast.reduce_scatter.default` | `hcom_reduceScatter_.csv` |
| `tensor_cast.all_to_all.default` | `hcom_alltoallv_.csv` |

鏂囦欢璺緞锛歚{data_dir}/{kernel_type}.csv`锛宍data_dir` 鐢?`op_mapping.yaml` 鐨?`communication_data_ref` 瀛楁鎸囧畾銆?

### 4.3 dtype 瀛楃涓插榻愰棶棰?

C9 閲囬泦鑴氭湰杈撳嚭 `DT_BF16 / DT_FP16 / DT_FLOAT / DT_INT8`锛岃€?`DTYPE_MAP` 鏄犲皠涓猴細

| torch dtype | DTYPE_MAP 杈撳嚭 | C9 鑴氭湰杈撳嚭 | 鏄惁涓€鑷?|
|-------------|---------------|------------|---------|
| `torch.bfloat16` | `DT_BF16` | `DT_BF16` | 鉁?|
| `torch.float16` | `DT_BF16` | `DT_FP16` | 鉁?|
| `torch.int8` | `INT8` | `DT_INT8` | 鉁?|
| `torch.float32` | `FLOAT` | `DT_FLOAT` | 鉁?|

**褰撳墠褰卞搷**锛歚_lookup_comm` 涓嶆寜 dtype 杩囨护锛屼笉褰卞搷鍛戒腑銆傝嫢鍚庣画鍔?dtype 杩囨护锛岄渶鍦?`_lookup_comm` 鍐呭仛褰掍竴鍖栵紝鎴栬姹?C9 鑴氭湰涓?`DTYPE_MAP` 瀵归綈銆?

---

## 5. op_mapping.yaml 閰嶇疆

閫氫俊绠楀瓙鍦?`operator_mappings` 涓爣璁?`category: communication`锛?

```yaml
operator_mappings:
  "tensor_cast.all_reduce.default":
    kernel_type: hcom_allReduce_
    category: communication
  "tensor_cast.all_gather.default":
    kernel_type: hcom_allGather_
    category: communication
  "tensor_cast.reduce_scatter.default":
    kernel_type: hcom_reduceScatter_
    category: communication
  "tensor_cast.all_to_all.default":
    kernel_type: hcom_alltoallv_
    category: communication
```

閫氫俊鏁版嵁璺緞閫氳繃椤跺眰瀛楁鎸囧畾锛?

```yaml
communication_data_ref: "../../hccl/v8.1.RC1/"
communication_fallback: analytic
```

---

## 6. 鐩綍缁撴瀯

```
data/
鈹斺攢鈹€ ATLAS_800_A3_752T_128G_DIE/
    鈹溾攢鈹€ vllm_ascend/v0.13.0/
    鈹?  鈹斺攢鈹€ op_mapping.yaml          # communication_data_ref 鎸囧悜 hccl 鐩綍
    鈹斺攢鈹€ hccl/
        鈹斺攢鈹€ v8.1.RC1/
            鈹溾攢鈹€ comm_config.yaml     # 鎷撴墤鎻忚堪
            鈹溾攢鈹€ hcom_allReduce_.csv
            鈹溾攢鈹€ hcom_allGather_.csv
            鈹溾攢鈹€ hcom_reduceScatter_.csv
            鈹斺攢鈹€ hcom_alltoallv_.csv
```

閫氫俊鏁版嵁涓?vLLM 鐗堟湰瑙ｈ€︼紝鎸?CANN 鐗堟湰瀛樺偍锛屽彲璺?vLLM 鐗堟湰澶嶇敤銆?

---

## 7. miss_reason 鏋氫妇

| miss_reason | 鍚箟 | 澶勭悊寤鸿 |
|-------------|------|---------|
| `unmapped` | op_mapping 涓棤璇ョ畻瀛?| 琛ュ厖 op_mapping.yaml |
| `csv_not_found` | kernel_type 瀵瑰簲 CSV 涓嶅瓨鍦?| 閲囬泦鏁版嵁锛圕10锛?|
| `csv_format_raw` | CSV 鏄師濮?profiling 鏍煎紡锛岀己灏?message_bytes/num_devices 鍒?| 鐢?microbenchmark 鏍煎紡鏇挎崲 |
| `invalid_args` | args[0] 涓嶆槸 Tensor 鎴?rank_group 涓嶆槸 list/tuple | 妫€鏌?TC op 娉ㄥ唽 |
| `shape_mismatch` | CSV 涓棤鍖归厤鐨?(message_bytes, num_devices, topology_tier) 缁勫悎 | 鎵╁厖閲囬泦鐭╅樀 |

---

## 8. 寰呭姙涓庡凡鐭ラ棶棰?

| 椤圭洰 | 鐘舵€?| 璇存槑 |
|------|------|------|
| B1 `_lookup_comm` topology_tier 鍖归厤 | 宸插畬鎴?| 43 涓祴璇曢€氳繃 |
| topology_tier 绫诲瀷楠岃瘉 | 寰呯‘璁?| CSV 涓繀椤绘槸 int锛宲andas 璇诲彇榛樿 int64锛屽簲鏃犻棶棰橈紝闇€璺戞祴璇曢獙璇?|
| dtype 杩囨护 | 鏆備笉瀹炵幇 | 褰撳墠涓嶆寜 dtype 杩囨护锛涘悗缁姞鏃堕渶澶勭悊 C9 鑴氭湰涓?DTYPE_MAP 鐨勪笉涓€鑷?|
| C10 HCCL 鏁版嵁閲囬泦 | HDY 璐熻矗锛?.13 鎴 | CSV 鍒颁綅鍚?_lookup_comm 鍙湡姝ｅ懡涓?|

