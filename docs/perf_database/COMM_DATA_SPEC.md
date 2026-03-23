# 閫氫俊绠楀瓙鏁版嵁琛ㄦ牸寮忚鑼?(COMM_DATA_SPEC)

**鐗堟湰**锛歷1.0
**浣滆€?*锛歓H
**鏃ユ湡**锛?026-03-10
**鐘舵€?*锛氬緟 HXW review
**鍏宠仈浠诲姟**锛欱1锛圸H锛夈€丆9/C10锛圚DY锛?

---

## 1. 鐩綍缁撴瀯

閫氫俊鏁版嵁鐙珛浜庤绠楁暟鎹紝鎸?CANN 鐗堟湰瀛樺偍锛堣法 vLLM 鐗堟湰澶嶇敤锛夛細

```
tensor_cast/performance_model/perf_database/data/
鈹斺攢鈹€ {device}/
    鈹斺攢鈹€ hccl/
        鈹斺攢鈹€ {cann_version}/
            鈹溾攢鈹€ comm_config.yaml          # 鎷撴墤鎻忚堪 + 绠楀瓙鏄犲皠
            鈹溾攢鈹€ hcom_allReduce_.csv       # AllReduce 鏁版嵁
            鈹溾攢鈹€ hcom_allGather_.csv       # AllGather 鏁版嵁
            鈹溾攢鈹€ HcomReduceScatter.csv     # ReduceScatter 鏁版嵁
            鈹斺攢鈹€ hcom_alltoallv_.csv       # AllToAll 鏁版嵁
```

**绀轰緥璺緞**锛圓TLAS_800_A3锛孋ANN 8.1.RC1锛夛細
```
data/ATLAS_800_A3_752T_128G_DIE/hccl/v8.1.RC1/
```

> 娉細`op_mapping.yaml`锛堜綅浜?`vllm_ascend/{version}/`锛夐€氳繃 `communication_data_ref` 瀛楁鎸囧悜姝ょ洰褰曪細
> ```yaml
> communication_data_ref: "../../hccl/v8.1.RC1/"
> ```

---

## 2. comm_config.yaml 鏍煎紡

瀹屾暣绀轰緥瑙?`docs/perf_database/examples/comm_config_example.yaml`銆?

**蹇呭～瀛楁**锛?

```yaml
device: ATLAS_800_A3_752T_128G_DIE
cann_version: "8.1.RC1"
collection_date: "2026-03-10"

topology:
  grid_shape: [48, 8, 2]   # ATLAS_800_A3: [pod鏁? node/pod, die/node]
  tiers:
    0: {name: "inter_pod",  bandwidth_gbps: 196, latency_us: 5.5, type: "CLOS"}
    1: {name: "intra_pod",  bandwidth_gbps: 196, latency_us: 0.5, type: "CLOS"}
    2: {name: "die_level",  bandwidth_gbps: 224, latency_us: 0.2, type: "SIO"}

comm_operator_mappings:
  "tensor_cast.all_reduce.default":
    kernel_type: hcom_allReduce_
  "tensor_cast.all_gather.default":
    kernel_type: hcom_allGather_
  "tensor_cast.reduce_scatter.default":
    kernel_type: HcomReduceScatter
  "tensor_cast.all_to_all.default":
    kernel_type: hcom_alltoallv_
```

---

## 3. 閫氫俊 CSV 鏍煎紡

### 3.1 鍒楀畾涔?

| 鍒楀悕 | 绫诲瀷 | 璇存槑 |
|------|------|------|
| `message_bytes` | int | 鍗曡澶囧彂閫?鎺ユ敹鐨勬暟鎹噺锛堝瓧鑺傦級銆侫llReduce/AllGather/ReduceScatter 涓?tensor 瀛楄妭鏁帮紱AllToAll 涓哄崟璁惧鍙戦€佹€婚噺 |
| `num_devices` | int | 鍙備笌閫氫俊鐨勮澶囨暟锛? `len(rank_group)`锛?|
| `dtype` | str | 鏁版嵁绫诲瀷锛岃 搂3.2 |
| `topology_tier` | int | 鎷撴墤灞傜骇锛岃 搂3.3 |
| `Duration(us)` | float | 骞冲潎鑰楁椂锛堝井绉掞級锛寃armup 鍚?N 娆￠噸澶嶇殑鍧囧€?|
| `bandwidth_gbps` | float | 鍙€夛紝瀹炴祴甯﹀锛圙B/s锛夛紝鐢ㄤ簬楠岃瘉 |

**绀轰緥**锛坄hcom_allReduce_.csv`锛夛細
```csv
message_bytes,num_devices,dtype,topology_tier,Duration(us),bandwidth_gbps
1048576,2,DT_BF16,2,45.3,22.1
1048576,8,DT_BF16,2,125.7,7.9
1048576,16,DT_BF16,1,198.4,5.0
1048576,16,DT_BF16,0,342.1,2.9
4194304,2,DT_BF16,2,156.2,25.6
4194304,8,DT_BF16,2,421.3,9.4
```

### 3.2 dtype 鍙栧€?

| dtype 瀛楃涓?| 瀵瑰簲 torch dtype |
|-------------|----------------|
| `DT_BF16` | `torch.bfloat16` |
| `DT_FP16` | `torch.float16` |
| `INT8` | `torch.int8` |
| `DT_FP8` | `torch.float8_e4m3fn` |

**浼樺厛閲囬泦**锛歚DT_BF16`锛圦wen3 BF16 鍦烘櫙锛夈€乣INT8`锛圖SV3 W8A8 鍦烘櫙锛夈€?

### 3.3 topology_tier 鍙栧€?

ATLAS_800_A3 涓夌淮缃戞牸 `[48, 8, 2]`锛坧od 脳 node/pod 脳 die/node锛夛細

| tier | 鍚嶇О | 鍚箟 | 鍏稿瀷 num_devices |
|------|------|------|----------------|
| `2` | die_level | 鍚屼竴 node 鍐?2 涓?die 闂达紙SIO锛?24 GB/s锛?| 2 |
| `1` | intra_pod | 鍚屼竴 pod 鍐呰法 node锛圕LOS锛?96 GB/s锛?| 2, 4, 8, 16 |
| `0` | inter_pod | 璺?pod锛圕LOS锛?96 GB/s锛?.5碌s latency锛?| 2, 4, 8, 16, 32, 64 |

**tier 鍒ゅ畾瑙勫垯**锛堜笌 `_get_topology_tier()` 涓€鑷达級锛?
- rank_group 鍐呮墍鏈?rank 鍦?grid 涓殑鍧愭爣锛屾壘鍒扮涓€涓湁宸紓鐨勭淮搴?`diff_dim`
- `topology_tier = diff_dim`锛?=inter_pod, 1=intra_pod, 2=die_level锛?

---

## 4. 閲囬泦鐭╅樀

### 4.1 鎺ㄨ崘閲囬泦鑼冨洿

| 缁村害 | 鍙栧€?|
|------|------|
| message_bytes | 1KB, 4KB, 16KB, 64KB, 256KB, 1MB, 4MB, 16MB, 64MB, 256MB, 1GB锛?x 閫掑锛?|
| num_devices | 2, 4, 8, 16, 32, 64锛堟寜 topology_tier 瀹為檯鍙敤鍊硷級 |
| dtype | DT_BF16锛堝繀椤伙級銆両NT8锛圖SV3 鍦烘櫙蹇呴』锛夈€丏T_FP16锛堝彲閫夛級 |
| topology_tier | 0, 1, 2锛堝悇灞傜骇鍒嗗埆娴嬶級 |

### 4.2 鍚勭畻瀛愪紭鍏堢骇

| 绠楀瓙 | CSV 鏂囦欢 | 浼樺厛绾?| 璇存槑 |
|------|---------|--------|------|
| AllReduce | `hcom_allReduce_.csv` | P0 | Qwen3 Prefill 鍗犳瘮 89.8% |
| AllGather | `hcom_allGather_.csv` | P0 | DSV3 Decode 楂橀 |
| AllToAll | `hcom_alltoallv_.csv` | P1 | DSV3 MoE EP 鍦烘櫙 |
| ReduceScatter | `HcomReduceScatter.csv` | P1 | DSV3 Decode |

### 4.3 Qwen3-32B 瀹為檯閫氫俊鍙傛暟锛堝弬鑰冿級

TP=16锛孊F16锛宧idden_size=7168锛?
- AllReduce message_bytes = `16 * 7168 * 2 bytes` = 229,376 bytes锛圔F16 姣忓厓绱?2 瀛楄妭锛?
- 瀹為檯 token 鏁板奖鍝?message_bytes锛屽缓璁鐩?128K ~ 16MB 鑼冨洿

---

## 5. 閲囬泦鏂规硶

### 5.1 Python microbenchmark锛堜富瑕佹柟娉曪級

浣跨敤 `tools/perf_data_collection/generate_comm_microbench.py` 鐢熸垚鑴氭湰锛圚DY 浠诲姟 C9锛夈€?

鏍稿績閫昏緫锛?
```python
import torch.distributed as dist
import torch_npu

# 鍒濆鍖?HCCL backend
dist.init_process_group(backend="hccl")

# 鎺у埗 topology_tier锛氶€氳繃 rank_group 鍙傛暟
# tier=2 (die_level): rank_group=[0, 1]锛堝悓 node 鍐?2 涓?die锛?
# tier=1 (intra_pod): rank_group=[0, 1, 2, 3, 4, 5, 6, 7]锛堝悓 pod 鍐?8 涓?die锛?
# tier=0 (inter_pod): rank_group 璺?pod

group = dist.new_group(ranks=rank_group, backend="hccl")

# 璁℃椂锛歸armup 10 娆?+ 娴嬮噺 100 娆?
for _ in range(warmup):
    dist.all_reduce(tensor, group=group)
torch.npu.synchronize()

start = torch.npu.Event(enable_timing=True)
end = torch.npu.Event(enable_timing=True)
start.record()
for _ in range(repeat):
    dist.all_reduce(tensor, group=group)
end.record()
torch.npu.synchronize()
duration_us = start.elapsed_time(end) * 1000 / repeat  # ms 鈫?us
```

### 5.2 HCCL Test锛堜氦鍙夐獙璇侊級

```bash
# 浣嶄簬 CANN toolkit
mpirun -n 8 ./bin/all_reduce_test -b 8K -e 2048M -f 2 -d fp16 -o sum -p 8
mpirun -n 8 ./bin/all_gather_test -b 8K -e 512M  -f 2 -d fp16 -p 8
mpirun -n 8 ./bin/all_to_all_test  -b 8K -e 256M  -f 2 -d fp16 -p 8
```

鍙傝€冩枃妗ｏ細https://www.hiascend.com/document/detail/zh/mindstudio/70RC1/mscommandtoolug/mscommandug/auxiliarydevtool_0017.html

Python benchmark 涓?HCCL Test 鍋忓樊搴?< 10%锛圚3 楠屾敹鏍囧噯锛夈€?

---

## 6. 涓?ProfilingDataSource 鐨勬帴鍙?

`_lookup_comm()` 鏌ヨ閫昏緫锛坄profiling_data_source.py`锛夛細

1. 浠?`OpInvokeInfo.args[0]` 璁＄畻 `message_bytes`锛坱ensor 瀛楄妭鏁帮級
2. 浠?`args[-1]`锛坮ank_group锛夎绠?`num_devices = len(rank_group)`
3. 鑻?`comm_grid` 鍙敤锛岃皟鐢?`_get_topology_tier(comm_grid, rank_group)` 寰楀埌 `topology_tier`
4. 鍦?CSV 涓簿纭尮閰?`(message_bytes, num_devices, topology_tier)`
5. 鏈懡涓?鈫?fallback 鍒?`CommAnalyticModel`

**CSV 蹇呴』鍖呭惈鐨勫垪**锛歚message_bytes`, `num_devices`, `topology_tier`, `Duration(us)`
**鍙€夊垪**锛歚dtype`锛堝綋鍓嶇増鏈笉鍙備笌鍖归厤锛屼粎渚涘弬鑰冿級銆乣bandwidth_gbps`

> 娉細褰撳墠 `_lookup_comm()` 涓嶅仛 dtype 杩囨护锛岄噰闆嗘椂寤鸿姣忎釜 dtype 鍗曠嫭瀛樻枃浠舵垨鍦ㄥ悓涓€鏂囦欢涓敤 dtype 鍒楀尯鍒嗐€?

---

## 7. 楠屾敹鏍囧噯锛圕10锛?

- [ ] 4 绉嶇畻瀛愶紙AllReduce, AllGather, ReduceScatter, AllToAll锛夊悇鏈?CSV 鏂囦欢
- [ ] 姣忎釜 CSV 瑕嗙洊 3 涓?topology_tier锛?/1/2锛?
- [ ] message_bytes 瑕嗙洊 1KB ~ 1GB锛堣嚦灏?8 涓暟閲忕骇鐐癸級
- [ ] DT_BF16 鍜?INT8 涓ょ dtype 鍧囨湁鏁版嵁
- [ ] `comm_config.yaml` 瀛樺湪涓旀牸寮忔纭?
- [ ] Python benchmark 涓?HCCL Test 鍋忓樊 < 10%锛圚3锛?

