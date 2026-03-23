# Op Mapping 鏁欑▼ v2

> 濡備綍鍒涘缓鍜岀淮鎶?`op_mapping.yaml` 鈥斺€?TensorCast 浠跨湡涓?NPU profiling 鏁版嵁涔嬮棿鐨勬ˉ姊併€?

## 1. op_mapping.yaml 鏄粈涔堬紵

`op_mapping.yaml` 灏嗘瘡涓?TensorCast (TC) 铏氭嫙绠楀瓙鏄犲皠鍒扮湡瀹炶澶?profiling 鏁版嵁涓殑 NPU 鍐呮牳绫诲瀷锛坘ernel type锛夈€傝繖涓槧灏勪娇寰?`EmpiricalPerformanceModel` 鑳藉鏌ユ壘鐪熷疄 profiling 寤惰繜锛岃€屼笉鏄娇鐢ㄨВ鏋愪及绠椼€?

**鏂囦欢浣嶇疆锛?* `tensor_cast/performance_model/perf_database/data/{device}/vllm_ascend/{version}/op_mapping.yaml`

**鐗堟湰鍛藉悕瑙勮寖锛?* `{version}` 缂栫爜浜嗙敤浜?profiling 鐨勮蒋浠舵爤鐗堟湰锛堜緥濡?`v0.13.0` 鎴?`vllm0.13.0_torch2.8.0_cann8.3`锛?

### 鏂囦欢缁撴瀯

```yaml
version: "<vllm_ascend_version>"
device: <DEVICE>
cann_version: "<cann_version>"

interpolation_policy:
  default_method: linear
  kernel_overrides:
    FusedInferAttentionScore:
      shape_transform: sqrt    # O(seq虏) 绠楀瓙鍦?sqrt 绌洪棿涓彃鍊?

operator_mappings:
  "aten.mm.default":
    kernel_type: MatMulV2              # Profiling Type 鍒?= CSV 鏂囦欢鍚?
    notes: "[HIGH] Path A. op-plugin: ..."

  "tensor_cast.matmul_all_reduce.default":
    composite: true                     # 鍒嗚В涓哄涓唴鏍?
    sub_kernels: [MatMulV2, hcom_allReduce_]

  "aten.view.default":
    zero_cost: true                     # 绾厓鏁版嵁鎿嶄綔锛屾棤 NPU 鎵ц

torch_npu_reference:
  MatMulV2:
    apis: [torch.mm, torch.matmul]
    aclnn: [aclnnMatmul, aclnnMatmulWeightNz]
```

### 鏉＄洰绫诲瀷锛堜簰鏂ワ級

| 绫诲瀷 | 瀛楁 | 鍚箟 | 绀轰緥 |
|------|------|------|------|
| **璁＄畻** | `kernel_type: X` | 閫氳繃 Type=X 鐩存帴鏌ヨ CSV | MatMulV2, SwiGlu |
| **澶嶅悎** | `composite: true` | 鍒嗚В涓?`sub_kernels`锛屽垎鍒煡璇?| matmul_all_reduce 鈫?[MatMulV2, hcom_allReduce_] |
| **闆跺紑閿€** | `zero_cost: true` | 杩斿洖 latency=0锛堢函鍏冩暟鎹畻瀛愶級 | view, permute, split |

姣忎釜鏉＄洰蹇呴』鎭板ソ鍏锋湁浠ヤ笂涓夌绫诲瀷涔嬩竴銆?

### 鍙€夊瓧娈?

- `alternate_kernel_types: [Type1, Type2]` 鈥斺€?涓荤被鍨嬫湭鍛戒腑鏃剁殑澶囬€?CSV 绫诲瀷
- `category: communication` 鈥斺€?瑙﹀彂 message_bytes+num_devices 鏌ヨ鑰岄潪 shape 鍖归厤
- `query_mode: attention_special` 鈥斺€?瑙﹀彂 (batch, seq, heads, head_dim) 鍖归厤
- `query_mode: elementwise` 鈥斺€?瑙﹀彂杈撳嚭褰㈢姸鍖归厤 + dtype 鏉惧紱缂╂斁锛堥€愬厓绱犵畻瀛愶級
- `notes: "..."` 鈥斺€?鍖呭惈缃俊搴︾骇鍒殑璇佹嵁閾?

#### 閫愬厓绱犵畻瀛?(Elementwise Ops)

瀵逛簬鍐呭瓨甯﹀鍙楅檺鐨勯€愬厓绱犵畻瀛?(`aten.add.Tensor`, `aten.mul.Tensor`, `aten.div.Tensor`),
浣跨敤 `query_mode: elementwise` 浠ｆ浛榛樿鐨勮緭鍏ュ舰鐘跺尮閰?

```yaml
"aten.add.Tensor":
  kernel_type: Add
  query_mode: elementwise
```

姝ゆā寮忔寜**杈撳嚭褰㈢姸**鍖归厤 CSV,骞舵敮鎸?dtype 鏉惧紱鍖归厤 (FP32 鈫?BF16 脳 2.0 瀛楄妭姣旂缉鏀?銆?
涓嶉渶瑕佽缃?`tc_input_count`銆?

## 2. 鏁版嵁娴侊細PyTorch 鈫?NPU 鍐呮牳

鐞嗚В姝ゆ祦姘寸嚎鏄垱寤烘纭槧灏勭殑鍩虹锛?

```
PyTorch aten 绠楀瓙 (濡?aten.mm)
  鈫?op-plugin 鍒嗗彂 (op_plugin_functions.yaml)
    鈫?C++ 瀹炵幇 (opapi/*.cpp)
      鈫?EXEC_NPU_CMD(aclnn*) 璋冪敤
        鈫?CANN aclnn Host API
          鈫?L0 OpType 娉ㄥ唽 (CMakeLists.txt)
            鈫?NPU 鍐呮牳鎵ц
              鈫?Profiling kernel_details.csv
```

**鍏抽敭鏍囪瘑绗︼細** `kernel_details.csv` 涓殑 `Type` 鍒?= CANN OPTYPE = op_mapping.yaml 涓殑 `kernel_type`銆?

**Name 鍒楋細** 鍏锋湁3娈电粨鏋勶細`aclnnAPI_DispatchFunc_L0OpType`銆傜3娈?= Type 鍒楃殑鍊笺€?

## 3. 涓夌鏄犲皠璺緞

### 璺緞 A锛歛ten 鈫?op-plugin 鈫?aclnn锛堟渶甯歌锛?

閫傜敤浜庨€氳繃 Ascend op-plugin 鍒嗗彂鐨勬爣鍑?PyTorch 绠楀瓙锛?

1. **鍦?op-plugin 涓煡鎵剧畻瀛愶細** `grep "aten::mm" op_plugin/config/op_plugin_functions.yaml`
2. **鎵惧埌 C++ 瀹炵幇锛?* `op_plugin/ops/opapi/MmKernelNpuOpApi.cpp`
3. **鎵惧埌 aclnn 璋冪敤锛?* `EXEC_NPU_CMD(aclnnMm, ...)` 鎴?`EXEC_NPU_CMD(aclnnMatmulWeightNz, ...)`
4. **鎵惧埌 OPTYPE锛?* 鍦?CANN 浠撳簱涓悳绱?aclnn 鍚嶇О 鈫?鎵惧埌 `OP_TYPE_REGISTER(MatMulV2)`
5. **瀵圭収 profiling 楠岃瘉锛?* 妫€鏌?`MatMulV2` 鏄惁鍑虹幇鍦?kernel_details.csv 鐨?Type 鍒椾腑

**绀轰緥锛歛ten.mm.default 鈫?MatMulV2**
```
op-plugin YAML 鈫?MmKernelNpuOpApi.cpp 鈫?EXEC_NPU_CMD(aclnnMm)
  鈫?cann-ops-nn/matmul/mat_mul_v2/ 鈫?OP_TYPE_REGISTER(MatMulV2)
  鈫?Profiling: Type=MatMulV2
```

### 璺緞 B锛歵orch_npu.npu_* 鈫?op-plugin 鈫?aclnn

閫傜敤浜庝娇鐢?torch_npu API 鐨?vLLM-ascend 涓撶敤绠楀瓙锛?

1. **鎵惧埌 vllm-ascend 璋冪敤锛?* `torch_npu.npu_grouped_matmul_swiglu_quant(...)`
2. **鍦?op-plugin 涓煡鎵撅細** `grep "npu_grouped_matmul_swiglu_quant" op_plugin/`
3. **娌跨浉鍚岀殑 aclnn 鈫?OPTYPE 閾捐矾杩借釜**

**绀轰緥锛歡rouped_matmul_quant_swiglu 鈫?GroupedMatmulSwigluQuant**
```
vllm-ascend moe_mlp.py 鈫?torch_npu.npu_grouped_matmul_swiglu_quant
  鈫?op-plugin 鈫?aclnnGroupedMatmulSwigluQuantWeightNZ
  鈫?cann-ops-transformer/gmm/ 鈫?OP_TYPE_REGISTER(GroupedMatmulSwigluQuant)
  鈫?Profiling: Type=GroupedMatmulSwigluQuant
```

### 璺緞 C锛歷LLM-ascend 鑷畾涔夌畻瀛?/ Triton 鍐呮牳

閫傜敤浜庝笉鍦?op-plugin 涓殑绠楀瓙锛堣嚜瀹氫箟鍐呮牳銆乀riton銆丄TB锛夛細

1. **鎵惧埌 vllm-ascend 鑷畾涔夌畻瀛愶細** 濡?`vllm_ascend/ops/attention.py`
2. **鍒ゆ柇鏄?Triton銆乧src 杩樻槸 ATB锛?* 鍑芥暟鍚嶉€氬父 = profiling Type
3. **鍦?profiling 鏁版嵁涓獙璇?*

**绀轰緥锛欰TB 鍐呮牳**
```
vllm-ascend mla_v1.py 鈫?torch_npu.atb.npu_ring_mla()
  鈫?ATB 鍐呮牳: RINGMLAPrefillBF16Kernel
  鈫?Profiling: Type=RINGMLAPrefillBF16Kernel
```

### 閫氫俊绠楀瓙 (HCCL)

閫氫俊绠楀瓙瀹屽叏缁曡繃 op-plugin锛?
```
TC all_reduce 鈫?torch.distributed.all_reduce 鈫?HCCL 鈫?hcom_allReduce_
```
杩欎簺绠楀瓙浣跨敤 `message_bytes + num_devices` 杩涜鏌ヨ锛岃€岄潪 shape 鍖归厤銆?

## 4. 濡備綍杩借釜鍗曚釜绠楀瓙锛堝垎姝ユ寚鍗楋級

**鐩爣锛?* 灏?`tensor_cast.swiglu.default` 鏄犲皠鍒板叾 NPU 鍐呮牳绫诲瀷銆?

**姝ラ 1锛氱悊瑙?TC 绠楀瓙**
```bash
grep -r "def swiglu" tensor_cast/ops/
# 鈫?tensor_cast/ops/activation.py: SwiGlu 婵€娲诲嚱鏁?(gate * sigmoid(gate) * up)
```

**姝ラ 2锛氭壘鍒?aten/torch_npu 璺緞**
SwiGlu 鏄竴涓嚜瀹氫箟 TC 绠楀瓙锛屽洜姝ゆ鏌?vLLM-ascend锛?
```bash
grep -r "swiglu\|silu_and_mul" /path/to/vllm-ascend/
# 鈫?vllm_ascend/ops/activation.py 鈫?torch_npu.npu_swiglu(...)
```

**姝ラ 3锛氭煡鎵?op-plugin 鏉＄洰**
```bash
grep "npu_swiglu" /path/to/op-plugin/op_plugin/config/op_plugin_functions.yaml
# 鈫?绗?5742 琛? npu_swiglu
```

**姝ラ 4锛氭煡鎵?EXEC_NPU_CMD**
```bash
grep -r "npu_swiglu" /path/to/op-plugin/op_plugin/ops/
# 鈫?SwigluKernelNpuOpApi.cpp: EXEC_NPU_CMD(aclnnSwiglu, ...)
```

**姝ラ 5锛氭煡鎵?OPTYPE**
```bash
grep -r "SwiGlu\|SWIGLU" /path/to/cann-ops-transformer/ --include="CMakeLists.txt"
# 鈫?set(OPTYPE "SwiGlu")
```

**姝ラ 6锛氬湪 profiling 涓獙璇?*
```bash
grep "SwiGlu" kernel_details.csv | head -3
# 鈫?Type=SwiGlu锛孌Sv3 涓?390 娆★紝Qwen3 涓?670 娆?
```

**缁撴灉锛?*
```yaml
"tensor_cast.swiglu.default":
  kernel_type: SwiGlu
  notes: "[HIGH] Path B. op-plugin: SwigluKernelNpuOpApi.cpp 鈫?aclnnSwiglu 鈫?SwiGlu."
```

## 5. 8 绉?Shape 宸紓

TC tensor shape 涓?NPU profiling shape 瀛樺湪宸紓銆俙profiling_data_source.py` 鑷姩澶勭悊杩欎簺宸紓锛屼絾璋冭瘯鏃堕渶瑕佺悊瑙ｅ畠浠細

| # | 宸紓绫诲瀷 | TC Shape | NPU Profiling Shape | 澶勭悊鏂瑰紡 |
|---|---------|----------|---------------------|----------|
| 1 | 鎵规缁村害 | `(1,S,D)` | `(S,D)` | 绉婚櫎涓よ竟鐨勫墠瀵?batch=1 |
| 2 | 搴忓垪濉厖 | `S=144` | `S=136` | block-padding 瀹瑰樊锛堝悜涓婂彇鏁村埌 16/32锛?|
| 3 | FRACTAL_NZ | `(K,N)` ND 鏍煎紡 | `[H,W,bh,bw]` 鍒嗗潡鏍煎紡 | `fractal_nz_to_nd()` 杩樺師 |
| 4 | ND 杞疆 | `(K,N)` | `(N,K)` | MatMul 鏉冮噸杞疆妫€鏌?|
| 5 | SwiGlu 鎷兼帴 | 2脳`(S,D/2)` | 1脳`(S,D)` | 鍦ㄦ渶鍚庣淮搴︿笂鎷兼帴杈撳叆 |
| 6 | RoPE 甯冨眬 | `(B,H,S,D)` Q,K | `(B,S,H,D)` K,Q | 杞疆缁村害 + 閲嶆帓杈撳叆 |
| 7 | RoPE 鍐呮牳 | 鍗曚釜 TC 绠楀瓙 | 澶氫釜 NPU 鍐呮牳 | `alternate_kernel_types` |
| 8 | 澶嶅悎绠楀瓙 | 铻嶅悎鐨?TC 绠楀瓙 | 鍒嗙鐨?NPU 鍐呮牳 | `sub_kernels` 鍒嗚В |

## 6. 浣跨敤 Profiling 鏁版嵁

### kernel_details.csv 鍒楄鏄?

| 鍒楀悕 | 鍚箟 | 鐢ㄩ€?|
|------|------|------|
| **Type** | CANN OPTYPE = 鎴戜滑鐨?`kernel_type` | 鑱氬悎鐨勪富閿?|
| **Name** | `aclnn_Dispatch_L0OpType` 涓夋寮?| 杩芥函鍒?aclnn API |
| **Input Shapes** | tensor shape 瀛楃涓?| CSV 涓殑 shape 鍖归厤 |
| **Duration(us)** | 鍐呮牳鎵ц鏃堕棿 | 鎬ц兘鏁版嵁 |
| **Accelerator Core** | AI Core 鎴?AI Vector Core | 纭欢鍒╃敤鐜?|

### 瑙ｆ瀽 Profiling 鏁版嵁

```bash
# 浠?kernel_details.csv 鐢熸垚鎸夊唴鏍告媶鍒嗙殑 CSV
python3.10 -m tools.perf_data_collection.parse_kernel_details \
  --device ATLAS_800_A3_752T_128G_DIE \
  --vllm-ascend-version <version_string> \
  --kernel-details-path /path/to/kernel_details.csv

# 楠岃瘉鐢熸垚鐨勬暟鎹簱
python3.10 -m tools.perf_data_collection.validate \
  --database tensor_cast/performance_model/perf_database/data/{device}/vllm_ascend/{version}/
```

### 鑾峰彇鍞竴鍐呮牳绫诲瀷

```python
import csv
from collections import Counter
with open('kernel_details.csv') as f:
    types = Counter(row['Type'] for row in csv.DictReader(f))
for t, c in types.most_common():
    print(f"{c:6d}  {t}")
```

## 7. 鏌ヨ鍒嗗彂绫诲埆

`ProfilingDataSource` 鏍规嵁 op_mapping 閰嶇疆閫氳繃 5 鏉¤矾寰勮矾鐢辨煡璇細

```
鏄鍚堢畻瀛愶紵 鈫?_lookup_composite(sub_kernels)
鏄€氫俊绠楀瓙锛?鈫?_lookup_comm(message_bytes, num_devices)
鏄壒娈婃敞鎰忓姏锛?鈫?_lookup_attention(batch, seq, heads, head_dim)
鏄浂寮€閿€锛?鈫?QueryResult(latency=0)
榛樿 鈫?_lookup_compute(kernel_type, alternate_kernel_types)
```

| 绫诲埆 | 鏌ヨ鏂瑰紡 | 鍖归厤渚濇嵁 |
|------|---------|---------|
| `compute` | CSV shape 鏌ユ壘 | 杈撳叆/杈撳嚭 tensor shape |
| `communication` | 娑堟伅瀛楄妭鏁?| `tensor_nbytes * dtype_size` |
| `attention_special` | 娉ㄦ剰鍔涚淮搴?| `(batch, seq_len, num_heads, head_dim)` |
| `composite` | 鍒嗚В + 姹傚拰 | 姣忎釜 sub_kernel 鐙珛鏌ヨ |
| `zero_cost` | 杩斿洖 0 | 鏃犻渶鏌ユ壘 |

## 8. 澶勭悊 CANN 鐗堟湰宸紓

鍐呮牳绫诲瀷浼氬湪涓嶅悓 CANN 鐗堟湰涔嬮棿鍙戠敓鍙樺寲銆傚父瑙佹ā寮忥細

| 鍙樻洿绫诲瀷 | 绀轰緥 | 澶勭悊鏂瑰紡 |
|---------|------|---------|
| **閲嶅懡鍚?* | `ScatterElements` 鈫?`ScatterElementsV2` | 鏇存柊 `kernel_type`锛岀敤 `alternate_kernel_types` 鍏煎 |
| **铻嶅悎** | 鐙珛鐨?matmul+activation 鈫?鍗曚釜铻嶅悎鍐呮牳 | 鏇存柊 `kernel_type`锛堜笉浠呬粎鏄?`alternate_kernel_types`锛?|
| **鎷嗗垎** | 涓€涓唴鏍?鈫?涓や釜鐙珛鍐呮牳 | 鍙兘闇€瑕?`composite: true` + `sub_kernels` |
| **绉婚櫎** | Triton 鍐呮牳琚?CANN 鍘熺敓铻嶅悎鏇夸唬 | 鍒犻櫎鏉＄洰鎴栨洿鏂颁负鏂板唴鏍哥被鍨?|
| **鏂板鍐呮牳** | 鏂扮殑 ATB/CANN 铻嶅悎鍐呮牳 | 娣诲姞鏂版潯鐩紝閫氳繃 5 灞傛祦姘寸嚎杩借釜 |

### 濡備綍鍙戠幇鐗堟湰宸紓

1. **瀵规瘮涓や釜 CANN 鐗堟湰鐨?profiling type锛?*
   ```bash
   # 浠庢瘡娆?profiling 鎻愬彇鍞竴绫诲瀷
   awk -F',' 'NR>1 {print $2}' old_kernel_details.csv | sort -u > old_types.txt
   awk -F',' 'NR>1 {print $2}' new_kernel_details.csv | sort -u > new_types.txt
   diff old_types.txt new_types.txt
   ```
2. **閫氳繃 5 灞傛祦姘寸嚎锛堣矾寰?A/B/C锛夎拷韪瘡涓樊寮?*锛岀‘瀹氭纭槧灏?
3. **浣跨敤 `alternate_kernel_types`**锛氬綋鏂版棫鍚嶇О鍙兘鍑虹幇鍦ㄤ笉鍚?profiling 鏁版嵁闆嗕腑鏃?

**鏍稿績缁忛獙锛?* 鏇存崲 CANN 鐗堟湰鏃跺姟蹇呴噸鏂扮敓鎴?op_mapping銆俻rofiling 鐨?`Type` 鍒楁槸鍞竴鐪熺浉銆?

### aclgraph 涓€鑷存€?

vllm-ascend 鐨?aclgraph 纭繚 **eager 妯″紡鍜?graph 妯″紡浜х敓瀹屽叏鐩稿悓鐨勭畻瀛?*锛堝寘鎷瀺鍚?pass锛夈€備袱绉嶆ā寮忕殑 profiling 鏁版嵁瀵?op_mapping 鍚屾牱鏈夋晥銆?

## 9. 绔埌绔獙璇?

楠岃瘉瑕佹眰**浠?profiling 鏁版嵁鏈韩鎺ㄥ**姝ｇ‘鐨?TC 浠跨湡鍙傛暟銆備娇鐢ㄩ敊璇弬鏁帮紙濡?profiling 鎹曡幏鐨勬槸 decode 浣嗙敤浜?prefill 鍙傛暟锛変細瀵艰嚧澶ч噺 shape 涓嶅尮閰嶏紝杩?*涓嶆槸** op_mapping 鐨勯棶棰樸€?

### 姝ラ 1锛氬垎鏋?profiling 鏁版嵁鎺ㄥ鍙傛暟

**鍒ゆ柇璐熻浇绫诲瀷锛坧refill vs decode锛夛細**
```bash
# 妫€鏌ヨ绠楀唴鏍哥殑鎵规缁村害 鈥斺€?灏忓€?(1-50) = decode锛屽ぇ鍊?(100+) = prefill
for f in MatMulV2.csv AddRmsNorm.csv SwiGlu.csv; do
  echo "=== $f ===" && awk -F',' 'NR>1 {print $3}' $DATA_DIR/$f | sort | uniq -c | sort -rn | head -5
done
```

**鍒ゆ柇閲忓寲鏂瑰紡锛?*
```bash
# QuantBatchMatmulV3.csv 瀛樺湪涓斿惈 INT8 鈫?W8A8_STATIC锛涗粎 BF16 MatMulV2 鈫?DISABLED
ls $DATA_DIR/*.csv | grep -i quant
```

**鍒ゆ柇骞惰搴?(TP/DP/EP)锛?*
- 姣旇緝 CSV 涓棿缁村害涓庢ā鍨嬮厤缃細`intermediate_per_card = model.intermediate_size / TP`
- 妫€鏌?FIA 澶存暟锛歚q_heads_per_card = model.num_attention_heads / TP`
- 瀛樺湪 MoE 绠楀瓙 (GroupedMatmul*) 鈫?鍚敤 EP
- 鎺ㄥ锛歚world_size = TP 脳 DP 脳 EP_size`

**鍒ゆ柇鎵规澶у皬锛?*
- 璁＄畻鍐呮牳涓渶楂橀鐨勬壒娆＄淮搴?= 鐩爣 `--num-queries`
- Decode锛歚--num-queries=<batch> --query-length=1 --context-length=4500`
- Prefill锛歚--num-queries=1 --query-length=<batch>`锛堟垨 nq=2 ql=batch/2锛?
- block-padding 鍖归厤锛歍C 鍚戜笂濉厖鍒?16 鐨勫€嶆暟锛屽洜姝?`ceil(nq*ql/16)*16` 蹇呴』涓?CSV seq 缁村害鍖归厤

### 姝ラ 2锛氫娇鐢ㄦ帹瀵肩殑鍙傛暟杩愯 TC 浠跨湡

```bash
python3.10 -m tensor_cast.scripts.text_generate $MODEL \
  --num-queries $NQ --query-length $QL [--context-length $CL] \
  --device $DEVICE --world-size $WS --tp-size $TP [--dp-size $DP] [--ep-size $EP] \
  --quantize-linear-action $QUANT \
  --performance-model profiling --compile \
  --perf-database $DATA_DIR
```

濡傛灉浣犲湪楠岃瘉 FlashCommV1 瀵归綈锛屽彲棰濆娣诲姞 `--enable-flashcomm-v1`銆?
璇ュ紑鍏冲彧鍦?`--compile` 鎵撳紑鏃剁敓鏁堬紱鍚屾椂瀹冧笌 `matmul_allreduce` 杩欑被
MC2 铻嶅悎璺緞浼氱珵浜夊悓涓€閮ㄥ垎閫氫俊瀛愬浘锛屽洜姝ら€氬父搴斾綔涓哄崟鐙厤缃樉寮忓紑鍚紝
涓嶈榛樿涓庡叾浠?compile pass 涓€璧锋贩鐢ㄣ€傚綋鍓嶈寮€鍏充粎鐢ㄤ簬 prefill 瀵归綈锛?
鍘熷 decode profiling 涓嶅惎鐢?FlashCommV1锛屽洜姝?decode 鍦烘櫙涓嬪簲淇濇寔鍏抽棴銆?

### 姝ラ 3锛氬姣忎釜 MISS 杩涜鍒嗙被

杈撳嚭浼氭樉绀?`EmpiricalPerformanceModel: X/Y ops matched`銆傚姣忎釜 MISS 杩涜鍒嗙被锛?

| 宸窛绫诲埆 | 绀轰緥 | 澶勭悊鏂瑰紡 |
|---------|------|---------|
| **绠楀瓙鏄犲皠閿欒** | kernel_type 閿欒鎴栫己灏戞潯鐩?| 淇 op_mapping.yaml |
| **Shape 鏁版嵁缂哄彛** | 鍐呮牳姝ｇ‘浣?CSV 涓病鏈夊搴?shape | 琛ュ厖 profiling 鏁版嵁鎴栧井鍩哄噯娴嬭瘯 |
| **TC 鍒嗚В涓嶅尮閰?* | TC 涓棿 shape 鈮?鐪熷疄 vLLM-ascend | 宸茬煡闄愬埗锛岄潪鏄犲皠闂 |
| **缁撴瀯鎬х己澶?* | Embedding銆並V cache銆侀€氫俊绠楀瓙 | 棰勬湡涔嬩腑 鈥斺€?TC 涓?NPU 鎺ュ彛涓嶅悓 |
| **鍙傛暟涓嶅尮閰?* | 閿欒鐨?batch/TP 瀵艰嚧缂哄け | 浠庢楠?1 閲嶆柊鎺ㄥ鍙傛暟 |

**鏍稿績鍘熷垯锛?* shape MISS + 姝ｇ‘鐨?kernel_type = 鏁版嵁瑕嗙洊缂哄彛銆俿hape MISS + 閿欒鐨?kernel_type = 绠楀瓙鏄犲皠閿欒銆傚彧鏈夊悗鑰呴渶瑕佷慨澶嶃€?

### 姝ラ 4锛氳凯浠?

1. 淇鎵€鏈夌畻瀛愭槧灏勯敊璇?
2. 濡傚彂鐜板弬鏁颁笉鍖归厤锛岀敤淇鍚庣殑鍙傛暟閲嶆柊杩愯
3. 閲嶅鐩村埌鏃犳柊鐨勭畻瀛愭槧灏勯敊璇?
4. 鎸夌被鍒褰曞墿浣欏樊璺?

### 姝ラ 5锛氳繍琛岃嚜鍔ㄥ寲娴嬭瘯

```bash
python3.10 -m pytest tests/perf_database/test_reference_data_e2e.py -v
```

### 鍚勬ā鍨嬬被鍨嬮鏈熺粨鏋?

| 妯″瀷绫诲瀷 | 鍏稿瀷鍖归厤鐜?| 璇存槑 |
|---------|----------|------|
| Dense BF16锛堝 Qwen3-32B锛?| 80-90% | 鍓╀綑缂哄け锛歛ttention銆並V cache銆乪mbedding銆侀€氫俊 |
| Dense W8A8 | 70-85% | 閲忓寲绠楀瓙鍙兘鏈変笉鍚岀殑涓棿 shape |
| MoE W8A8锛堝 DSv3锛?| 35-50% | MoE 璺敱浜х敓鍙彉鎵规澶у皬锛汳LA 鍒嗚В澶嶆潅 |

MoE 妯″瀷鍖归厤鐜囪緝浣庢槸棰勬湡鐨勶紝鍥犱负 TC 鐨?compile pass 浜х敓鐨勪腑闂?shape 涓庣湡瀹?vLLM-ascend 涓嶅悓锛堝挨鍏舵槸 MLA 鎶曞奖鍜?MoE 鍒嗗彂閮ㄥ垎锛夈€?

## 10. 甯歌闄烽槺

1. **缂哄皯 `--compile`**锛氫笉鍔犳鍙傛暟鏃讹紝铻嶅悎绠楀瓙锛圫wiGlu銆丄ddRmsNorm銆丮C2銆丗lashCommV1锛変細鍒嗚В涓?70+ 涓?aten 鍘熷绠楀瓙锛屾棤娉曞尮閰?profiling 鍐呮牳銆俻rofiling 妯″紡涓嬪姟蹇呬娇鐢?`--compile`銆?

2. **娣风敤 `--enable-flashcomm-v1` 涓?MC2 閰嶇疆**锛欶lashCommV1 涓?`matmul_allreduce` 浼氭敼鍐欓儴鍒嗛噸鍙犵殑閫氫俊妯″紡锛岄€氬父搴旇涓轰簩閫変竴鐨?compile 閰嶇疆銆傚仛 profiling 瀵归綈鏃讹紝鍏堟槑纭綋鍓嶈楠岃瘉鍝潯璺緞锛屽啀鍐冲畾鏄惁娣诲姞 `--enable-flashcomm-v1`銆傚彟澶栵紝褰撳墠 FlashCommV1 鍙敤浜?prefill锛屽榻?decode profiling 鏃朵笉瑕佸紑鍚€?

3. **楠岃瘉鍙傛暟閿欒**锛氱敤 prefill 鍙傛暟锛坄--query-length 3500`锛夊幓楠岃瘉 decode 鐨?profiling 鏁版嵁锛坄batch=4, query-length=1`锛変細瀵艰嚧澶ч噺 shape 涓嶅尮閰嶃€傚姟蹇呭厛浠?CSV shape 鎺ㄥ鍙傛暟锛堣绗?9 鑺傛楠?1锛夈€?

4. **閲嶅懡鍚嶅唴鏍哥被鍨嬮敊璇?*锛欳ANN 鐗堟湰鍙兘閲嶅懡鍚嶅唴鏍搞€傚姟蹇呮牳瀹?profiling 鏁版嵁涓殑 `Type` 鍒楋紝骞剁敤 `alternate_kernel_types` 瀹炵幇璺ㄧ増鏈吋瀹广€?

5. **娣锋穯 Name 鍜?Type 鍒?*锛歚Type` 鍒楁槸骞插噣鐨?OPTYPE锛堟垜浠殑鏌ヨ閿級锛宍Name` 鍒楁槸瀹屾暣鐨勫眰绾ц矾寰勩€傚缁堟寜 Type 鑱氬悎銆?

6. **澶嶅悎 vs 鍗曚竴**锛氭煇浜?TC 绠楀瓙鏄犲皠鍒板涓?NPU 鍐呮牳锛圡LA decode = BatchMatMulV2 + FIA + batch_matmul_transpose锛夈€備娇鐢?`composite: true` + `sub_kernels`銆?

7. **Shape 涓嶅尮閰?鈮?鏄犲皠閿欒**锛歴hape MISS锛圕SV 涓棤鍖归厤 shape锛変笌鏄犲皠閿欒锛坘ernel_type 閿欒锛変笉鍚屻€俿hape 缂哄け鏄暟鎹鐩栫己鍙ｏ紝涓嶆槸 op_mapping 鐨勯棶棰樸€?

8. **MoE 妯″瀷浣庡尮閰嶇巼**锛歁oE 妯″瀷锛圖Sv3锛夊ぉ鐒跺尮閰嶇巼杈冧綆锛?5-50%锛夛紝鍥犱负 TC 鐨?MoE 鍒嗗彂鍜?MLA 鍒嗚В浜х敓鐨勪腑闂?shape 涓庣湡瀹?vLLM-ascend 涓嶅悓銆傝繖鏄凡鐭ョ殑 TC 浠跨湡闄愬埗锛屼笉鏄畻瀛愭槧灏勯敊璇€?

9. **閫氫俊绠楀瓙涓嶄娇鐢?shape**锛欻CCL 绠楀瓙锛坅llreduce銆乤llgather 绛夛級浣跨敤 message_bytes锛屼笉浣跨敤 tensor shape銆備笉瑕佸皾璇?shape 鍖归厤銆?

## 11. 蹇€熷弬鑰冿細甯哥敤鏄犲皠

### 鏍囧噯 aten 绠楀瓙
| TC 绠楀瓙 | NPU 鍐呮牳 | 璇存槑 |
|---------|---------|------|
| aten.mm | MatMulV2 | 鏍囧噯鐭╅樀涔樻硶 |
| aten.bmm | BatchMatMulV2 | 鎵规鐭╅樀涔樻硶锛堝閫夛細batch_matmul_transpose锛?|
| aten.addmm | MatMulV2 | Bias 铻嶅悎鍒?MatMulV2 |
| aten.add.Tensor | Add | 閫愬厓绱犲姞娉?|
| aten.mul.Tensor | Mul | 閫愬厓绱犱箻娉?|
| aten.div.Tensor | Div | 闄ゆ硶锛堝閫夛細RealDiv锛?|
| aten.embedding | GatherV2 | Embedding 鏌ユ壘锛堝閫夛細GatherV3锛?|
| aten.to.dtype | Cast | 绫诲瀷杞崲锛堝閫夛細TensorMove锛?|
| aten.clone | TensorMove | 鍐呭瓨鎷疯礉 |
| aten.scatter.value | ScatterElementsV2 | Scatter 鍐欏叆 |
| aten.sum.dim_IntList | ReduceSum | 姹傚拰褰掔害 |
| aten.topk | TopKV2 | Top-K 閫夋嫨 |

### TensorCast 铻嶅悎绠楀瓙
| TC 绠楀瓙 | NPU 鍐呮牳 | 璇存槑 |
|---------|---------|------|
| tc.swiglu | SwiGlu | SwiGlu 婵€娲?|
| tc.rms_norm | RmsNorm | RMS 褰掍竴鍖?|
| tc.add_rms_norm/2 | AddRmsNorm | 娈嬪樊 + RmsNorm |
| tc.apply_rope | InterleaveRope | RoPE锛堝閫夛細ApplyRotaryPosEmb锛?|
| tc.attention | FusedInferAttentionScore | 铻嶅悎娉ㄦ剰鍔?|
| tc.reshape_and_cache | ReshapeAndCacheNdKernel | KV cache 鍐欏叆 |
| tc.kv_rmsnorm_rope_cache | KvRmsNormRopeCache | 铻嶅悎 KV norm+RoPE+cache |

### 閲忓寲绠楀瓙
| TC 绠楀瓙 | NPU 鍐呮牳 | 璇存槑 |
|---------|---------|------|
| tc.static_quant_linear | QuantBatchMatmulV3 | INT8 鐭╅樀涔樻硶 |
| tc.static_quant_linear_int4 | QuantBatchMatmulV3 | INT4 鐭╅樀涔樻硶 |
| tc.fp8_linear | QuantBatchMatmulV3 | FP8 鐭╅樀涔樻硶 |
| tc.quantize | AscendQuantV2 | 闈欐€侀噺鍖?|
| tc.dynamic_quantize_symmetric | DynamicQuant | 鍔ㄦ€侀噺鍖?|
| tc.grouped_matmul_quant_swiglu | GroupedMatmulSwigluQuant | MoE 铻嶅悎 gate-up |
| tc.grouped_matmul_quant | GroupedMatmul | MoE 鐭╅樀涔樻硶 |

### 閫氫俊绠楀瓙
| TC 绠楀瓙 | NPU 鍐呮牳 | 璇存槑 |
|---------|---------|------|
| tc.all_reduce | hcom_allReduce_ | HCCL all-reduce |
| tc.all_gather | hcom_allGather_ | HCCL all-gather |
| tc.all_to_all | hcom_alltoallv_ | HCCL all-to-all锛圡oE锛?|
| tc.reduce_scatter | HcomReduceScatter | HCCL reduce-scatter |

### 澶嶅悎绠楀瓙
| TC 绠楀瓙 | 瀛愬唴鏍?| 璇存槑 |
|---------|-------|------|
| tc.matmul_all_reduce | MatMulV2 + hcom_allReduce_ | MC2 铻嶅悎 |
| tc.static_quant_linear_all_reduce | QuantBatchMatmulV3 + hcom_allReduce_ | 閲忓寲 MC2 |
| tc.multihead_latent_attention | BatchMatMulV2 + FIA + batch_matmul_transpose | MLA decode |
| tc.mlapo | MatMulV2 + KvRmsNormRopeCache | MLA 棰勫鐞?|

### 闆跺紑閿€绠楀瓙
view, permute, split, split_with_sizes, select, slice, transpose, unsqueeze, expand, full, detach, alias, arange, t, convert_element_type

## 12. 宸ュ叿鍙傝€?

| 宸ュ叿 | 鐢ㄩ€?| 鍏抽敭鍙傛暟 |
|------|------|---------|
| `parse_kernel_details.py` | 灏?kernel_details.csv 鎷嗗垎涓洪€愬唴鏍?CSV | `--device`, `--vllm-ascend-version`, `--kernel-details-path` |
| `generate_shape_grid.py` | 鐢熸垚寰熀鍑嗘祴璇?shape 缃戞牸 | 鈥?|

## 13. 鐩稿叧鏂囨。

- [璁捐鏂囨。](../OPERATOR_PERF_DATABASE_DESIGN_zh_v1.2.md) 鈥斺€?瀹屾暣鏋舵瀯鍜岃璁″師鐞?
- [Op Mapping 鎶€鑳絔(../skills/op-mapping/SKILL.md) 鈥斺€?浣跨敤骞惰瀛愪唬鐞嗙殑鑷姩鍖?op_mapping 鐢熸垚
- [Spike 鎶ュ憡](../reports/spike_executive_summary_zh.md) 鈥斺€?鍒濆 spike 璋冪爺缁撴灉

