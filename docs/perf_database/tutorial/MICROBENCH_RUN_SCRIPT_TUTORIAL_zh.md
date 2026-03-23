# Microbench `xxx_run.py` 鐢熸垚鏁欑▼

鏈枃璇存槑濡備綍涓烘煇涓?profiling kernel 鐢熸垚 `tools/perf_data_collection/op_replay/<KernelType>_run.py`锛屽苟璁╁畠鑳藉鍦?NPU 涓婄湡瀹炲洖鏀捐绠楀瓙锛屼緵 `run_all_op.py` 鍜?`profile_and_update_db.py` 浣跨敤銆?

## 1. 鐩爣

杈撳叆锛?

- 涓€涓畻瀛?CSV锛屼緥濡傦細
  `tensor_cast/performance_model/perf_database/data/<device>/vllm_ascend/<version>/<KernelType>.csv`
- 鍚岀増鏈殑 `op_mapping.yaml`
- 瀵瑰簲涓婃父浠撳簱閲岀殑鎺ュ彛鏂囨。銆佹祴璇曟垨瀹炵幇

杈撳嚭锛?

- `tools/perf_data_collection/op_replay/<KernelType>_run.py`

杩欎釜鑴氭湰鐨勮亴璐ｆ槸锛?

1. 璇诲彇 CSV 鐨勬瘡涓€琛?
2. 鏍规嵁 `Input Shapes / Input Data Types / Input Formats` 閲嶅缓杈撳叆
3. 璋冪敤鐪熷疄鎺ュ彛鎵ц绠楀瓙
4. `torch.npu.synchronize()`
5. 杈撳嚭涓€琛岀畝娲佺殑 `[OK]` 鏃ュ織

## 2. 鍏堢湅 `op_mapping.yaml`

涓嶈鍏堢寽鎺ュ彛鍚嶃€傚厛鎵撳紑瀵瑰簲鐗堟湰鐨勬暟鎹洰褰曚笅鐨?`op_mapping.yaml`锛屾煡 `torch_npu_reference`锛?

```yaml
torch_npu_reference:
  <KernelType>:
    microbench_api: "..."
```

渚嬪锛?

- `AscendQuantV2 -> torch_npu.npu_quantize`
- `DynamicQuant -> torch_npu.npu_dynamic_quant`
- `TensorMove -> torch.Tensor.copy_`
- `split_qkv_rmsnorm_rope_kernel -> torch.ops.vllm.qkv_rmsnorm_rope`

杩欓噷鐨?`microbench_api` 灏辨槸瑕佸湪 `xxx_run.py` 閲岀湡姝ｈ皟鐢ㄧ殑鎺ュ彛銆?

## 3. 鍒板摢閲屾壘鏂囨。鍜岀敤渚?

浼樺厛浣跨敤鏈湴浠撳簱銆傛病鏈夊氨 clone 鍒?`msmodeling` 鍚岀骇鐩綍銆?

寤鸿鐨勭洰褰曞竷灞€锛?

```text
<workspace>/
鈹溾攢 msmodeling/
鈹溾攢 vllm/
鈹溾攢 vllm-ascend/
鈹溾攢 op-plugin/
鈹溾攢 pytorch/
鈹溾攢 cann-ops-nn/
鈹溾攢 cann-ops-transformer/
鈹溾攢 cann-ops-math/
鈹斺攢 ascend-transformer-boost/
```

鎺ㄨ崘鎼滅储椤哄簭锛?

1. `op-plugin/docs/context/` 涓嬬殑绠楀瓙鏂囨。
2. `op-plugin/test/` 涓嬬殑娴嬭瘯鐢ㄤ緥
3. `vllm-ascend` 涓殑 Python/Triton/custom op 瀹炵幇
4. `op-plugin` / `pytorch-npu` 涓殑 op-api 鎴栨敞鍐屼唬鐮?
5. CANN / ATB 浠撳簱閲岀殑 layout銆乧ache銆佽瀺鍚堣涔?

甯哥敤鎼滅储鍛戒护锛?

```bash
git grep -n "torch_npu.npu_dynamic_quant"
git grep -n "npu_kv_rmsnorm_rope_cache"
git grep -n "qkv_rmsnorm_rope"
git grep -n "reshape_and_cache"
```

## 4. 鍏堣 CSV锛屽啀鍐冲畾鑴氭湰绛栫暐

鍏堢湅 CSV 鐨勮繖浜涘垪锛?

- `Input Shapes`
- `Input Data Types`
- `Input Formats`
- `Output Shapes`
- `Output Data Types`

瑕佸厛鍥炵瓟鍑犱釜闂锛?

1. 杩欎釜 kernel 涓€琛岄噷鍒板簳鏈夊嚑涓緭鍏ユЫ浣嶏紵
2. 鏈夋病鏈夌┖妲戒綅锛?
3. 杈撳嚭鏄竴涓繕鏄涓紵
4. CSV 閲屾湁娌℃湁璁板綍鍏ㄩ儴鍙傛暟锛岃繕鏄己灏戜竴閮ㄥ垎闈?Tensor 鍙傛暟锛?
5. 褰撳墠鐗堟湰鐨勬暟鎹槸涓嶆槸鍙湁涓€绉嶅舰鎬侊紵

鍘熷垯锛?

- 浼樺厛鍏堟敮鎸佲€滃綋鍓?CSV 鐪熷疄瀛樺湪鐨勮鈥?
- 涓嶈涓轰簡娉涘寲鎶婅剼鏈啓寰楄繃搴﹀鏉?

## 5. 鐢熸垚鑴氭湰鏃剁殑鍥哄畾濂楄矾

鏂扮殑 `xxx_run.py` 鍩烘湰閮介伒寰笅闈㈢粨鏋勶細

```python
from common import (
    build_input_tensor,
    build_standard_argparser,
    ensure_npu_available,
    get_runtime_modules,
    get_target_data_dir,
    iter_csv_rows,
    parse_list_field,
    parse_shape,
)
```

鐒跺悗閫氬父鍖呭惈锛?

- `build_argparser()`
- `build_row_case(row)`
- `run_row(csv_path, row_index, row)`
- `main()`

琛屼负妯″紡锛?

1. 鐢?`parse_list_field` / `parse_shape` 瑙ｆ瀽 metadata
2. 鐢?`build_input_tensor` 閲嶅缓 Tensor 杈撳叆
3. 瀵圭己澶辩殑闈?Tensor 鍙傛暟鍋氭渶灏忔帹瀵?
4. 璋冪湡瀹?API
5. `runtime_torch.npu.synchronize()`
6. 鎵撳嵃 `[OK]`

## 6. 甯歌鍙傛暟鎬庝箞琛?

寰堝绠楀瓙 CSV 涓嶄細鎶婃墍鏈夊弬鏁伴兘璁颁笅鏉ワ紝杩欐椂瑕佷粠鏂囨。鍜屾祴璇曢噷琛ャ€?

甯歌渚嬪瓙锛?

- `AscendQuantV2`
  - `axis=-1`
  - `div_mode=False`
  - 杈撳嚭 dtype 鏍规嵁 `Output Data Types` 鎺ㄥ

- `DynamicQuant`
  - 褰撳墠 CSV 鍙湁 `x`锛屾病鏈?`smooth_scales` / `group_index`
  - 鐩存帴璋冪敤 `torch_npu.npu_dynamic_quant(x)`

- `TensorMove`
  - CSV 鍙湁婧愬紶閲?
  - 鐩爣寮犻噺瑕佹寜鐩稿悓 shape/dtype/format 閲嶅缓锛屽啀鎵ц `dst.copy_(src)`

- `ReshapeAndCacheNdKernel`
  - `slot_mapping` 蹇呴』鍚堟硶锛屼笉鑳借秺鐣?
  - 瑕佹牴鎹?cache capacity 鏋勯€犱竴涓湁鏁堢储寮?

- `KvRmsNormRopeCache`
  - 鍙兘鏈?12 涓緭鍏ユЫ浣嶏紝浣嗗悗鍑犱釜鏄┖鐨?
  - 蹇呴』淇濈暀妲戒綅鏁伴噺锛屼笉鑳芥妸绌烘Ы浣嶄涪鎺?
  - `cache_mode`銆乣epsilon`銆乣is_output_kv` 寰€寰€闇€瑕佷粠娴嬭瘯鍜岃緭鍑?shape 鎺ㄥ

- `split_qkv_rmsnorm_rope_kernel`
  - 闇€瑕佸厛娉ㄥ唽 `torch.ops.vllm.*` 鑷畾涔?op
  - 鍙兘杩樿澶勭悊 `vllm_ascend` 鐨?import fallback

## 7. 浠€涔堟椂鍊欓渶瑕佺壒鍒鐞嗘牸寮?

浼樺厛鐪?`Input Formats`锛?

- `ND`
  - 閫氬父鐩存帴鐢?`build_input_tensor`

- `FRACTAL_NZ`
  - `common.py` 閲屽凡缁忔湁 `normalize_shape` 鍜?`npu_format_cast`
  - 涓€鑸笉瑕佽嚜宸辨墜鍐?NZ 灞曞紑閫昏緫锛岄櫎闈炲綋鍓嶇畻瀛愭湁鐗规畩缂撳瓨甯冨眬

- 璁板綍鍊煎儚 `NCL`
  - 鍦?`build_input_tensor` 閲屼笉浼氬仛鐗规畩 cast
  - 杩欑鎯呭喌閫氬父鍙渶瑕佹寜璁板綍 shape 閲嶅缓鍗冲彲

## 8. 濡備綍澶勭悊鑷畾涔?op

濡傛灉 `microbench_api` 鏄細

- `torch.ops.vllm.*`
- `torch.ops._C_ascend.*`
- `atb.*`

閭ｅ氨涓嶈兘鍙湅 `torch_npu` 鏂囨。浜嗭紝蹇呴』鍚屾椂妫€鏌ワ細

1. 鑷畾涔?op 鏄惁宸叉敞鍐?
2. 鏄惁闇€瑕佹墜鍔?import 娉ㄥ唽妯″潡
3. 鏄惁渚濊禆棰濆鐜鍙橀噺鎴?sibling repo 璺緞

渚嬪 `vllm_ascend` 鍦烘櫙鎺ㄨ崘鍋氭硶锛?

- 鍏堢洿鎺?`import vllm_ascend...`
- 濡傛灉澶辫触锛屽啀灏濊瘯锛?
  - 鐜鍙橀噺 `VLLM_ASCEND_PATH`
  - 鍚岀骇鐩綍 fallback

杩欐牱鑴氭湰鏃㈣兘鍦ㄦ湰鍦板紑鍙戠幆澧冭窇锛屼篃鑳藉湪宸插畨瑁呯幆澧冭窇銆?

## 9. 鏈€灏忛獙璇佹楠?

鐢熸垚鑴氭湰鍚庯紝鑷冲皯鍋氫笅闈袱姝ワ細

```bash
py -3 -m py_compile tools/perf_data_collection/op_replay/<KernelType>_run.py
py -3 tools/perf_data_collection/op_replay/<KernelType>_run.py --help
```

濡傛灉鐜鏈?NPU锛屽啀鍋氫竴娆＄湡瀹炶繍琛岋細

```bash
python tools/perf_data_collection/op_replay/<KernelType>_run.py \
  --device ATLAS_800_A3_752T_128G_DIE \
  --vllm-ascend-version 0.13.0
```

## 10. 鎻愪氦鏃惰娉ㄦ剰浠€涔?

閫氬父鍙彁浜わ細

- `tools/perf_data_collection/op_replay/<KernelType>_run.py`

涓嶈璇彁浜わ細

- 鏈湴 clone 涓嬫潵鐨勪笂娓镐粨搴?
- `perf_database/data/...` 閲屼綘涓存椂鐢熸垚鎴栬В鍘嬬殑鍐呭
- profiling 杈撳嚭鐩綍

鎺ㄨ崘鍏堢湅锛?

```bash
git status --short
```

鍐嶅崟鐙?add锛?

```bash
git add -- tools/perf_data_collection/op_replay/<KernelType>_run.py
```

## 11. 涓€涓疄闄呭伐浣滄祦绀轰緥

浠?`DynamicQuant` 涓轰緥锛?

1. 鎵撳紑 `op_mapping.yaml`
2. 鎵惧埌 `torch_npu_reference.DynamicQuant.microbench_api = torch_npu.npu_dynamic_quant`
3. 璇?`op-plugin/docs/context/torch_npu-npu_dynamic_quant.md`
4. 璇?`DynamicQuant.csv`
5. 鍙戠幇褰撳墠 CSV 鍙湁鍗曡緭鍏?`x`
6. 鐢熸垚鑴氭湰锛岀洿鎺ヨ皟鐢細

```python
output, scale = runtime_torch_npu.npu_dynamic_quant(x_tensor)
```

7. 鍋?`py_compile` 鍜?`--help` 楠岃瘉
8. 鍗曟枃浠舵彁浜?

## 12. 寤鸿鐨勫垽鏂師鍒?

- 鍏堜俊 `op_mapping.yaml` 缁欏嚭鐨?`microbench_api`
- 鍐嶄俊娴嬭瘯鐢ㄤ緥
- 鍐嶄俊瀹炵幇鏂囦欢
- 鏈€鍚庢墠鍋氬繀瑕佹帹鏂?

濡傛灉瑕佹帹鏂弬鏁帮紝閬靛畧涓ゆ潯锛?

1. 鍙负鈥滃綋鍓?CSV 鐪熷疄瀛樺湪鐨?case鈥濇帹鏂?
2. 閫夋嫨鏈€灏忋€佹渶绋冲畾銆佹渶涓嶅鏄撹窇鍋忕殑瑙勫垯

杩欐牱鐢熸垚鍑烘潵鐨?`xxx_run.py` 鏇村鏄撶湡瀹炲洖鏀?profiling 鏁版嵁锛屼篃鏇村鏄撳悗缁淮鎶ゃ€?

