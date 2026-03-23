# TensorCast 绠楀瓙鎬ц兘鏁版嵁搴擄細鎶€鏈璁℃枃妗?

**鐗堟湰**: 1.3.1
**鏃ユ湡**: 2026.3.12
**鍙樻洿**: CANN 8.5 鐩爣閫傞厤 + Phase 1 杩涘睍鍙嶆槧 + 鎺ュ彛鍙樻洿钀藉疄
**鑼冨洿**: 闈㈠悜 LLM 浠跨湡鐨勫彲鎵╁睍瀹炴祴鎬ц兘妯″瀷锛屼笉缁戝畾鍏蜂綋绠楀姏鍗★紝鏀寔鍩轰簬瀹炴祴 Profiling 鏁版嵁鍜?Microbenchmark 鏁版嵁鐨勭畻瀛愭€ц兘浼扮畻銆?
**鍒濇湡鐩爣妯″瀷**: DeepSeek-V3銆丵wen3-32B

浣滆€咃細HXW
瀹℃牳浜猴細GJ

---

## 1. 鍔熻兘姒傝堪

### 1.1 鐩爣

涓?TensorCast 浠跨湡鍣ㄦ瀯寤哄熀浜庡疄娴嬫暟鎹殑绠楀瓙鎬ц兘浼扮畻绯荤粺銆傞噸鏋勭幇鏈?`EmpiricalPerformanceModel`锛屼娇鍏舵帴鍙楅€氱敤鐨?`DataSource` 鎶借薄鎺ュ彛锛屾敮鎸佸绉嶆暟鎹潵婧愶紙棰勯噰闆?Profiling 鏁版嵁搴撱€丣IT Benchmark 缂撳瓨绛夛級銆傞€氳繃 `ProfilingDataSource` 瀹炵幇鍩轰簬棰勯噰闆嗘暟鎹殑鏌ヨ锛屼笌鐜版湁 `AnalyticPerformanceModel`锛圧oofline锛夊苟鍒楋紝浣滀负鐢ㄦ埛鍙€夌殑鎬ц兘妯″瀷銆?

鏍稿績鏋舵瀯涓?`EmpiricalPerformanceModel` + `DataSource` 妯″紡锛?
- `EmpiricalPerformanceModel`锛氱粺涓€鐨勬€ц兘妯″瀷鍏ュ彛锛屾帴鍙椾笉鍚?`DataSource` 瀹炰緥瀹炵幇涓嶅悓琛屼负
- `ProfilingDataSource`锛氬熀浜庨閲囬泦 Profiling CSV 鐨勫彧璇绘暟鎹簮锛屽唴閮ㄥ鐞嗙畻瀛愭槧灏勩€丗RACTAL_NZ 鏍煎紡杞崲銆乨type 鍖归厤
- `InterpolatingDataSource`锛歐rapper DataSource锛屽湪搴曞眰 DataSource 鍩虹涓婃彁渚涙彃鍊?杩戜技鏌ヨ鑳藉姏
- `CacheKeyDataSource`锛氬熀浜?`OpInvokeInfo.cache_key` 鐨勭簿纭尮閰嶆暟鎹簮锛堟湭鏉ユ墿灞曪級

### 1.2 鏍稿績鍔熻兘锛堟湰鏂规鑼冨洿锛?

1. **DataSource 鎶借薄灞傦紙鏂板锛?*锛氬畾涔夋爣鍑嗗寲鏌ヨ鎺ュ彛 `lookup(OpInvokeInfo)`锛屼笌鏁版嵁鏉ユ簮瑙ｈ€︺€俙ProfilingDataSource` 涓洪瑕佸疄鐜帮紝鍐呴儴澶勭悊 `op_mapping.yaml` 鏄犲皠 + CSV 鏁版嵁鏌ヨ + FRACTAL_NZ 鏍煎紡杞崲
2. **EmpiricalPerformanceModel锛堥噸鏋勶級**锛氭帴鍙?`DataSource` 瀹炰緥锛屼粠鏁版嵁婧愭煡璇㈢畻瀛愯€楁椂锛涙湭鍛戒腑鏃跺唴閮?fallback 鑷?`AnalyticPerformanceModel`锛堣绠楃畻瀛愶級鎴?`CommAnalyticModel`锛堥€氫俊绠楀瓙锛?
3. **InterpolatingDataSource锛堟柊澧烇級**锛歐rapper 妯″紡锛屽湪搴曞眰 DataSource 鐨勭簿纭尮閰嶅熀纭€涓婃彁渚涙彃鍊?澶栨帹鑳藉姏
4. **op_mapping.yaml 閰嶇疆椹卞姩锛堟柊澧烇級**锛氱函鍚嶅瓧鏄犲皠锛圱ensorCast func 鈫?Profiling kernel Type锛夛紝涓嶅惈 per-op 缁村害鎻愬彇閫昏緫锛屾墍鏈夌淮搴﹀尮閰嶅拰鏍煎紡杞崲鐢?ProfilingDataSource 閫氱敤浠ｇ爜澶勭悊
5. **鏁版嵁閲囬泦宸ュ叿閾?*锛堢嫭绔嬪瓙绯荤粺锛屼綅浜?`tools/perf_data_collection/`锛夛細Profiling 鏁版嵁瑙ｆ瀽銆丮icrobenchmark 鑴氭湰鐢熸垚銆佹暟鎹簱鏋勫缓涓庨獙璇?

### 1.3 涓嶅湪鏈柟妗堣寖鍥村唴锛堝缓璁悗缁敮鎸侊級

- **CompositePerformanceModel 椤跺眰璋冨害鍣?*锛氱粺涓€缂栨帓澶氱 PerformanceModel锛屽疄鐜板彲閰嶇疆鐨勯檷绾ф垨缁勫悎绛栫暐锛堣瑙佺 9.3 鑺傦級
- **璺ㄧ‖浠舵硾鍖?*锛氬綋鍓嶄粎鏀寔鏄囪吘 A3锛屽叾浠栫‖浠堕渶鐙珛閲囬泦鏁版嵁
- **鑷姩鍖栨寔缁泦鎴?*锛氶殢 vLLM-Ascend / CANN 鐗堟湰鍙戝竷鑷姩瑙﹀彂鏁版嵁閲囬泦

### 1.4 鍒濇湡鐩爣

- **鐩爣妯″瀷**锛欴eepSeek-V3銆丵wen3-32B
- **鐩爣纭欢**锛欰tlas 800 A3锛?52T锛?28G DIE锛?
- **鐩爣鍚庣**锛歷llm-ascend 0.15.0锛圕ANN 8.5锛宼orch 2.9.0锛?
- **绮惧害鐩爣**锛氱鍒扮浠跨湡璇樊 <15%锛堝姣斿疄闄?vLLM Profiling锛?
- **浜や粯鏃堕棿**锛?026.3.23 瀹屾垚绔埌绔泦鎴愬苟瀹屾垚鍒濆鏁版嵁閲囬泦鍜岄泦鎴愭祴璇?

> **v1.3.1 娉?*锛氱洰鏍囩増鏈粠 vllm-ascend 0.13.0锛圕ANN 8.3锛夊崌绾ц嚦 0.15.0锛圕ANN 8.5锛夈€侰ANN 8.3 鏁版嵁淇濈暀浣滀负鍙傝€冨熀绾裤€?

---

## 2. 鎶€鏈垎鏋?

### 2.1 闂闄堣堪

TensorCast 褰撳墠閲囩敤**鍩轰簬 Roofline 鐨勮В鏋愭ā鍨?*锛坄AnalyticPerformanceModel`锛変及绠楃畻瀛愭墽琛岃€楁椂銆傝妯″瀷鍩轰簬娴偣杩愮畻閲忥紙FLOPs锛変笌璁垮瓨瀛楄妭鏁拌绠?`max(璁＄畻鑰楁椂, 璁垮瓨鑰楁椂)`锛屼富瑕佺敤浣滅悊璁烘€ц兘涓婄晫璇勪及銆?

**Roofline 鍋忓樊鏍瑰洜**锛?

| 鍋忓樊鏉ユ簮 | 褰卞搷鍦烘櫙 | 鎻忚堪 |
|---------|---------|------|
| **纭欢鍒╃敤鐜囧彈 tiling 绛栫暐褰卞搷** | 浣庝及灏?batch 鍦烘櫙鑰楁椂锛堝惈 Decode 鍜屽皬 batch Prefill锛?| 灏?batch 涓?AI Core 鍗犵敤鐜囦笉瓒筹紙tile 鏁板皯浜?AI Core 鏁伴噺瀵艰嚧绌洪棽锛夈€乼iling 绮掑害涓嶅尮閰嶏紙鐭╅樀缁村害鏈榻?tile size 浜х敓 padding 娴垂锛夈€丮TE-Cube-Vector 娴佹按绾挎棤娉曞～婊?|
| **铻嶅悎鍐呮牳宸紓** | 閮ㄥ垎绠楀瓙 | vLLM 瀹為檯閮ㄧ讲浣跨敤娣卞害铻嶅悎鐨勭‖浠跺唴鏍革紙濡?`DequantSwigluQuant`銆乣KvRmsNormRopeCache`锛夛紝涓?TensorCast 鐨勫師瀛愮畻瀛?dispatch 瀛樺湪绮掑害宸紓 |
| **纭欢鐗规€ф湭瀹屾暣寤烘ā** | Memory-bound 鍦烘櫙 | Cache 灞傜骇銆乥ank conflict銆佸唴瀛樿闂?pattern 绛夊井鏋舵瀯鐗规€ф湭鍦?Roofline 涓綋鐜?|

> 鍏充簬灏?batch 鍦烘櫙鐨勮缁嗗垎鏋愶紙鍚?Decode 鍜屽皬 batch Prefill 鐨勫尯鍒級瑙侀檮褰?A銆?

### 2.2 Profiling 鏁版嵁鍒嗘瀽

鍩轰簬 DeepSeekV3 Decode锛?2 鍗★級鍜?Qwen3-30B Prefill锛?6 鍗★級鐨?kernel_details.csv 鍒嗘瀽锛?

**DeepSeekV3 Decode 鍏抽敭绠楀瓙锛堟寜璋冪敤娆℃暟鎺掑簭锛孴op 15锛?*锛?
QuantBatchMatmulV3: 15006, AscendQuantV2: 10004, Add: 7545, TransposeBatchMatMul: 5002, InplaceAddRmsNorm: 5002, DequantSwigluQuant: 4879, GroupedMatmul: 4756, FusedInferAttentionScore: 2501, KvRmsNormRopeCache: 2501, InterleaveRope: 2501, DynamicQuant: 2501, MoeGatingTopK: 2378, MoeDistributeDispatchV2: 2378, MoeDistributeCombineV2: 2378, MatMul: 2378, MatMulV2: 41

**Qwen3-30B Prefill 鍏抽敭绠楀瓙锛堟寜璋冪敤娆℃暟鎺掑簭锛?*锛?
TensorMove: 386, hcom_allReduce_: 276, MatMulV2: 275, AddRmsNorm: 131, FusedInferAttentionScore: 67, SwiGlu: 67, ReshapeAndCacheNdKernel: 67, split_qkv_rmsnorm_rope_kernel: 64

> **v1.3.1 娉?*锛氫笂杩版暟鎹熀浜?CANN 8.1/8.3 鐗堟湰銆侰ANN 8.5 寮曞叆浜?`DispatchFFNCombine` 瓒呯骇铻嶅悎绠楀瓙锛堣瀺鍚?`all_to_all脳2 + GroupedMatmul脳2 + SwiGlu + MoE routing`锛夛紝鍗?DSV3 Decode 绔埌绔€楁椂 **35.3%**锛屾槸 DSV3 鏈€閲嶈鐨勫崟涓€ kernel銆俆C 閫氳繃 `composite: true` 鍒嗚В鏌ヨ瑕嗙洊銆傝瑙?搂9.1 铻嶅悎 Gap 鐘舵€併€?

### 2.3 鐗堟湰褰卞搷鍒嗘瀽

绠楀瓙鎬ц兘鍙椾互涓嬬増鏈洜绱犲奖鍝嶏細
- **CANN 鐗堟湰**锛氬奖鍝?kernel 瀹炵幇銆乼iling 绛栫暐銆侀€氫俊搴擄紙HCCL锛夎涓?
- **vLLM-Ascend 鐗堟湰**锛氬奖鍝嶈瀺鍚堢畻瀛愬疄鐜帮紙濡?MC2锛夈€乲ernel 璋冨害绛栫暐
- **纭欢鍨嬪彿**锛氱洿鎺ュ喅瀹氳绠?璁垮瓨/閫氫俊鎬ц兘鍙傛暟

鏁版嵁瀛樺偍鎸?`{device}/{backend}/{version}/` 灞傜骇绠＄悊锛岄€氫俊鏁版嵁鎸?`{device}/hccl/{cann_version}/` 鍗曠嫭绠＄悊锛堣法 vLLM 鐗堟湰澶嶇敤锛夈€?

### 2.4 鐜版湁鍩虹璁炬柦

TensorCast 宸叉湁瀹屽杽鐨勬€ц兘妯″瀷妗嗘灦锛?
- **`PerformanceModel` 鍩虹被**锛氱粺涓€鎺ュ彛 `process_op(OpInvokeInfo) 鈫?Result`
- **`AnalyticPerformanceModel`**锛歊oofline 妯″瀷锛坄tensor_cast/performance_model/analytic.py`锛?
- **`CommAnalyticModel`**锛氶€氫俊绠楀瓙瑙ｆ瀽妯″瀷锛坄tensor_cast/performance_model/comm_analytic.py`锛?
- **`CachingPerformanceModel`**锛歋HA256 cache_key 绾у埆鐨?session 鍐呯紦瀛?
- **`OpInvokeInfo`**锛氱畻瀛愬厓鏁版嵁锛坒unc, args, cache_key锛夛紝鏀寔 `register_op_properties` 鎵╁睍
- **`DeviceProfile` + `CommGrid`**锛氱‖浠惰鏍?+ 缃戠粶鎷撴墤鎻忚堪

### 2.5 AI Configurator 鍙傝€?

[AI Configurator](https://github.com/ai-dynamo/aiconfigurator) 鏄?NVIDIA 鐨?LLM 鎺ㄧ悊鎬ц兘棰勪及宸ュ叿锛屽叾璁捐瀵规湰鏂规鏈夊弬鑰冧环鍊硷細

- **鏁版嵁鏍煎紡**锛欳SV锛坄.txt` 鍚庣紑锛夛紝姣忚鍚噸澶嶇殑 framework/version/device 鍒楀疄鐜拌嚜鎻忚堪鎬э紝鏀寔璺ㄧ増鏈仛鍚?
- **鎻掑€肩瓥鐣?*锛?D+1D 娣峰悎鎻掑€硷紝瀵?O(n虏) 鐨?Attention 绠楀瓙鍋?sqrt 鍙樻崲鍚庡啀鎻掑€?
- **DatabaseMode**锛歚SILICON 鈫?HYBRID 鈫?EMPIRICAL 鈫?SOL` 绾ц仈妯″紡
- **Git LFS 绠＄悊**锛歚.gitattributes` 涓厤缃?`systems/**/*.txt filter=lfs`
- **閫氫俊鏁版嵁**锛歂CCL 鍘熺敓閫氫俊鎸?`nccl/{version}/` 鐙珛瀛樺偍锛岃法妗嗘灦鐗堟湰鍏变韩锛汣ustom AllReduce 鍜屾鏋剁増鏈粦瀹?

> 閫氫俊鏂规鐨勮缁嗗姣斿垎鏋愯闄勫綍 F銆?

鎴戜滑鐨勬柟妗堜笌 AI Configurator 鐨勫尯鍒細
- **涓嶅鍒堕噸澶嶅垪妯″紡**锛欳SV 鎸夌洰褰曞垎绾у瓨鍌紝涓婁笅鏂囦俊鎭湪 `op_mapping.yaml` 涓?
- **鍙傝€冨叾 Git LFS 绠＄悊鏂瑰紡**
- **鍙傝€冨叾閫氫俊鏁版嵁鎸夊簱鐗堟湰鐙珛瀛樺偍鐨勬ā寮?*锛坔ccl/{cann_version}/锛?

---

## 3. 绯荤粺鏋舵瀯

### 3.1 鏁翠綋鏋舵瀯

```
鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
鈹?                          TensorCast Runtime                              鈹?
鈹?                                                                          鈹?
鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?    鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹? 鈹?
鈹? 鈹? Runtime            鈹?    鈹? 鐢ㄦ埛鍙厤缃€夋嫨 PerformanceModel          鈹? 鈹?
鈹? 鈹? (TorchDispatchMode)鈹傗攢鈹€鈹€鈹€鈻垛攤                                          鈹? 鈹?
鈹? 鈹?                    鈹?    鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹? 鈹?
鈹? 鈹? 鎷︽埅鎵€鏈夌畻瀛愯皟鐢?  鈹?    鈹? 鈹?EmpiricalPerformanceModel锛堥噸鏋勶級鈹?   鈹? 鈹?
鈹? 鈹? 鐢熸垚 OpInvokeInfo  鈹?    鈹? 鈹? DataSource.lookup(OpInvokeInfo) 鈹?   鈹? 鈹?
鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?    鈹? 鈹? 鏈懡涓?鈫?fallback (Analytic)    鈹?   鈹? 鈹?
鈹?                             鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹? 鈹?
鈹?                             鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹? 鈹?
鈹?                             鈹? 鈹?AnalyticPerformanceModel锛堢幇鏈夛級 鈹?   鈹? 鈹?
鈹?                             鈹? 鈹? Roofline 鐞嗚妯″瀷               鈹?   鈹? 鈹?
鈹?                             鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹? 鈹?
鈹?                             鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹? 鈹?
鈹?                                          鈹?                               鈹?
鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹尖攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
                                            鈹?
                      鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹尖攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
                      鈹?                    鈻?                    鈹?
                      鈹?      DataSource锛堟娊璞℃帴鍙ｏ級              鈹?
                      鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹?
                      鈹? 鈹? ProfilingDataSource              鈹?   鈹?
                      鈹? 鈹? op_mapping.yaml 鏄犲皠 + CSV 鏌ヨ  鈹?   鈹?
                      鈹? 鈹? FRACTAL_NZ 鎭㈠ + dtype 鍖归厤     鈹?   鈹?
                      鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹?
                      鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹?
                      鈹? 鈹? InterpolatingDataSource (Wrapper)鈹?   鈹?
                      鈹? 鈹? 绮剧‘鍖归厤 鈫?鎻掑€?鈫?澶栨帹            鈹?   鈹?
                      鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹?
                      鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹?
                      鈹? 鈹? CacheKeyDataSource锛堟湭鏉ユ墿灞曪級   鈹?   鈹?
                      鈹? 鈹? JIT benchmark 缂撳瓨               鈹?   鈹?
                      鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹?
                      鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹?
                      鈹? 鈹? 瀛樺偍灞?                           鈹?   鈹?
                      鈹? 鈹? 璁＄畻: vllm_ascend/{version}/*.csv 鈹?   鈹?
                      鈹? 鈹? 閫氫俊: hccl/{cann_version}/*.csv   鈹?   鈹?
                      鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹?
                      鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?

鏁版嵁閲囬泦宸ュ叿锛堢绾挎墽琛岋紝鐙珛瀛愮郴缁燂級
鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
鈹? 涓夋璧帮細                                                     鈹?
鈹? 1. vLLM Profiling 鈫?kernel_details.csv 鈫?鎸?Type 鎷嗗垎 CSV  鈹?
鈹? 2. Microbenchmark 缃戞牸閬嶅巻 鈫?鎵╁厖 CSV 瑕嗙洊鑼冨洿              鈹?
鈹? 3. 绔埌绔獙璇?鈫?绮惧害鎶ュ憡                                    鈹?
鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
```

**璁捐鍘熷垯**锛?
- **閰嶇疆浼樹簬浠ｇ爜**锛氱畻瀛愭槧灏勯€氳繃 `op_mapping.yaml` 閰嶇疆锛屾柊澧炵畻瀛愬彧闇€淇敼 YAML锛屼笉闇€鍐?Python 浠ｇ爜
- **鏁版嵁涓庨€昏緫瑙ｈ€?*锛歚ProfilingDataSource` 澶勭悊鎵€鏈夋槧灏勫拰鏍煎紡杞崲锛屼笂灞?`EmpiricalPerformanceModel` 鍙叧蹇?`lookup(OpInvokeInfo) 鈫?Optional[QueryResult]`
- **閫氱敤 FRACTAL_NZ 澶勭悊**锛氭牸寮忚浆鎹㈢敱 `ProfilingDataSource` 鍐呴€氱敤鍑芥暟澶勭悊锛屼笉渚濊禆绠楀瓙绫诲瀷

### 3.2 妯″潡缁撴瀯

```
tensor_cast/performance_model/
鈹溾攢鈹€ analytic.py                           # AnalyticPerformanceModel锛堢幇鏈夛級
鈹溾攢鈹€ comm_analytic.py                      # CommAnalyticModel锛堢幇鏈夛級
鈹溾攢鈹€ memory_tracker.py                     # MemoryTracker锛堢幇鏈夛級
鈹溾攢鈹€ empirical.py                          # EmpiricalPerformanceModel锛堥噸鏋勶級
鈹斺攢鈹€ perf_database/                        # 鏂板锛欴ataSource + 鏁版嵁瀛樺偍
    鈹溾攢鈹€ __init__.py
    鈹溾攢鈹€ data_source.py                    # DataSource ABC + QueryResult
    鈹溾攢鈹€ profiling_data_source.py          # ProfilingDataSource锛圕SV 鏌ヨ + FRACTAL_NZ锛?
    鈹溾攢鈹€ interpolating_data_source.py      # InterpolatingDataSource锛圵rapper 鎻掑€硷級
    鈹斺攢鈹€ data/                             # 鎬ц兘鏁版嵁瀛樺偍锛圙it LFS 绠＄悊 .csv锛?
        鈹斺攢鈹€ ATLAS_800_A3_752T_128G_DIE/
            鈹溾攢鈹€ vllm_ascend/
            鈹?  鈹溾攢鈹€ v0.13.0/                         # CANN 8.3 legacy
            鈹?  鈹溾攢鈹€ vllm0.13.0_torch2.8.0_cann8.3/  # CANN 8.3 瀹屾暣鍛藉悕
            鈹?  鈹斺攢鈹€ vllm0.15.0_torch2.9.0_cann8.5/  # CANN 8.5 鐢熶骇鐩爣
            鈹?      鈹溾攢鈹€ op_mapping.yaml
            鈹?      鈹斺攢鈹€ {KernelType}.csv             # ~52 涓?CSV
            鈹斺攢鈹€ hccl/v8.5/               # HCCL 閫氫俊锛堝拰 CANN 鐗堟湰缁戝畾锛岃法 vLLM 鐗堟湰澶嶇敤锛?
                鈹溾攢鈹€ hcom_allReduce_.csv
                鈹溾攢鈹€ hcom_allGather_.csv
                鈹溾攢鈹€ hcom_reduceScatter_.csv
                鈹斺攢鈹€ hcom_alltoallv_.csv

tools/perf_data_collection/               # 鏁版嵁閲囬泦涓庢暟鎹簱鏋勫缓宸ュ叿
鈹溾攢鈹€ parse_kernel_details.py               # 瑙ｆ瀽 kernel_details.csv 鈫?鎸?Type 鎷嗗垎
鈹溾攢鈹€ database build tool                     # 浠?Profiling 鏁版嵁鏋勫缓 CSV 鏁版嵁搴?
鈹溾攢鈹€ microbenchmark generation tool                # 鐢熸垚璁＄畻绠楀瓙 microbenchmark Python 鑴氭湰
鈹溾攢鈹€ generate_comm_microbench.py           # 鐢熸垚閫氫俊绠楀瓙 microbenchmark 鑴氭湰
鈹溾攢鈹€ generate_shape_grid.py                # 鐢熸垚 Shape 閬嶅巻缃戞牸
鈹溾攢鈹€ database validation tool                           # 鏁版嵁搴撶簿搴﹂獙璇?
鈹斺攢鈹€ operator coverage check tool                 # 鏂版ā鍨嬬畻瀛愬彂鐜?
```

### 3.3 鏁版嵁瀛樺偍缁撴瀯

**璁捐鍐崇瓥**锛?

| 鍐崇瓥椤?| 鏂规 | 鐞嗙敱 |
|-------|------|------|
| 鏁版嵁浣嶇疆 | `tensor_cast/performance_model/perf_database/data/` | 鏁版嵁鏄?TensorCast 鍔熻兘鐨勪竴閮ㄥ垎锛堢被浼?`device_profiles/`锛夛紝鐢ㄦ埛閫氳繃 `--performance-model profiling` 鏃堕渶瑕佹暟鎹殢鍖呭彲鐢?|
| 鏂囦欢鏍煎紡 | CSV锛圥rofiling 鍘熷鏍煎紡锛?| 涓?kernel_details.csv 瀹屽叏瀵归綈锛屼繚璇佸彲婧簮 |
| CSV 鍛藉悕 | Profiling Type 鍒楀師濮嬪ぇ灏忓啓 | `MatMulV2.csv`銆乣GroupedMatmul.csv` 绛夛紝涓?Profiling 鐩存帴瀵瑰簲 |
| 澶ф枃浠剁鐞?| Git LFS | `.gitattributes` 涓厤缃?`tensor_cast/performance_model/perf_database/data/**/*.csv filter=lfs diff=lfs merge=lfs -text` |
| 璁＄畻/閫氫俊鍒嗙 | 璁＄畻鍦?`vllm_ascend/{version}/`锛岄€氫俊鍦?`hccl/{cann_version}/` | HCCL 閫氫俊琛屼负鍙彇鍐充簬 CANN 鐗堟湰鍜岀‖浠讹紝涓?vLLM 鐗堟湰鏃犲叧锛涘崌绾?vLLM 涓嶉渶瑕侀噸娴嬮€氫俊 |
| 鐩綍鍛藉悕绾﹀畾 | `vllm{ver}_torch{ver}_cann{ver}` 瀹屾暣鐗堟湰瀛楃涓?| 閬垮厤鐗堟湰姝т箟锛屾敮鎸佸悓涓€璁惧涓嬪鐗堟湰鍏卞瓨锛坴1.3.1 鏇存柊锛?|
| MC2 鏁版嵁浣嶇疆 | `vllm_ascend/{version}/` | MC2锛坄npu_mm_all_reduce_base`锛夊拰 vLLM-Ascend 瀹炵幇缁戝畾 |

---

## 4. 鏍稿績妯″潡璁捐

### 4.1 DataSource 鎶借薄鍩虹被

```python
# tensor_cast/performance_model/perf_database/data_source.py

class QuerySource(Enum):
    MEASURED = auto()          # 绮剧‘鍖归厤锛堢疆淇″害: 1.0锛?
    INTERPOLATED = auto()      # 鎻掑€间及绠楋紙缃俊搴? 0.7-0.95锛?
    EXTRAPOLATED = auto()      # 鍑稿寘澶栨帹锛堢疆淇″害: 0.3-0.6锛?

@dataclass
class QueryResult:
    latency_us: float
    confidence: float
    source: QuerySource
    details: Dict[str, Any] = field(default_factory=dict)

class DataSource(ABC):
    """閫氱敤鏁版嵁婧愭娊璞″熀绫汇€?
    TensorCast 鍙€氳繃 OpInvokeInfo 鏌ヨ锛屼笉鎰熺煡搴曞眰鏄犲皠鍏崇郴鍜屾暟鎹牸寮忋€?""

    @abstractmethod
    def lookup(self, op_invoke_info: "OpInvokeInfo") -> Optional[QueryResult]:
        """浠?OpInvokeInfo 鏌ヨ绠楀瓙鎬ц兘銆傚唴閮ㄥ鐞嗘槧灏?缁村害鎻愬彇/鏁版嵁鏌ユ壘銆?
        杩斿洖 None 琛ㄧず鏈懡涓€?""
        ...

    def store(self, op_invoke_info: "OpInvokeInfo", result: QueryResult) -> None:
        """瀛樺偍鎬ц兘鏁版嵁锛堝彲閫夛紝閮ㄥ垎瀛愮被鍙锛夈€傞粯璁や笉鏀寔鍐欏叆銆?""
        raise NotImplementedError("This DataSource is read-only")
```

### 4.2 ProfilingDataSource

鍩轰簬棰勯噰闆?Profiling CSV 鐨勫彧璇绘暟鎹簮銆傛牳蹇冭亴璐ｏ細`op_mapping.yaml` 鏄犲皠 鈫?CSV 鏌ヨ 鈫?FRACTAL_NZ 鏍煎紡杞崲 鈫?dtype 鍖归厤銆?

**鍒濆鍖?*锛氭帴鍙?`(db_path, device_profile)` 鍙傛暟锛坴1.3.1锛歚device_profile` 鏇夸唬鍘?`comm_grid`锛岀粺涓€璁＄畻+閫氫俊鏁版嵁鐨勭‖浠跺弬鏁拌闂級銆傚姞杞?`op_mapping.yaml` + 鎸?`communication_data_ref` 鍔犺浇閫氫俊鏁版嵁锛堝鏈夛級銆侰SV 寤惰繜鍔犺浇骞剁紦瀛樸€?

**鏌ヨ鍒嗘淳**锛坄lookup()` 鍐呴儴锛夛細

```
func_name 鈫?op_mapping.yaml 鏌?mapping
  鈹溾攢 mapping 涓嶅瓨鍦?鈫?return None
  鈹溾攢 composite == true 鈫?_lookup_composite()
  鈹溾攢 category == "communication" 鈫?_lookup_comm()
  鈹溾攢 query_mode == "attention_special" 鈫?_lookup_attention()
  鈹斺攢 榛樿 鈫?_lookup_compute()
```

**璁＄畻绠楀瓙鍖归厤**锛坄_lookup_compute` + `_inputs_match`锛夛細浠?`OpInvokeInfo.args` 鎻愬彇鎵€鏈?tensor 鐨?`(shape, dtype)`锛岄€愯鍖归厤 CSV銆傚 FRACTAL_NZ 鏍煎紡鐨?input 鍏堣皟鐢?`fractal_nz_to_nd()` 鎭㈠涓?ND shape锛屽啀鍋氱簿纭尮閰嶃€侽utput shape 浣滀负楠岃瘉銆?

**Input 鏁伴噺涓嶅尮閰嶅鐞?*锛坴1.3.1 鏂板锛夛細TC dispatch 鐨?input 鏁伴噺鍙兘涓?Profiling CSV 涓嶅悓銆俙_inputs_match()` 鎸?kernel_type 搴旂敤杩囨护瑙勫垯锛?
- **榛樿**锛氬彇 `min(len(tc_inputs), len(csv_inputs))` 涓?input 鍋氬尮閰?
- **QuantBatchMatmulV3**锛氫粎鍖归厤鍓?2 涓?input锛坰cale/zero_point 鐢?NPU 鍐呴儴鐢熸垚锛孋SV 涓鍑?2 涓級
- **ReshapeAndCacheNdKernel**锛氫粎鍖归厤鍓?4 涓?input锛圕SV 涓?K/V cache 鎷嗗垎涓?5 涓級
- **TensorMove**锛氫粎鍖归厤 src tensor锛坉st 鍦?CSV 涓笉璁板綍锛?

> 鍘熷洜锛歂PU kernel 瀹為檯鎺ユ敹鐨?input 鍙兘鍖呭惈 TC 灞備笉鍙鐨勫唴閮ㄥ弬鏁帮紙閲忓寲 scale銆乥uffer 鎸囬拡绛夛級銆?

**閫氫俊绠楀瓙鍖归厤**锛坄_lookup_comm`锛夛細浠?`args[0]` 璁＄畻 `message_bytes`锛屼粠 `rank_group`锛堜綅缃洜绠楀瓙鑰屽紓锛岃涓嬭〃锛夋帹瀵?`topology_tier = comm_grid._get_topology_idx_for_group(rank_group)`锛屽湪 CSV 涓簿纭尮閰?`(num_devices, topology_tier)`銆?

| Op | rank_group 浣嶇疆 |
|----|----------------|
| all_reduce | args[2] |
| all_gather / reduce_scatter | args[3] |
| all_to_all | args[4] |

**FusedAttention 鍖归厤**锛坄_lookup_attention`锛夛細浠?`args[6]`锛坰eq_lens锛夎绠?`batch_size` 鍜?`avg_seq_len`锛屽湪 FusedAttention CSV 涓寜 `(batch_size, avg_seq_len)` 鏌ヨ銆?

**Composite 鏌ヨ**锛坄_lookup_composite`锛夛細閽堝 1:N 鏄犲皠锛堝 MLA锛夛紝ProfilingDataSource 涓诲姩鍒嗚В涓哄涓瓙鍐呮牳鏌ヨ骞舵眰鍜屻€傛祦绋嬶細

1. 鏍规嵁 `func_name` 鏌ユ壘娉ㄥ唽鐨勫垎瑙ｅ嚱鏁帮紙`COMPOSITE_DECOMPOSERS`锛夛紱鏈敞鍐屽垯 return None fallback
2. 鍒嗚В鍑芥暟浠?`OpInvokeInfo.args` 鎺ㄥ姣忎釜瀛愬唴鏍哥殑 `(kernel_type, shapes, dtype)`锛堝尯鍒?prefill/decode锛?
3. 閫愪釜瀛愬唴鏍歌皟鐢?`_lookup_compute_by_shapes()` 鏌ヨ CSV
4. 浠讳竴瀛愬唴鏍告湭鍛戒腑 鈫?鏁翠綋 return None锛涘叏閮ㄥ懡涓?鈫?姹傚拰 Duration 杩斿洖

**MLA 鍒嗚В鍑芥暟**锛氬鐢?`performance_model/__init__.py` 宸叉湁鐨?shape 鎺ㄥ閫昏緫銆侻LA 鎸?prefill/decode 鍒嗚В涓轰笉鍚屽瓙鍐呮牳锛?

| 闃舵 | 瀛愬唴鏍?| Shape 鎺ㄥ |
|------|--------|-----------|
| Prefill | 1脳 TransposeBatchMatMul | `(num_tokens, kv_lora_rank) @ kv_b_proj` |
| Prefill | 1脳 FusedInferAttentionScore | `(num_tokens, num_heads, qk_head_dim)` + decompressed KV |
| Decode | 1脳 TransposeBatchMatMul (Q@W_UK_T) | `(num_tokens, num_heads, qk_nope_head_dim) @ W_UK_T` |
| Decode | 1脳 FusedInferAttentionScore | compressed attention via latent space |
| Decode | 1脳 TransposeBatchMatMul (AV@W_UV) | `(num_tokens, num_heads, kv_lora_rank) @ W_UV` |

缁村害浠?`OpInvokeInfo.args` 鎻愬彇锛歚num_heads = q.size(1)`, `kv_lora_rank = W_UK_T.size(-1)`, `qk_rope_head_dim = kv_cache.size(-1) - kv_lora_rank` 绛夈€侾refill/Decode 閫氳繃 `query_lens` 闃堝€煎垽瀹氥€?

> 闀挎湡鏂规锛歁LA decomposition pass 瀹屾垚鍚庯紙瑙?9.1 鑺傦級锛屾瘡涓瓙 op 鐙珛鐢熸垚 OpInvokeInfo锛岃蛋鏅€?`_lookup_compute` 璺緞锛宑omposite 閫昏緫鑷劧搴熷純銆?

**鍏抽敭杈呭姪鍑芥暟**锛?

```python
def fractal_nz_to_nd(nz_shape):
    """[..., H, W, block_h, block_w] 鈫?[..., H*block_w, W*block_h]
    268 琛岄浂渚嬪楠岃瘉銆傛仮澶嶅悗涓?aten.mm args[1] 涓€鑷达紝鏃犻渶杞疆銆?""
    *batch, H, W, block_h, block_w = nz_shape
    return (*batch, H * block_w, W * block_h)

DTYPE_MAP = {  # torch dtype 鈫?Profiling dtype string
    torch.bfloat16: "DT_BF16", torch.float16: "DT_BF16",
    torch.int8: "INT8", torch.int32: "INT32", torch.int64: "INT64",
    torch.float32: "FLOAT", torch.bool: "BOOL",
}
```

> FRACTAL_NZ 甯冨眬璇︾粏鍒嗘瀽鍜岄獙璇佹暟鎹闄勫綍 B銆?

### 4.3 EmpiricalPerformanceModel

閲嶆瀯鍚庣殑缁熶竴鍏ュ彛锛氭帴鍙?`DataSource` 瀹炰緥锛宍process_op()` 鍏堟煡鏁版嵁婧愶紝鏈懡涓洖閫€鑷?`fallback_model`锛堥粯璁?`AnalyticPerformanceModel`锛夈€?

```python
# tensor_cast/performance_model/empirical.py
class EmpiricalPerformanceModel(PerformanceModel):
    def __init__(self, device_profile, data_source: DataSource,
                 fallback_model: Optional[PerformanceModel] = None):
        self.data_source = data_source
        self.fallback_model = fallback_model or AnalyticPerformanceModel(device_profile)

    def process_op(self, op_invoke_info):
        result = self.data_source.lookup(op_invoke_info)
        if result is not None:
            return Result(execution_time_s=result.latency_us * 1e-6, ...)
        return self.fallback_model.process_op(op_invoke_info)
```

**浣跨敤绀轰緥**锛?

```python
# Profiling 鏁版嵁搴撻┍鍔?
pm = EmpiricalPerformanceModel(device_profile,
    data_source=ProfilingDataSource("data/.../vllm_ascend/v0.13.0/", device_profile))

# Profiling + 鎻掑€?
pm = EmpiricalPerformanceModel(device_profile,
    data_source=InterpolatingDataSource(ProfilingDataSource("data/...", device_profile)))
```

### 4.4 InterpolatingDataSource

Wrapper 妯″紡锛氱簿纭尮閰嶅鎵樼粰 `base_source`锛屾湭鍛戒腑鏃?`find_neighbors()` 鏌ユ壘杩戦偦鏁版嵁鐐癸紝鐒跺悗鎻掑€?澶栨帹銆?

**閫氱敤鎻掑€肩瓥鐣?*锛堜笉闇€瑕?per-operator 缁村害澹版槑锛夛細
- **dtype + format 绮剧‘鍖归厤**锛氬凡鍦?`ProfilingDataSource._inputs_match()` 涓疄鐜帮紝鎵€鏈夌畻瀛愰€氱敤
- **shape 缁村害绾挎€ф彃鍊?*锛氭湭绮剧‘鍛戒腑鏃讹紝鍦?CSV 涓壘 dtype/format 绮剧‘鍖归厤鐨勮锛屽 shape 缁村害鍋氭渶杩戦偦鎼滅储 + 绾挎€ф彃鍊?
- **鐗规畩鍙樻崲**锛氫粎 `FusedInferAttentionScore` 闇€瑕?sqrt 鍙樻崲锛圓ttention 寤惰繜 鈭?seq虏锛屽湪 鈭歴eq 绌洪棿鎻掑€肩簿搴︽洿楂橈級

姝よ璁″熀浜庝互涓嬭瀵燂細95% 鐨勭畻瀛愶紙GEMM銆丯orm銆丒lementwise銆丮oE 绛夛級鍏变韩鐩稿悓鐨勬彃鍊奸€昏緫鈥斺€攄type 绮剧‘鍖归厤 + shape 绾挎€ф彃鍊笺€傚彧鏈?FIA 鍥?O(n虏) 澶嶆潅搴﹂渶瑕侀澶栧鐞嗐€傚弬鑰?AI Configurator 鐨勫疄鐜帮紝鍏?12 涓?`query_*` 鏂规硶涓彧鏈?Attention 绯诲垪浣跨敤浜?sqrt 鍙樻崲锛屽叾浣欏潎涓烘爣鍑嗙嚎鎬?绔嬫柟鎻掑€笺€?

`op_mapping.yaml` 鐨?`interpolation_policy` 澹版槑榛樿绛栫暐鍜屽皯閲?kernel_type override銆?

### 4.5 op_mapping.yaml 瑙勬牸

`op_mapping.yaml` 鏄函鍚嶅瓧鏄犲皠閰嶇疆锛屼笉鍚?per-op 缁村害鎻愬彇閫昏緫銆傚畬鏁寸ず渚嬭 [`docs/examples/op_mapping_example.yaml`](examples/op_mapping_example.yaml)銆?

**`op_mapping.yaml` 椤跺眰缁撴瀯**锛?

```yaml
version: "0.15.0"                              # vLLM-Ascend 鐗堟湰
device: ATLAS_800_A3_752T_128G_DIE
cann_version: "8.5"
communication_data_ref: "../../hccl/v8.5/"     # 閫氫俊鏁版嵁鐩稿璺緞
communication_fallback: analytic                # 閫氫俊鏁版嵁涓嶅瓨鍦ㄦ椂 fallback

interpolation_policy:
  default_method: linear                       # 鎵€鏈夌畻瀛愰粯璁わ細dtype+format 绮剧‘鍖归厤锛宻hape 缁村害绾挎€ф彃鍊?
  kernel_overrides:                            # 浠呭垪鍑洪渶瑕佺壒娈婂鐞嗙殑 kernel_type
    FusedInferAttentionScore:
      shape_transform: sqrt                    # O(seq虏) 鈫?鍦?鈭歴eq 绌洪棿鎻掑€?

operator_mappings:
  "aten.mm.default":                           # 鏍囧噯 aten 绠楀瓙
    kernel_type: MatMulV2
  "tensor_cast.static_quant_linear.default":   # 閲忓寲 Matmul
    kernel_type: QuantBatchMatmulV3
  "tensor_cast.attention.default":             # 鐗规畩鏌ヨ妯″紡
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
  "tensor_cast.all_reduce.default":            # 閫氫俊绠楀瓙锛堝紩鐢?hccl 鏁版嵁搴擄級
    kernel_type: hcom_allReduce_
    category: communication
  "tensor_cast.multihead_latent_attention.default":  # 澶嶅悎鏄犲皠锛?:N锛?
    composite: true
    sub_kernels: [TransposeBatchMatMul, FusedInferAttentionScore]
  ...                                          # 瀹屾暣鍒楄〃绾?25 鏉?

torch_npu_reference:                           # microbenchmark 鑴氭湰鐢熸垚鐢?
  MatMulV2:
    apis: [{name: "torch.mm"}, {name: "torch_npu.npu_linear"}]
    microbench_api: "torch.mm"
  QuantBatchMatmulV3:
    apis: [{name: "torch_npu.npu_weight_quant_batchmatmul"}]
    microbench_api: "torch_npu.npu_weight_quant_batchmatmul"
  ...                                          # 瀹屾暣鍒楄〃绾?11 鏉?
```

> 瀹屾暣绀轰緥瑙?[`docs/examples/op_mapping_example.yaml`](examples/op_mapping_example.yaml)銆?

**`comm_config.yaml` 椤跺眰缁撴瀯**锛堜綅浜?`hccl/{cann_version}/` 鐩綍锛夛細

```yaml
device: ATLAS_800_A3_752T_128G_DIE
cann_version: "8.5"

topology:
  grid_shape: [48, 8, 2]                       # 涓夌淮鎷撴墤缃戞牸
  tiers:
    0: {name: "inter_pod", bandwidth_gbps: 196, latency_us: 5.5, type: "CLOS"}
    1: {name: "intra_pod", bandwidth_gbps: 196, latency_us: 0.5, type: "CLOS"}
    2: {name: "die_level", bandwidth_gbps: 224, latency_us: 0.2, type: "SIO"}

comm_operator_mappings:
  "tensor_cast.all_reduce.default": hcom_allReduce_
  "tensor_cast.all_gather.default": HcomAllGather
  ...
```

> 瀹屾暣绀轰緥瑙?[`docs/examples/comm_config_example.yaml`](examples/comm_config_example.yaml)銆?

**`operator_mappings` 姣忔潯璁板綍鏀寔鐨勫瓧娈?*锛?

| 瀛楁 | 璇存槑 | 绀轰緥 |
|-----|------|------|
| `kernel_type` | Profiling Type 鍒楀悕 | `MatMulV2` |
| `category` | 绠楀瓙绫诲埆锛堥┍鍔ㄦ煡璇㈠垎娲撅級 | `communication` |
| `query_mode` | 鐗规畩鏌ヨ妯″紡 | `attention_special` |
| `composite` | 澶嶅悎鏄犲皠锛?:N锛?| `true` |
| `sub_kernels` | 澶嶅悎鏄犲皠瀛愬唴鏍?| `[TransposeBatchMatMul, ...]` |
| `notes` | 鏂囨。璇存槑 | |

### 4.6 鏌ヨ绀轰緥

**MatMulV2 (BF16, FRACTAL_NZ weight)**锛?
1. Runtime 鎷︽埅 `aten.mm.default(A[136,5120], B[5120,768])`
2. op_mapping.yaml: `aten.mm.default` 鈫?`MatMulV2`
3. 鍔犺浇 MatMulV2.csv, 閫愯鍖归厤:
   - CSV: Input Shapes=`"136,5120;320,48,16,16"`, Formats=`"ND;FRACTAL_NZ"`
   - input[0]: ND (136,5120) 鈫?match 鉁?
   - input[1]: FRACTAL_NZ (320,48,16,16) 鈫?`fractal_nz_to_nd` 鈫?(5120, 768) 鈫?match 鉁?
   - dtype: DT_BF16 鈫?torch.bfloat16 鈫?match 鉁?
4. Duration=45.3渭s 鈫?`QueryResult(latency_us=45.3)`

### 4.7 閫氫俊绠楀瓙鏁版嵁鏍煎紡

閫氫俊鏁版嵁鏉ユ簮鏄?HCCL Test / `torch.distributed` microbenchmark锛堥潪鍏ㄦā鍨?Profiling锛夈€?

**閫氫俊 CSV 鏍煎紡**锛?

```csv
message_bytes,num_devices,dtype,topology_tier,Duration(us),bandwidth_gbps
602112,8,DT_BF16,2,125.3,4.6
602112,4,DT_BF16,2,87.2,6.6
1204224,16,DT_BF16,0,342.1,1.7
```

**topology_tier 鍚箟**锛圓TLAS_800_A3 [48,8,2] 涓夌淮缃戞牸锛夛細tier 2 = die 绾?SIO锛?24 GB/s锛夛紝tier 1 = pod 鍐?CLOS锛?96 GB/s锛夛紝tier 0 = pod 闂?CLOS锛?96 GB/s, 5.5碌s latency锛夈€?

**HCCL v8.5 鏁版嵁璐ㄩ噺娉ㄦ剰浜嬮」**锛坴1.3.1锛孋10 閲囬泦缁忛獙锛夛細
- 鍒濈増鑴氭湰姣忎釜 op 鐙珛 `torchrun` 瀵艰嚧 HCCL JIT 閲嶆柊鍒濆鍖栵紝灏忔秷鎭紙4KB/16KB锛夊欢杩熷紓甯稿亸楂?
- 淇鏂规锛氬崟 session 杩愯鎵€鏈?op + 鍏ㄥ眬棰勭儹锛堟瘡涓?op/group 鍏堣窇 1KB 瑙﹀彂 JIT 缂栬瘧锛? `WARMUP_ITERS=20`
- tier=2锛坉ie_level锛? 鍗★級鏁版嵁宸查噰闆嗭紱tier=1锛坕ntra_pod锛?6 鍗★級鏁版嵁宸查噰闆?
- tier=0锛坕nter_pod锛夐渶澶氳妭鐐癸紙>16 鍗★級鐜锛屾殏 fallback analytic

**閫氫俊绠楀瓙 OpInvokeInfo args 瀹屾暣甯冨眬**锛?

| Op | args[0] | args[1] | args[2] | args[3] | args[4] |
|----|---------|---------|---------|---------|---------|
| all_reduce | Tensor x | int rank | List rank_group | | |
| all_gather | Tensor x | int dim | int rank | List rank_group | |
| reduce_scatter | Tensor x | int dim | int rank | List rank_group | |
| all_to_all | Tensor x | List out_splits | List in_splits | int rank | List rank_group |

### 4.8 FusedAttention 鐗规畩澶勭悊

Profiling 涓?`FusedInferAttentionScore` 鐨?KV Input Shapes 鏄鍒嗛厤 buffer shape锛屼笉鏄疄闄?KV 闀垮害锛屾棤娉曠洿鎺ュ尮閰嶃€?

**涓轰粈涔堢敤 `avg_seq_len` 鑰岄潪閫?request 鐨?`actual_seq_lengths_kv`**锛氫竴娆?FusedAttention kernel 璋冪敤澶勭悊鏁翠釜 batch锛屾瘡涓?request 鐨?KV 闀垮害涓嶅悓锛坄actual_seq_lengths_kv = [seq_len_0, seq_len_1, ...]`锛夛紝浣?kernel 鍙骇鐢熶竴涓?`Duration(us)`銆傛棤娉曟寜鍗曚釜 request 鎷嗗垎鑰楁椂锛屽彧鑳界敤 batch 绾у埆鐨勭粺璁￠噺浣滀负绱㈠紩銆俙avg_seq_len = mean(actual_seq_lengths_kv)` 鏄渶绠€鍗曠殑鑱氬悎鏂瑰紡锛屽湪 batch 鍐?seq_len 鍒嗗竷杈冨潎鍖€鏃惰冻澶熷噯纭€?

**褰撳墠鏂规**锛氶€氳繃 microbenchmark 鏋勫缓鐙珛鏁版嵁锛屼互 `(batch_size, avg_seq_len, num_heads, head_dim, dtype)` 涓虹储寮曘€侻icrobenchmark 鏃堕€氳繃 `actual_seq_lengths_kv`锛? `OpInvokeInfo.args[6]`锛夎缃瘡涓?request 鐩稿悓鐨?KV 闀垮害鏉ユ帶鍒?avg_seq_len銆傚湪 `op_mapping.yaml` 涓爣璁?`query_mode: attention_special`銆?

**FusedAttention CSV 鏍煎紡**锛坢icrobenchmark 鏋勫缓锛夛細

```csv
batch_size,avg_seq_len,num_heads,head_dim,dtype,Duration(us)
1,4096,64,128,DT_BF16,1250.3
16,512,64,128,DT_BF16,890.7
```

> 璇︾粏鍒嗘瀽瑙侀檮褰?C锛涢暱鏈熸柟妗堣绗?10.2 鑺傘€?

### 4.9 FRACTAL_NZ 閫氱敤澶勭悊

鎭㈠鍏紡 `[..., H, W, block_h, block_w] 鈫?[..., H*block_w, W*block_h]` 瑙?4.2 鑺傘€俆ile 澶у皬浠?CSV shape 鏈€鍚庝袱缁寸洿鎺ヨ鍙栵紙BF16: 16脳16, INT8: 16脳32锛夈€傛仮澶嶅悗涓?`aten.mm args[1]` 涓€鑷淬€?

> 璇︾粏鐨勫竷灞€鍒嗘瀽銆佹簮鐮佹函婧愬拰楠岃瘉鏁版嵁瑙侀檮褰?B銆?

---

## 5. TensorCast 闆嗘垚鎺ュ彛

### 5.1 Runtime 闆嗘垚

`Runtime` 绫绘棤闇€淇敼銆傚畠宸叉敮鎸佹帴鏀?`PerformanceModel` 瀹炰緥鍒楄〃锛屽苟鑷姩鍖呰涓?`CachingPerformanceModel`銆?

```python
# 浣跨敤 ProfilingDataSource
data_source = ProfilingDataSource(
    "perf_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/v0.13.0",
    device_profile,
)
perf_model = EmpiricalPerformanceModel(device_profile, data_source)
runtime = Runtime(perf_models=perf_model, device_profile=device_profile)
```

### 5.2 CLI 鎺ュ彛

鍦?`cli/inference/text_generate.py` 涓敮鎸佸鎬ц兘妯″瀷骞惰锛?

```python
parser.add_argument("--performance-model",
                    action="append",
                    default=None,
                    help="鎬ц兘妯″瀷绫诲瀷锛屽彲澶氭鎸囧畾銆?
                         "'analytic': Roofline 妯″瀷锛堥粯璁わ紝鏃犻渶鏁版嵁锛夈€?
                         "'profiling': 鍩轰簬瀹炴祴 Profiling 鏁版嵁搴撶殑 EmpiricalPerformanceModel "
                         "锛堥渶瑕?--profiling-database锛夈€?)
parser.add_argument("--profiling-database", type=str, default=None,
                    help="鎬ц兘鏁版嵁搴撹矾寰勶紙profiling 妯″紡鐢熸晥锛夛紝"
                         "鎸囧悜鍖呭惈 op_mapping.yaml 鍜?CSV 鏁版嵁鏂囦欢鐨勭洰褰?)

# 榛樿鍊煎鐞?
if args.performance_model is None:
    args.performance_model = ["analytic"]
```

**浣跨敤绀轰緥**锛?

```bash
# 鍗曚釜妯″瀷锛堥粯璁わ級
--performance-model analytic

# 澶氫釜妯″瀷骞惰杩愯
--performance-model analytic --performance-model profiling --profiling-database /path/to/db
```

| CLI 閫夐」 | 鍒涘缓鐨勬ā鍨?| 鏄惁闇€瑕佺墿鐞嗚澶?| 鏄惁闇€瑕佹暟鎹簱 |
|---------|----------|----------------|---------------|
| `--performance-model analytic` | `AnalyticPerformanceModel` | 鍚?| 鍚?|
| `--performance-model profiling` | `EmpiricalPerformanceModel(ProfilingDataSource(...))` | 鍚?| 鏄紙`--profiling-database`锛?|
| 澶氭鎸囧畾 | 澶氫釜妯″瀷骞惰杩愯锛岃緭鍑哄浠界粨鏋?| 鍚?| 鎸夐渶 |

### 5.3 UserInputConfig 閰嶇疆

`UserInputConfig.performance_model` 鏀寔 `Union[str, List[str]]`锛?

```python
@dataclass
class UserInputConfig:
    performance_model: Union[str, List[str]] = "analytic"
    """鎬ц兘妯″瀷绫诲瀷锛?analytic' | 'profiling'銆?
    鍙互鏄崟涓瓧绗︿覆鎴栧瓧绗︿覆鍒楄〃浠ヨ繍琛屽涓ā鍨嬨€?""

    def _normalize_performance_model(self):
        """灏?performance_model 瑙勮寖鍖栦负瀛楃涓插垪琛ㄣ€?""
        pm = self.performance_model
        if isinstance(pm, str):
            self.performance_model = [pm]
```

### 5.4 ModelRunner 澶氭ā鍨嬫敮鎸?

`ModelRunner.__init__` 鏋勫缓澶氫釜 `PerformanceModel` 瀹炰緥锛?

```python
class ModelRunner:
    def __init__(self, user_input: UserInputConfig):
        perf_model_types: List[str] = user_input.performance_model
        self.perf_models: List[PerformanceModel] = []
        
        for perf_model_type in perf_model_types:
            if perf_model_type == "profiling":
                data_source = ProfilingDataSource(
                    user_input.profiling_database,
                    self.device_profile,
                )
                self.perf_models.append(
                    EmpiricalPerformanceModel(
                        self.device_profile,
                        data_source=data_source,
                        fallback_model=AnalyticPerformanceModel(self.device_profile),
                    )
                )
            elif perf_model_type == "analytic":
                self.perf_models.append(AnalyticPerformanceModel(self.device_profile))
```

### 5.5 ModelRunnerMetrics 杈撳嚭

`ModelRunnerMetrics` 瀛樺偍姣忎釜妯″瀷鐨勬墽琛屾椂闂村拰 TPS锛?

```python
@dataclass
class ModelRunnerMetrics:
    single_card_tps: float  # 绗竴涓ā鍨嬬殑 TPS锛堝悜鍚庡吋瀹癸級
    execution_time_s: Dict[str, float]  # 姣忎釜妯″瀷鐨勬墽琛屾椂闂达紝鎸夋ā鍨嬪悕绱㈠紩
    tps_per_model: Dict[str, float]  # 姣忎釜妯″瀷鐨?TPS锛屾寜妯″瀷鍚嶇储寮?
    # ... 鍏朵粬瀛楁

    def print_info(self):
        for model_name, exec_time in self.execution_time_s.items():
            print(f"[{model_name}] Execution time: {exec_time:.6f} s")
            tps = self.tps_per_model.get(model_name)
            if tps is not None:
                print(f"[{model_name}] TPS/Device: {tps:.4g} token/s")
```

### 5.6 鏁版嵁娴?

`--performance-model profiling` 鈫?鍒涘缓 `EmpiricalPerformanceModel(ProfilingDataSource(data_dir))` 鈫?Runtime 鎷︽埅绠楀瓙鐢熸垚 `OpInvokeInfo` 鈫?`data_source.lookup()` 鏌ヨ锛堝垎娲鹃€昏緫瑙?4.2 鑺傦級鈫?鍛戒腑杩斿洖瀹炴祴鑰楁椂锛屾湭鍛戒腑 fallback 鑷?Roofline/CommAnalytic銆?

---

## 6. 鏁版嵁搴撴瀯寤虹瓥鐣?

### 6.1 涓夋璧扮瓥鐣?

| 姝ラ | 鍐呭 | 杈撳嚭 |
|-----|------|------|
| **姝ラ涓€**锛氬崟娆?vLLM Profiling | 鎷夎捣涓€娆?vLLM 瀹炰緥锛岄噰闆?kernel_details.csv | 鎸?Type 鍒楁媶鍒嗙殑 `{KernelType}.csv` + `op_mapping.yaml` 鍒濈増 |
| **姝ラ浜?*锛歁icrobenchmark 缃戞牸閬嶅巻 | 閽堝姣忎釜绠楀瓙锛岄€氳繃 `torch_npu` 鍜?`torch.distributed` API 閬嶅巻 shape 缃戞牸锛岀敤 `msprof` 閲囬泦 device 渚ф暟鎹?| 鎵╁厖鍚庣殑 CSV 鏁版嵁搴?|
| **姝ラ涓?*锛氾紙鍙€夛級绔埌绔獙璇?| 璺戝嚑娆″吀鍨?vLLM 閰嶇疆锛屽姣?microbenchmark 棰勬祴 vs Profiling 瀹炴祴 | 绮惧害鎶ュ憡 + 涓嶇ǔ瀹氱畻瀛愭爣璁?|

**姝ラ涓€璇︾粏娴佺▼**锛?
1. 鎷夎捣 vLLM 瀹炰緥锛岄噰闆?kernel_details.csv锛堝彧闇€鑳借窇璧锋潵鍗冲彲锛孍P=1 閰嶇疆锛?
2. `parse_kernel_details.py` 鈫?鎸?Type 鍒楁媶鍒嗕负鍚?`{KernelType}.csv`
3. 鍚屾椂鍩轰簬鏄犲皠琛?+ 杩愯鏃剁幆澧冧俊鎭敓鎴?`op_mapping.yaml` 鍒濈増

> 娉細EP=1 鏃?Profiling 涓笉浼氬嚭鐜?AllToAll 閫氫俊锛孉llToAll 鏁版嵁閫氳繃姝ラ浜岀殑閫氫俊 microbenchmark 鐙珛閲囬泦銆?

**姝ラ浜岃缁嗘祦绋?*锛?

璁＄畻绠楀瓙锛?
1. `generate_shape_grid.py` 鈫?鏍规嵁妯″瀷鍙傛暟鐢熸垚 shape 閬嶅巻鑼冨洿
2. `microbenchmark generation tool` 鈫?璇?`op_mapping.yaml` 鐨?`torch_npu_reference`锛岀敓鎴?Python 鑴氭湰
3. 鐢?`msprof` 閲囬泦 device 渚ф暟鎹紙Duration, aic_time, aiv_time, cube_utilization 绛夛級
4. 鐗规畩: FusedAttention 鐢?`actual_seq_lengths_kv` 鎺у埗瀹為檯 KV 闀垮害

閫氫俊绠楀瓙锛?
1. `generate_comm_microbench.py` 鈫?鐢熸垚 `torch.distributed` 鑴氭湰
2. 鎺у埗 `rank_group` 娴嬭瘯鍚?`topology_tier`
3. 鏁版嵁鍐欏叆 `hccl/{cann_version}/` 鐩綍
4. HCCL Test 浣滀负浜ゅ弶楠岃瘉

### 6.2 璁＄畻绠楀瓙 Microbenchmark

閫氳繃 `torch_npu` 鐩存帴瀵瑰崟涓畻瀛愯繘琛岀粏绮掑害 Shape 瑕嗙洊娴嬭瘯銆俙op_mapping.yaml` 鐨?`torch_npu_reference` 娈垫彁渚涗簡姣忎釜绠楀瓙瀵瑰簲鐨?`microbench_api`锛屼緵鑴氭湰鐢熸垚宸ュ叿鑷姩浣跨敤銆?

**Shape 缃戞牸鐢熸垚绛栫暐**锛歚generate_shape_grid.py` 鍐呴儴鎸?kernel_type 鍒嗘淳鐢熸垚閫昏緫锛堝伐鍏蜂晶鐭ヨ瘑锛屼笉鍦?op_mapping.yaml 涓厤缃級锛?
- **GEMM 绫?*锛圡atMulV2, QuantBatchMatmulV3, GroupedMatmul 绛夛級锛氫粠妯″瀷閰嶇疆鎻愬彇 N/K锛坔idden_size, intermediate_size 绛夛級锛孧 鐢?powers-of-2 缃戞牸閬嶅巻
- **Attention**锛團usedInferAttentionScore锛夛細浠庢ā鍨嬮厤缃彁鍙?num_heads/head_dim锛宐atch 脳 seq_len 鐢ㄧ綉鏍奸亶鍘?
- **Elementwise 绫?*锛圓dd, RmsNorm, SwiGlu 绛夛級锛氫粠妯″瀷閰嶇疆鎺ㄥ tensor 澶у皬锛宯um_tokens 鐢ㄧ綉鏍奸亶鍘?
- **閫氫俊绠楀瓙**锛歮essage_size 鐢?powers-of-2 缃戞牸閬嶅巻

```python
# tools/perf_data_collection/microbenchmark generation tool
# 璇诲彇 op_mapping.yaml 涓殑 torch_npu_reference锛?
# 涓烘瘡涓畻瀛愮敓鎴愰亶鍘嗚剼鏈紝鐢?msprof 閲囬泦 device 渚ф暟鎹?
```

> op-plugin 搴擄紙https://github.com/Ascend/op-plugin锛夊彲鐢ㄤ簬鍙嶆煡 Profiling Type 鈫?torch_npu API锛岃缁嗗垎鏋愯闄勫綍 E銆?

### 6.3 閫氫俊绠楀瓙 Microbenchmark

**缁熶竴鏂规**锛氳绠楀拰閫氫俊閮界敤 Python 鑴氭湰 + `msprof`锛孒CCL Test 浣滀负浜ゅ弶楠岃瘉銆?

```python
# tools/perf_data_collection/generate_comm_microbench.py
# 鍒濆鍖?HCCL backend锛岄亶鍘?message_size 缃戞牸锛?
# 鐢?torch.npu.Event 璁℃椂 warmup + N 娆￠噸澶嶇殑 dist.all_reduce/all_gather/all_to_all锛?
# 閫氳繃 rank_group 鎺у埗娴嬭瘯鍚?topology_tier
```

**MC2 microbenchmark**锛歁C2锛坄npu_mm_all_reduce_base`锛夊皢 MatMul+AllReduce 铻嶅悎锛屾棤娉曞垎寮€娴嬭瘯銆傜敤 Python 鑴氭湰鐩存帴璋冪敤 `torch_npu.npu_mm_all_reduce_base` 閬嶅巻涓嶅悓 M/N/K + num_devices銆侻C2 鏁版嵁瀛樺偍鍦?`vllm_ascend/{version}/` 涓嬶紙鍜屾鏋跺疄鐜扮粦瀹氾級銆?

**HCCL Test 宸ュ叿**锛堜氦鍙夐獙璇佺敤锛夛細
```bash
# 浣嶄簬 CANN toolkit: /usr/local/Ascend/ascend-toolkit/latest/tools/hccl_test/
mpirun -n 8 ./bin/all_reduce_test -b 8K -e 2048M -f 2 -d fp16 -o sum -p 8
mpirun -n 8 ./bin/all_gather_test -b 8K -e 512M -f 2 -d fp16 -p 8
mpirun -n 8 ./bin/all_to_all_test -b 8K -e 256M -f 2 -d fp16 -p 8
```

> 鍙傝€冩枃妗ｏ細https://www.hiascend.com/document/detail/zh/mindstudio/70RC1/mscommandtoolug/mscommandug/auxiliarydevtool_0017.html

### 6.4 FusedAttention Microbenchmark

FusedAttention 闇€鏋勯€犲悎娉曠殑 paged KV cache 杈撳叆锛氶鍒嗛厤 KV buffer + `block_table` + `actual_seq_lengths_kv` 鎺у埗瀹為檯 KV 闀垮害銆傞亶鍘?`(batch_size, seq_len, num_heads, head_dim)` 缁勫悎锛岃皟鐢?`torch_npu.npu_fused_infer_attention_score`銆傝瑙侀檮褰?C 鐨?args 甯冨眬銆?

### 6.5 Profiling 杈撳嚭瑙ｆ瀽鍣?

```python
# tools/perf_data_collection/parse_kernel_details.py

class KernelDetailsParser:
    def parse(self, csv_path: Path) -> pd.DataFrame:
        """瑙ｆ瀽 kernel_details.csv 鈫?缁撴瀯鍖?DataFrame銆?
        鎸?Type 鍒楀垎缁勶紝淇濈暀 Profiling 鍘熷鍒椼€?""

    def split_by_type(self, df: pd.DataFrame, output_dir: Path):
        """鎸?Type 鍒楁媶鍒嗕负鍚?{KernelType}.csv銆?
        CSV 淇濇寔 Profiling 鍘熷鏍煎紡锛?
        Input Shapes, Input Data Types, Input Formats,
        Output Shapes, Output Data Types, Output Formats,
        Duration(us), Accelerator Core, Block Dim,
        aicore_time(us), aic_mac_time(us), aic_mac_ratio,
        aic_scalar_time(us), aiv_time(us), aiv_vec_time(us),
        aiv_scalar_time(us), cube_utilization(%)
        """
```

### 6.6 鏂版ā鍨嬬畻瀛愬彂鐜?

```python
# tools/perf_data_collection/operator coverage check tool

def discover_operators(profiling_output: Path, op_mapping_yaml: Path) -> Dict:
    """鍙戠幇 Profiling 涓瓨鍦ㄤ絾褰撳墠 op_mapping.yaml 涓己澶辩殑绠楀瓙銆?
    杩斿洖 known 鍜?unknown 璁℃暟銆?""
```

---

## 7. 鍏ㄩ潰绠楀瓙瑕嗙洊绛栫暐

### 7.1 绠楀瓙鍒嗙骇

| 灞傜骇 | 鍒ゅ畾鏍囧噯 | 澶勭悊鏂瑰紡 |
|-----|---------|---------|
| **Tier 1** | 鎵ц鑰楁椂鍗犳瘮 >2%锛堜换涓€妯″瀷锛?| 瀹屾暣 Shape 缃戞牸 Microbenchmark |
| **Tier 2** | 鎵ц鑰楁椂鍗犳瘮 0.5-2% | 绮剧畝 Shape 缃戞牸鎴?Profiling 鐩存帴瀵煎叆 |
| **Tier 3** | 鎵ц鑰楁椂鍗犳瘮 <0.5% | Roofline 鍏滃簳 |

### 7.2 Tier 1 瀹屾暣绠楀瓙鍒楄〃

> 鏁版嵁鏉ユ簮锛氬叏閲忓唴鏍歌€楁椂锛堝惈閫氫俊锛夛紝Qwen3-32B Prefill 16 鍗?/ DSV3 Decode 32 鍗°€?

| Profiling Type | Qwen3 鍗犳瘮 | DSV3 鍗犳瘮 |
|---------------|-----------|---------|
| hcom_allReduce\_ | **89.8%** | 8.1% |
| GroupedMatmul | - | **18.7%** |
| QuantBatchMatmulV3 | - | **18.3%** |
| FusedInferAttentionScore | 1.7% | **16.5%** |
| MoeDistributeDispatchV2 | - | **7.0%** |
| MatMulV2 | **4.1%** | 0.5% |
| AscendQuantV2 / DynamicQuant | - | **3.9%** |
| TransposeBatchMatMul | - | **3.8%** |
| MoeDistributeCombineV2 | - | **3.6%** |
| InterleaveRope | - | **2.5%** |
| DequantSwigluQuant / SwiGlu | 0.5% | **2.4%** |
| InplaceAddRmsNorm / AddRmsNorm | 0.8% | 1.8% |
| HcomAllGather | 0.6% | 1.7% |

### 7.3 Shape 缃戞牸绛栫暐

**绗竴姝?*锛氫粠 HuggingFace 妯″瀷閰嶇疆鑷姩鎻愬彇缁村害鍙傛暟锛坔idden_size, num_heads, intermediate_size 绛夛級銆?

**绗簩姝?*锛氫互閫氱敤鐨?2 鐨勫箓娆＄綉鏍艰ˉ鍏呮ā鍨嬮棿鐨勬彃鍊奸棿闅欍€?

**棰勪及 Shape 鎬婚噺**锛氭瘡涓増鏈害 5,000 涓€?

### 7.4 鏃犳硶 Microbenchmark 鐨勭畻瀛愬奖鍝嶅垎鏋?

| 绠楀瓙 | DSV3 Decode 鍗犳瘮 | Qwen3 Prefill 鍗犳瘮 | 鍘熷洜 | 鏂规 |
|-----|-------------|-------------|------|------|
| TensorMove | 0.06% | 1.03% | 瀹屽叏鐢?runtime 璋冨害鍐冲畾 | 蹇界暐锛岀敤 Roofline 鍏滃簳 |
| split_qkv_rmsnorm_rope_kernel | 0% (DSV3 鏃? | 0.50% | 鏃犲叕寮€ torch_npu API | 鍏ㄦā鍨?Profiling 鑾峰彇锛涙垨寰呮柊澧炶瀺鍚?pass |
| ReshapeAndCacheNdKernel | 0% (DSV3 鏃? | 0.32% | 鍚屼笂 | 鍚屼笂 |
| Graph Mode 铻嶅悎绠楀瓙 | - | - | 浠呭湪 ACLGraph 妯″紡涓嬭Е鍙?| 鏃犳硶鐙珛 benchmark |
| **鍚堣** | **0.06%** | **1.85%** | | |

**缁撹**锛氭棤娉?microbenchmark 鐨勭畻瀛愪粎鍗?0.06%-1.85%锛屽绔埌绔簿搴﹀奖鍝嶆瀬灏忥紙杩滃湪 <15% 鐩爣鍐咃級銆?

---

## 8. 寮€鍙戣鍒?

**鍥㈤槦鍒嗗伐**锛?
- **鍏，宸ュ叿鍥㈤槦**锛歍ensorCast 渚э紙EmpiricalPerformanceModel 閲嶆瀯銆丏ataSource 鎺ュ彛銆丆LI 闆嗘垚銆佽瀺鍚?Pass 琛ラ綈锛?
- **灏忓阀鐏靛洟闃?*锛氭暟鎹晶锛圥rofilingDataSource 瀹炵幇銆丆SV 瑙ｆ瀽銆丮icrobenchmark 鑴氭湰銆佹暟鎹噰闆嗭級

**鍓嶇疆渚濊禆**锛歍ensorCast 绠楀瓙杩借釜瀵归綈锛堣瑙佺 9.1 鑺傦級锛岄渶鍦ㄩ泦鎴愭祴璇曞墠瀹屾垚銆?

**鐩爣**锛?026.3.23 瀹屾垚绔埌绔泦鎴愶紝DeepSeek-V3 / Qwen3-32B 鑳藉榻愮殑绠楀瓙瑕嗙洊绔埌绔?>90% 鐨勬椂闂淬€?

### Phase 1锛氳绠楃畻瀛愮┛鍒猴紙3.4 - 3.14锛?

鑱斿悎瀵归綈 Qwen3-32B 鍜?DeepSeek-V3 鍦ㄧ洰鏍囩増鏈笂鐨勭畻瀛愬垪琛ㄥ拰 op_mapping.yaml銆?

| 鍏，鍥㈤槦 | 灏忓阀鐏靛洟闃?|
|---------|-----------|
| `EmpiricalPerformanceModel` 閲嶆瀯 + DataSource 鎺ュ彛 | `op_mapping.yaml` 瑙勬牸瀹氫箟 + CLI 闆嗘垚 |
| `ProfilingDataSource` 瀹炵幇锛圕SV 鏌ヨ + FRACTAL_NZ锛?| Profiling 瑙ｆ瀽鍣紙`parse_kernel_details.py`锛?|
| 鍩轰簬绀轰緥鏁版嵁搴撶┛鍒虹鍒扮浠跨湡 | 绌垮埡鏁版嵁閲囬泦锛堝崟娆?Profiling + 鍒濆 CSV锛?|

**Phase 1 浜や粯鐗?*锛歅rofiling CSV 瀵煎叆 + ProfilingDataSource + 绔埌绔煡璇㈤獙璇侊紙璁＄畻绠楀瓙锛?

### Phase 2锛氶€氫俊绠楀瓙鎺ュ叆 + 鎻掑€硷紙3.14 - 3.20锛?

| 鍏，鍥㈤槦 | 灏忓阀鐏靛洟闃?|
|---------|-----------|
| `InterpolatingDataSource` 瀹炵幇 | 閫氫俊 microbenchmark 鑴氭湰锛坄generate_comm_microbench.py`锛?|
| 鎵╁睍铻嶅悎 Pass锛圡C2銆並vRmsNormRopeCache 绛夛級 | hccl CSV 鏋勫缓 + `comm_config.yaml` |
| CommGrid 闆嗘垚鍒?ProfilingDataSource | Shape 缃戞牸鎵╁厖锛坄generate_shape_grid.py`锛?|

**Phase 2 浜や粯鐗?*锛氶€氫俊 microbenchmark + hccl CSV + InterpolatingDataSource

### Phase 3锛氶泦鎴愰獙璇侊紙3.20 - 3.23锛?

鑱斿悎浣跨敤 Qwen3-32B 鍜?DeepSeek-V3 杩涜绔埌绔簿搴﹂獙璇併€?

**楠岃瘉鏍囧噯**锛?

| 鎸囨爣 | 鐩爣鍊?|
|-----|-------|
| 绔埌绔€楁椂璇樊 | <15%锛堝姣斿疄闄?vLLM Profiling锛?|
| 鍗曠畻瀛愯宸紙宸插尮閰嶇畻瀛愶級 | <20% |
| 鏃堕棿瑕嗙洊鐜?| >90% |

**Phase 3 浜や粯鐗?*锛氱簿搴﹀姣旀姤鍛?+ 鏂囨。鏀跺熬

---

## 9. 澶栭儴渚濊禆涓庢墿灞曞缓璁?

### 9.1 鍓嶇疆渚濊禆锛歍ensorCast 绠楀瓙杩借釜瀵归綈

鏈柟妗堣姹?TensorCast dispatch trace 涓?vLLM Profiling 鐨勭‖浠跺唴鏍稿垪琛ㄥ湪鍏抽敭绠楀瓙涓?1:1 瀵归綈銆?

**褰撳墠鐘舵€?*锛坴1.3.1 鏇存柊锛屽熀浜?feat/perf-database 鍒嗘敮 + Phase 1 楠岃瘉锛夛細

| Gap 椤?| 鐘舵€?| 璇︽儏 | Profiling 鍗犳瘮 |
|--------|------|------|---------------|
| SwiGlu 铻嶅悎 | **宸插叧闂?* 鉁?| `patterns/swiglu.py` 鏀寔鍙屽簭 mul 妯″紡 | DSV3 2.4% |
| GroupedMatmul+SwiGlu 铻嶅悎 | **宸插叧闂?* 鉁?| `freezing_passes/grouped_matmul_swiglu_pass.py`锛? 绉嶉噺鍖栧彉浣?| DSV3 鍚湪 GroupedMatmul 涓?|
| MC2锛圡atMul+AllReduce锛?| **宸查獙璇?* 鉁?| BF16 + W8A8 dispatch trace 纭铻嶅悎姝ｇ‘锛?f82c2b锛夛紱composite 鍒嗚В鏌ヨ宸插疄鐜?| composite 鍒嗚В |
| KvRmsNormRopeCache | **宸插叧闂?* 鉁?| mlapo op 宸茶鐩栨铻嶅悎锛屾棤闇€鐙珛 pass锛堢┛鍒洪獙璇?3f82c2b锛?| DSV3 0.8% |
| TP Padding bug | **宸蹭慨澶?* 鉁?| 浠庡叏灞€ input padding 绉诲埌 MoE layer-local锛?9cf184锛夛紝淇 DSV3 Decode 8x 楂樹及 | P0 |
| MLA 鍒嗚В | composite 鍏滃簳 | `_lookup_composite()` 鍒嗚В鏌ヨ宸插疄鐜帮紙3a5af02锛夛紱闀挎湡闇€ decomposition pass | DSV3 鐩稿叧 |
| DispatchFFNCombine | **鏂板锛屽紑鏀?* | CANN 8.5 瓒呯骇铻嶅悎锛岃瀺鍚?`all_to_all脳2 + GroupedMatmul脳2 + SwiGlu + MoE routing`銆俆C 鐢?composite 鍒嗚В鍏滃簳锛岄渶瀛愬唴鏍?CSV 鏁版嵁锛圕11锛?| **DSV3 35.3%** |
| split_qkv_rmsnorm_rope | 浠嶅紑鏀?| 鏃犲搴?pass/op | Qwen3 0.5% |
| aten.topk / MoeGatingTopK | 浠嶅紑鏀?| 鏃犲尯鍒嗘満鍒?| DSV3 (2378 娆¤皟鐢? |

> 娉細develop 鍒嗘敮鍙戠幇涓€涓?bug锛歚grouped_matmul_*_swiglu` 绯诲垪 ops 杈撳嚭 shape 涓?(M, N) 浣嗗簲涓?(M, N//2)锛圫wiGlu 灏?gate+up 瀵瑰崐鍒嗗悗杈撳嚭 half width锛夈€?

### 9.2 铻嶅悎绠楀瓙璇箟涓€鑷存€у垎鏋?

澶ч儴鍒?TensorCast op 鈫?Profiling kernel Type 鏄犲皠 shape 涓€鑷达紙濡?MatMul銆丷msNorm銆丼wiGlu銆侀€氫俊绠楀瓙锛夈€傞渶瑕佺壒娈婂鐞嗙殑鏈夛細
- **FusedAttention**锛歍C 鐢?`(num_tokens, hidden_size)`锛孭rofiling 鐢?`(batch, num_heads, q_len, head_dim)`锛岄€氳繃 `attention_special` 妯″紡澶勭悊
- **MLA**锛?:N 鏄犲皠锛堜竴涓?TC op 瀵瑰簲 TransposeBatchMatMul + FIA锛夛紝褰撳墠閫氳繃 `composite: true` + `_lookup_composite()` 鍒嗚В鏌ヨ瑕嗙洊锛坴1.3.1锛夛紝闀挎湡闇€ decomposition pass
- **permute_tokens**锛歍C 鍙惈鏈湴 permute锛孭rofiling 鐨?MoeDistributeDispatchV2 鍚€氫俊锛岄€氫俊鐢?all_to_all 鍒嗗紑璁℃椂

> 瀹屾暣鏄犲皠琛ㄥ拰宸插彂鐜扮殑 bug 瑙侀檮褰?H銆?

### 9.3 CompositePerformanceModel

闀挎湡寤鸿璁捐 `CompositePerformanceModel`锛屾寜浼樺厛绾х粍鍚堝涓?PerformanceModel锛屽疄鐜板彲閰嶇疆鐨勯檷绾х瓥鐣ワ紙`Profiling 鈫?Empirical 鈫?Analytic`锛夈€傚紩鍏ュ悗 `EmpiricalPerformanceModel` 涓嶅啀闇€瑕佸唴閮ㄦ寔鏈?`fallback_model`锛屼笁绉嶆ā鍨嬪畬鍏ㄨВ鑰︺€?

### 9.4 鍏朵粬鎵╁睍寤鸿

- **璺ㄧ‖浠舵硾鍖?*锛氭敮鎸佹洿澶?DeviceProfile锛堝 Atlas A2銆丟PU锛夛紝闇€涓烘瘡绉嶇‖浠剁嫭绔嬮噰闆嗘暟鎹?
- **鑷姩鍖?CI**锛氶殢 vLLM-Ascend / CANN 鐗堟湰鍙戝竷鑷姩瑙﹀彂鏁版嵁閲囬泦娴佹按绾?
- **棰勬彃鍊间紭鍖?*锛氬弬鑰?AI Configurator 鐨?`_extrapolate_data_grid`锛屽湪鏁版嵁搴撳姞杞芥椂棰勫～鍏呭父鐢ㄧ綉鏍肩偣
- **SOL 鏁版嵁鏍℃**锛氱敤 Roofline 鐞嗚涓嬬晫鏍℃寮傚父娴嬮噺鍊?

---

## 10. 閬楃暀闂涓?Future Work

### 10.1 vLLM 鑷畾涔夌畻瀛愯鐩?

**闂**锛歷LLM 浣跨敤鐨?NPU pybind 绠楀瓙锛堝 `npu_fused_infer_attention_score`锛変笉璧?PyTorch Dispatch锛孴ensorCast 浠庡師鐞嗕笂鏃犳硶鐩存帴鎹曡幏銆?

**褰撳墠鏂规锛堜腑鏈燂級**锛?
1. **TensorCast 铻嶅悎 Pass 1:1 瀵归綈**锛氬浜庢瘡涓?pybind 绠楀瓙锛孴ensorCast 宸叉湁瀵瑰簲鐨勮嚜瀹氫箟绠楀瓙锛堝 `tensor_cast.attention` 瀵瑰簲 `npu_fused_infer_attention_score`锛夛紝閫氳繃缂栬瘧 Pass 灏?aten 绠楀瓙铻嶅悎涓哄搴旂殑 TensorCast 绠楀瓙
2. **MC2 鏂板铻嶅悎 Pass**锛氬皢 `aten.mm + tensor_cast.all_reduce` 铻嶅悎涓烘柊鐨?`tensor_cast.mm_all_reduce`
3. **鎵嬪伐鏄犲皠琛?*锛氬浜庢棤娉曡嚜鍔ㄥ榻愮殑绠楀瓙锛屽湪 `op_mapping.yaml` 涓墜宸ョ淮鎶ゆ槧灏?

**闀挎湡鏂规锛團uture Work锛?*锛?
1. **璺緞 A锛歷LLM FX Graph 鎶撳彇**锛氱洿鎺ヤ粠 vLLM 杩愯鏃剁敤 `torch.compile` 鎴?`torch.fx.symbolic_trace` 鎶撳彇绠楀瓙鍥撅紝閬垮厤渚濊禆 TensorCast 鐨?dispatch trace
2. **璺緞 B锛氫镜鍏ュ紡 DispatchMode**锛氫镜鍏ュ紡淇敼 vLLM 娣诲姞 DispatchMode锛堝唴閮ㄦ箾鍗㈠洟闃熸柟妗堬紝寰呬氦娴侊級锛屽皢 `torch_npu` 鐨?pybind 璋冪敤涔熺撼鍏?trace

> 瀹屾暣鐨?vLLM 鑷畾涔夌畻瀛愯鐩栧垎鏋愯闄勫綍 D銆?

### 10.2 FusedAttention KV Cache 缁村害

**闂**锛歅rofiling 涓?`FusedInferAttentionScore` 鐨?Input Shapes 鍖呭惈棰勫垎閰?KV Cache buffer shape锛屼笉鏄疄闄?KV 闀垮害锛屼笌 TensorCast 鐞嗚寤烘ā鏃犳硶鐩存帴鍖归厤銆?

**褰撳墠鏂规**锛氶€氳繃 microbenchmark 鏋勫缓鏁版嵁搴擄紝浠?`(batch_size, avg_seq_len, num_heads, head_dim, dtype)` 涓虹储寮曪紝浣跨敤 `actual_seq_lengths_kv` 鍙傛暟鎺у埗瀹為檯 KV 闀垮害銆?

**闀挎湡鏂规锛團uture Work锛?*锛?
1. **璺緞 A**锛氫粠 `torch.compile` FX graph 鎶撳彇 vLLM 绠楀瓙鍥撅紝鐩存帴鑾峰彇甯︾湡瀹炵淮搴︾殑绠楀瓙 trace
2. **璺緞 B**锛氫镜鍏ュ紡淇敼 vLLM 娣诲姞 DispatchMode锛堝唴閮ㄦ箾鍗㈠洟闃熸柟妗堬級锛屼粠 vLLM 瀹炶窇鑾峰彇甯︾湡瀹?seq_lens 鐨?profiling 鏁版嵁

> 璇︾粏鍒嗘瀽瑙侀檮褰?C銆?

### 10.3 閫氫俊绠楀瓙鏁版嵁搴撳畬鍠?

Phase 2 瀹炴柦閫氫俊绠楀瓙 microbenchmark 鏁版嵁搴擄細
- 閫氳繃 `torch.distributed` Python 鑴氭湰绮剧‘鎺у埗 `rank_group` 娴嬭瘯鍚?`topology_tier`
- HCCL Test 宸ュ叿浜ゅ弶楠岃瘉
- MC2 铻嶅悎閫氫俊鍗曠嫭澶勭悊

---

## 11. 渚濊禆椤?

**鏂板**锛歚scipy`锛堟彃鍊硷級銆乣packaging`锛堢増鏈В鏋愶級
**鐜版湁**锛歚pandas`銆乣numpy`銆乣pyyaml`銆乣torch`
**鍙€?*锛歚pyarrow`锛圥arquet 鏍煎紡鏀寔锛屽垵鏈熷彲浠呯敤 CSV锛?

---

## 12. 鍙傝€冭祫鏂?

- [vLLM Ascend GitHub](https://github.com/vllm-project/vllm-ascend)
- [vLLM Ascend Profiling 鎸囧崡](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/service_profiling_guide.html)
- [vLLM CustomOp Replacement Tracker](https://github.com/vllm-project/vllm/issues/32676)
- [torch_npu Operator Inventory](https://github.com/vllm-project/vllm-ascend/issues/1511)
- [vLLM torch_bindings.cpp](https://github.com/vllm-project/vllm/blob/main/csrc/torch_bindings.cpp)
- [op-plugin (torch_npu op mapping)](https://github.com/Ascend/op-plugin)
- [鍗庝负鏄囪吘 Profiler 鏂囨。](https://support.huaweicloud.com/intl/en-us/bestpractice-modelarts/modelarts_llm_infer_5906034.html)
- [HCCL Test 鏂囨。](https://www.hiascend.com/document/detail/zh/mindstudio/70RC1/mscommandtoolug/mscommandug/auxiliarydevtool_0017.html)
- [AI Configurator](https://github.com/ai-dynamo/aiconfigurator)
- [Intel NPU Cost Model](https://github.com/intel/npu-nn-cost-model)
- [msModeling Wiki](https://deepwiki.com/Horacehxw/msModeling)
- MC2 鍙傝€? [vllm-ascend#6092](https://github.com/vllm-project/vllm-ascend/issues/6092), [vllm-ascend#5743](https://github.com/vllm-project/vllm-ascend/issues/5743)

---

## 闄勫綍 A锛氬皬 Batch 鍦烘櫙 Roofline 鍋忓樊鍒嗘瀽

**灏?batch Prefill 鍚屾牱鍙楀奖鍝嶏紝鐢氳嚦鏇撮毦棰勬祴**锛?
- **Decode (bs=1, seq=1)**锛欸EMM shape 涓?M=1锛屾湰璐ㄦ槸鐭╅樀-鍚戦噺涔橈紝鏄庣‘ memory-bound锛孯oofline 鑷冲皯鑳芥纭瘑鍒摱棰堟柟鍚戯紝浣嗕細楂樹及鍙揪甯﹀
- **灏?batch Prefill (bs=1, seq=256-1024)**锛歁=seq_len 澶勪簬 memory-bound 鍒?compute-bound 鐨勮繃娓″尯鍩燂紙Roofline 鏇茬嚎鐨?鎷愮偣"浣嶇疆锛夛紝姝ゆ椂鎬ц兘楂樺害渚濊禆 kernel 瀹炵幇璐ㄩ噺锛坱iling 绛栫暐銆乧ache 鍒╃敤鐜囷級锛孯oofline 璇樊鏈€澶?
- **澶?batch Prefill**锛歁 瓒冲澶э紝tile 鑳藉～婊℃墍鏈?AI Core锛屾渶鎺ヨ繎 Roofline 棰勬祴

**NPU 寰灦鏋勫垎鏋?*锛欴a Vinci 鏍稿績鐨?Cube 鍗曞厓鏈熸湜澶у瀷 tile 杈撳叆锛堥€氬父 16脳16 鏈€灏忥級锛孧=1 鏃?Cube 澶勭悊澶ч噺 padding 闆跺€笺€?

## 闄勫綍 B锛欶RACTAL_NZ 甯冨眬鍒嗘瀽涓庨獙璇?

### 甯冨眬鍏紡

鏍囧噯鍏紡锛堜粠 `torch_npu` 婧愮爜 `FormatHelper.cpp` 纭锛夛細

瀵圭煩闃?`[rows, cols]` 鍋?tiling 鈫?`[ceil(cols/N0), ceil(rows/M0), M0, N0]`

鍏朵腑 `M0 = 16`锛堝浐瀹氾級锛宍N0 = BLOCKBYTES / min(itemsize, 2)`锛?
- BF16/FP16: N0 = 32/2 = 16
- INT8: N0 = 32/1 = 32

**鎭㈠鍏紡**锛歚[..., H, W, block_h, block_w]` 鈫?`[..., K, N]`锛屽叾涓?`K = H * block_w, N = W * block_h`

### 楠岃瘉鏁版嵁

| 绠楀瓙 | Weight 鍩虹甯冨眬 | Tile 灏哄 | FRACTAL_NZ Shape | 鎭㈠鍏紡 |
|-----|---------------|---------|-----------------|---------|
| MatMulV2 (BF16) | [N,K] (鍚?nn.Linear) | 16脳16 | [K/16, N/16, 16, 16] | K=dim0脳16, N=dim1脳16 |
| QuantBatchMatmulV3 (INT8) | [K,N] (杞疆瀛樺偍) | 16脳32 | [N/32, K/16, 16, 32] | K=dim1脳16, N=dim0脳32 |
| GroupedMatmul (INT8) | [E, K, N] | 16脳32 | [E, N/32, K/16, 16, 32] | K=dim2脳16, N=dim1脳32 |

**瀹為檯楠岃瘉**锛?
- Qwen3 BF16: input [136,5120] + weight FRACTAL_NZ [320,48,16,16] 鈫?K=320脳16=5120, N=48脳16=768 鈫?output [136,768] 鉁?
- DSV3 INT8: input [42,7168] + weight FRACTAL_NZ [48,448,16,32] 鈫?K=448脳32=14336? 鈫?鐢ㄩ€氱敤鍏紡: K=H脳block_w=448脳32, N=W脳block_h=48脳16=768

**268 琛岄浂渚嬪楠岃瘉**锛氬湪 Qwen3-30B Prefill 鐨?268 琛?FRACTAL_NZ MatMulV2 鏁版嵁涓婏紝浣跨敤閫氱敤鎭㈠鍏紡涓?Output Shapes 浜ゅ弶楠岃瘉锛屽叏閮ㄩ€氳繃銆?

**鍏抽敭缁撹**锛?
- 鎭㈠鍚?shape 鍜?`aten.mm` 鐨?`args[1]` 瀹屽叏涓€鑷达紝**鏃犻渶杞疆**
- 鍘熷洜锛歚nn.Linear` 鐨?weight `[N,K]` dispatch 鍒?`aten.mm` 鏃跺凡杞疆涓?`[K,N]`锛孨PU 鐨?FRACTAL_NZ 瀛樼殑涔熸槸杞疆鍚庣殑 `[K,N]` 褰㈠紡
- 娌℃湁 TransData 绠楀瓙鈥斺€擣RACTAL_NZ 杞崲鍦?torchair 鍥剧紪璇戞椂瀹屾垚锛屼笉浜х敓杩愯鏃?kernel

## 闄勫綍 C锛欶usedAttention KV Cache 缁村害鍒嗘瀽

闂鎻忚堪鍜岃В鍐虫柟妗堣绗?4.8 鑺傘€傛澶勮ˉ鍏?TensorCast `OpInvokeInfo` 涓?FusedAttention 鐨?args 甯冨眬锛?

| 浣嶇疆 | 鍚箟 | 澶囨敞 |
|-----|------|------|
| `args[0]` | query tensor | (num_tokens, hidden_size) |
| `args[1]` | key tensor | |
| `args[2]` | value tensor | |
| `args[6]` | `seq_lens` | 姣忎釜 request 鐨?KV cache 闀垮害锛屽搴?`actual_seq_lengths_kv` |
| `args[7]` | `query_lens` | 姣忎釜 request 鐨勬柊 query token 鏁?|

## 闄勫綍 D锛歷LLM 鑷畾涔夌畻瀛愯鐩栧垎鏋?

閫氳繃鍒嗘瀽 vLLM `torch_bindings.cpp` 鍜?vllm-ascend issue #1511锛屼富瑕佺殑 vLLM/Ascend 鐗规湁绠楀瓙锛?

| 绠楀瓙 | PyTorch Dispatch 鍙崟鑾凤紵 | TensorCast 瑕嗙洊鎯呭喌 |
|-----|------------------------|-------------------|
| npu_fused_infer_attention_score | 鍚︼紙pybind锛?| 宸叉湁 `tensor_cast.attention` 瀵瑰簲 |
| npu_grouped_matmul | 鍚︼紙pybind锛?| 宸叉湁 `tensor_cast.grouped_matmul` 瀵瑰簲 |
| npu_mm_all_reduce_base (MC2) | 鍚︼紙pybind锛?| 鏈鐩栵紝闇€鏂板 TensorCast 铻嶅悎 pass |
| npu_moe_distribute_dispatch/combine | 鍚︼紙pybind锛?| 宸叉湁 `tensor_cast.permute_tokens/unpermute_tokens` |
| npu_dequant_swiglu_quant | 鍚︼紙pybind锛?| develop 鍒嗘敮宸叉湁铻嶅悎 pass |
| npu_kv_rmsnorm_rope_cache | 鍚︼紙pybind锛?| 鏈鐩?|
| npu_dynamic_quant | 鍚︼紙pybind锛?| 宸叉湁 `tensor_cast.dynamic_quantize_*` |
| vLLM silu_and_mul (CUDA) | 鏄紙torch.library锛?| 宸叉湁铻嶅悎 pass |
| vLLM paged_attention_v1/v2 (CUDA) | 鏄紙torch.library锛?| 宸叉湁 `tensor_cast.attention` |

## 闄勫綍 E锛歰p-plugin Type 鈫?torch_npu 鏄犲皠鍒嗘瀽

op-plugin 搴擄紙https://github.com/Ascend/op-plugin锛夌殑 `op_plugin/config/op_plugin_functions.yaml` 鍖呭惈 7148 琛屻€?200+ 绠楀瓙鏄犲皠銆?

**鏍稿績 kernel Type 鍒?torch_npu API 鐨勬槧灏?*锛?

| Profiling Type | torch_npu API | 鍦?op_plugin_functions.yaml? | 澶囨敞 |
|---------------|--------------|---------------------------|------|
| MatMulV2 | npu_linear / aten::mm | 鉁?(ACL path: OpCommand.Name("MatMulV2")) | |
| QuantBatchMatmulV3 | npu_weight_quant_batchmatmul | 鉁?(OpAPI path) | |
| FusedInferAttentionScore | npu_fused_infer_attention_score | 鉁?(鏈?V2/V3/V4 鐗堟湰) | |
| GroupedMatmul | npu_grouped_matmul | 鉁?| |
| AddRmsNorm | npu_add_rms_norm | 鉁?| |
| DequantSwigluQuant | npu_dequant_swiglu_quant | 鉁?| |
| KvRmsNormRopeCache | npu_kv_rmsnorm_rope_cache | 鉁?| |
| MoeGatingTopK | npu_moe_gating_top_k | 鉁?| |
| AscendQuantV2 | npu_quantize | 鉁?(鍚嶅瓧瀹屽叏涓嶅悓) | |
| DynamicQuant | npu_dynamic_quant | 鉁?| |
| SwiGlu | npu_swiglu | 鉁?| |
| InterleaveRope | npu_interleave_rope | 鉁?| |
| hcom_allReduce_ | torch.distributed.all_reduce | 鉁?(HCCL 閫氫俊搴? | |
| split_qkv_rmsnorm_rope_kernel | 鏃犲叕寮€ API | 鉁?(vLLM-Ascend 鑷畾涔?kernel) | |

**Microbenchmark 榛樿 API**锛坄op_mapping.yaml` 涓?`torch_npu_reference.{type}.microbench_api`锛夛細

| Profiling Type | microbench_api |
|---------------|---------------|
| MatMulV2 | `torch.mm` |
| QuantBatchMatmulV3 | `torch_npu.npu_weight_quant_batchmatmul` |
| FusedInferAttentionScore | `torch_npu.npu_fused_infer_attention_score` |
| GroupedMatmul | `torch_npu.npu_grouped_matmul` |
| AddRmsNorm | `torch_npu.npu_add_rms_norm` |
| SwiGlu | `torch_npu.npu_swiglu` |
| DynamicQuant | `torch_npu.npu_dynamic_quant` |
| AscendQuantV2 | `torch_npu.npu_quantize` |
| hcom_allReduce\_ | `torch.distributed.all_reduce` |
| HcomAllGather | `torch.distributed.all_gather` |
| hcom_alltoall\_ | `torch.distributed.all_to_all` |

**鍏抽敭鍙戠幇**锛?
- 13/15 涓牳蹇?kernel Type 鍙€氳繃 op-plugin 杩芥函鍒?torch_npu API
- 鍛藉悕涓嶄竴鑷达細MatMulV2鈫抧pu_linear銆丄scendQuantV2鈫抧pu_quantize锛屾棤娉曡嚜鍔ㄦ帹瀵?
- op-plugin 鏈変袱鏉¤矾寰勶細ACL path 鐢?`OpCommand.Name("KernelType")` 鐩存帴鍖归厤 Type 鍒楋紝OpAPI path 鐢?aclnn 鍓嶇紑

## 闄勫綍 F锛欰IConfigurator 閫氫俊鏂规瀵规瘮

| 缁村害 | AIConfigurator | 鏈柟妗?|
|-----|---------------|-------|
| 鏁版嵁鏉ユ簮 | NCCL intra-node 瀹炴祴 + 甯﹀缂╂斁鎺ㄧ畻 inter-node | 鍚?topology_tier 鍒嗗埆瀹炴祴 |
| 瀛樺偍缁撴瀯 | `data/{device}/nccl/{version}/nccl_perf.txt` | `data/{device}/hccl/{cann_version}/` |
| 鎷撴墤澶勭悊 | CSV 涓嶅瓨鎷撴墤锛屽姩鎬佹帹瀵?`_get_p2p_bandwidth()` | CSV 瀛?`topology_tier` 鏁存暟鍒?|
| 璺ㄥ眰閫氫俊 | 甯﹀姣斾緥缂╂斁锛坰cale_factor = base_bw / target_bw锛?| 鐩存帴瀹炴祴鍚勫眰绾?|

**閫夋嫨鐞嗙敱**锛?
- AIConfigurator 鍙兘 intra-node 瀹炴祴 + 瑙ｆ瀽缂╂斁锛堝洜涓哄湪 NVIDIA 涓?NCCL 鐨勮涓哄鏉傦級
- 鎴戜滑鍦?Ascend 涓婄敤 HCCL Test / `torch.distributed` 鍙互绮剧‘鎺у埗 `rank_list` 鏉ユ祴浠绘剰鎷撴墤灞傜骇
- 涓嶉渶瑕佽В鏋愮缉鏀撅紝鐩存帴鐢ㄥ疄娴嬫暟鎹洿鍑嗙‘

## 闄勫綍 G锛欰I 杈呭姪寮€鍙戝疄璺靛缓璁?

寤鸿鍦ㄦ湰椤圭洰寮€鍙戜腑閲囩敤浠ヤ笅 AI 杈呭姪寮€鍙戝疄璺碉細

1. **鎺ュ彛鍏堣**锛氬厛瀹氫箟 `DataSource` 鎶借薄鎺ュ彛鍜?`op_mapping.yaml` schema
2. **Planning**锛氫娇鐢?`/superpowers:writing-plans` 鐢熸垚瀹炵幇璁″垝
3. **TDD**锛氫娇鐢?`/superpowers:test-driven-development`锛屽厛鐢熸垚娴嬭瘯鍐嶅疄鐜?
4. **CLAUDE.md 缁存姢**锛氫娇鐢?`/claude-md-management:revise-claude-md` 淇濇寔椤圭洰涓婁笅鏂囨洿鏂?
5. **浠ｇ爜璇勫**锛氫娇鐢?`/superpowers:requesting-code-review` 鍦?PR 鍓嶈嚜鍔ㄨ瘎瀹?
6. **Agent 寮€鍙?*锛氫娇鐢?`/superpowers:dispatching-parallel-agents` 骞惰寮€鍙戠嫭绔嬫ā鍧楋紙濡?ProfilingDataSource 鍜?InterpolatingDataSource 鍙互骞惰寮€鍙戯級

鍛藉悕鏈搴斿尮閰嶆湰鏂囨。锛歚EmpiricalPerformanceModel`銆乣DataSource`銆乣ProfilingDataSource`銆乣InterpolatingDataSource`銆乣op_mapping.yaml` 绛夈€?

## 闄勫綍 H锛氳瀺鍚堢畻瀛愯涔変竴鑷存€ц缁嗗垎鏋?

瀹屾暣鐨?TensorCast op 鈫?Profiling kernel Type 鏄犲皠鍜岃涔変竴鑷存€у垎鏋愶細

| TensorCast Op | Profiling Kernel | Shape 涓€鑷? | 璇存槑 |
|--------------|-----------------|------------|------|
| tensor_cast.static_quant_linear | QuantBatchMatmulV3 | 鉁?M/K/N 瀵归綈 | INT4 鍙樹綋 w=[K/2,N]锛岄渶娉ㄦ剰 K 鎭㈠ |
| tensor_cast.attention | FusedInferAttentionScore | 闇€杞崲 | TC 鐢?(num_tokens, hidden_size)锛孭rofiling 鐢?(batch, num_heads, q_len, head_dim)銆傞€氳繃 `attention_special` 鏌ヨ妯″紡澶勭悊 |
| tensor_cast.multihead_latent_attention | 1:N 鏄犲皠 | MISMATCH | 涓€涓?TC op 瀵瑰簲澶氫釜 kernel (TransposeBatchMatMul + FIA)銆俀1 閫氳繃 `composite: true` + `_lookup_composite()` 鍒嗚В鏌ヨ瀛愬唴鏍稿苟姹傚拰锛涢暱鏈熼渶 MLA decomposition pass |
| tensor_cast.mlapo | 鏃犵洿鎺ュ搴?| N/A | Qwen3 瀵瑰簲 split_qkv_rmsnorm_rope_kernel锛孌SV3 瀵瑰簲澶氫釜鍒嗙珛 kernel |
| tensor_cast.permute_tokens | MoeDistributeDispatchV2 | 閮ㄥ垎 | TC 鍙惈鏈湴 permute锛孭rofiling 鍚€氫俊锛汿C 鐨勯€氫俊鐢?all_to_all 鍒嗗紑璁℃椂 |
| tensor_cast.add_rms_norm | AddRmsNorm / InplaceAddRmsNorm | 鉁?| |
| tensor_cast.swiglu | SwiGlu | 鉁?| DequantSwigluQuant 鏄洿澶х殑铻嶅悎锛岄渶鍗曠嫭 pass |
| 鎵€鏈?comm ops | hcom_allReduce_ 绛?| 鉁?| message_bytes + num_devices 瀵归綈 |

**宸插彂鐜扮殑 Bug**锛?
1. gitcode/develop 鍒嗘敮鐨?`grouped_matmul_*_swiglu` 杈撳嚭 shape 涓?(M, N) 浣嗗簲涓?(M, N//2)
2. `CommAnalyticModel` 涓?`reduce_scatter` 缂哄皯 dispatch 鍒嗘敮锛堝凡鏈夋柟娉曚絾鏈璋冪敤锛?

---

## Change Log

### v1.2 鈫?v1.3 (2026.3.10)

#### 鏂板鍔熻兘

- **[CLI]** `--performance-model` 鏀寔澶氭鎸囧畾锛屽彲鍚屾椂杩愯澶氫釜鎬ц兘妯″瀷
  - 鏃х敤娉? `--performance-model analytic` (鍗曢€?
  - 鏂扮敤娉? `--performance-model analytic --performance-model profiling` (澶氶€?
  - 榛樿鍊? `["analytic"]`

- **[ModelRunnerMetrics]** 鏂板 `tps_per_model: Dict[str, float]` 瀛楁
  - 瀛樺偍姣忎釜鎬ц兘妯″瀷鐙珛璁＄畻鐨?TPS
  - `print_info()` 閬嶅巻杈撳嚭姣忎釜妯″瀷鐨勭粨鏋?

#### 鎺ュ彛鍙樻洿

| 缁勪欢 | 鍙樻洿 | 褰卞搷 |
|------|------|------|
| `UserInputConfig.performance_model` | `str` 鈫?`Union[str, List[str]]` | 鍚戝悗鍏煎锛屽瓧绗︿覆鑷姩鍖呰涓哄垪琛?|
| `ModelRunnerMetrics.execution_time_s` | `float` 鈫?`Dict[str, float]` | **Breaking**: 涓嬫父浠ｇ爜闇€閫傞厤瀛楀吀绫诲瀷 |
| `ModelRunner.perf_model` | `PerformanceModel` 鈫?`List[PerformanceModel]` | 鍐呴儴鍙樻洿锛孉PI 涓嶅彉 |
| `ProfilingDataSource.__init__` | 绉婚櫎 `comm_grid` 鍙傛暟锛屾敼鐢?`device_profile` | **Breaking**: 璋冪敤鏂归渶鏇存柊鍙傛暟 |

#### 浠ｇ爜璐ㄩ噺

- **[model_runner.py]** `PerformanceModel` 瀵煎叆绉昏嚦 `TYPE_CHECKING` 鍧?(RUFF TC001)
- **[user_config.py]** 鏂板 `_normalize_performance_model()` 瑙勮寖鍖栭€昏緫
- **[user_config.py]** 鏂板 `word_embedding_tp_mode` 瀛楁鍙?`_normalize_embedding_tp_mode()` 鏂规硶

#### 娴嬭瘯閫傞厤

- `test_text_generate.py`: `execution_time_s` 鐩稿叧鏂█閫傞厤 `Dict[str, float]`
- `test_vl_compile.py`: 鍚屼笂
- `test_text_generate.py`: `ModelRunnerMetrics` 鏋勯€犳柊澧?`tps_per_model` 鍙傛暟

#### 鏂囦欢鍙樻洿娓呭崟

```
cli/inference/text_generate.py           | CLI 鍙傛暟鏀逛负 action="append"
tensor_cast/core/user_config.py          | performance_model 绫诲瀷鍙樻洿 + WordEmbeddingTPMode
tensor_cast/core/model_runner.py         | 澶氭ā鍨嬫敮鎸?+ ModelRunnerMetrics 瀛楁鍙樻洿
tests/test_tensor_cast/test_text_generate.py  | 娴嬭瘯閫傞厤
tests/test_tensor_cast/test_vl_compile.py      | 娴嬭瘯閫傞厤
```

### v1.1 鈫?v1.2

**鏋舵瀯鍙樻洿**锛?
1. `ProfilingPerformanceModel` 鈫?`EmpiricalPerformanceModel` + `DataSource` 妯″紡
2. `PerfDatabase` 鈫?`DataSource` ABC锛岄瑕佸疄鐜?`ProfilingDataSource`
3. `OperatorSchema` 绫绘秷闄?鈫?鑱岃矗鍒嗘暎鍒?`op_mapping.yaml`锛堝悕瀛楁槧灏勶級+ `InterpolatingDataSource`锛堟彃鍊奸厤缃級+ `ProfilingDataSource`锛堥€氱敤鍖归厤閫昏緫锛?
4. `QueryEngine` 鈫?`InterpolatingDataSource`锛圵rapper 妯″紡锛?
5. `OperatorKey` 娑堥櫎 鈫?`ProfilingDataSource.lookup(OpInvokeInfo)` 鐩存帴鏌ヨ

**鏁版嵁鏍煎紡鍙樻洿**锛?
6. CSV 鍛藉悕锛氳泧褰?鈫?Profiling Type 鍒楀師濮嬪ぇ灏忓啓锛坄MatMulV2.csv`锛?
7. CSV 鏍煎紡锛氳嚜瀹氫箟鍒楋紙m,k,n锛夌Щ闄?鈫?Profiling 鍘熷鏍煎紡锛圛nput Shapes, Input Data Types, Input Formats 绛夛級
8. `metadata.yaml` 鍚堝苟鍏?`op_mapping.yaml`
9. 閫氫俊 CSV锛氱嫭绔嬫牸寮忥紝鍚?`topology_tier` 鏁存暟鍒?
10. FusedAttention CSV锛氱壒娈?microbenchmark 鏍煎紡锛坆atch_size, avg_seq_len, num_heads, head_dim, dtype锛?

**op_mapping.yaml 璁捐鍙樻洿**锛?
11. 绾悕瀛楁槧灏勶紝涓嶅惈 per-op 缁村害鎻愬彇閫昏緫
12. 鏂板 `interpolation_policy`锛堟寜绫诲埆閰嶇疆绮剧‘鍖归厤/鎻掑€肩淮搴︼級
13. 鏂板 `communication_data_ref`锛堟寚鍚?HCCL 鏁版嵁鐩綍鐨勭浉瀵硅矾寰勶級
14. 鏂板 `communication_fallback: analytic`
15. `torch_npu_reference` 缁撴瀯鍖栦负 `apis` 鍒楄〃 + `microbench_api`
16. 閫氫俊鎷撴墤鎻忚堪绉昏嚦 `hccl/{cann_version}/comm_config.yaml`

**鏌ヨ閫昏緫鍙樻洿**锛?
17. FRACTAL_NZ锛氶€氱敤 `fractal_nz_to_nd()` 鎭㈠鍑芥暟锛屾棤闇€杞疆锛?68 琛岄獙璇侀浂渚嬪锛?
18. 鍖归厤绛栫暐锛氬尮閰嶆墍鏈?input shape + dtype锛宱utput shape 浣滀负楠岃瘉
19. 閫氫俊鏌ヨ锛歚rank_group 鈫?CommGrid._get_topology_idx_for_group() 鈫?topology_tier`
20. FusedAttention锛歚query_mode: attention_special`锛屼娇鐢?`args[6]`锛坰eq_lens锛?
21. Composite 鏌ヨ锛歚composite: true` 鏃朵富鍔ㄥ垎瑙ｄ负澶氫釜瀛愬唴鏍告煡璇㈠苟姹傚拰锛圡LA 鍒嗚В鍑芥暟澶嶇敤 analytic model shape 鎺ㄥ锛夛紝涓嶅啀 fallback to analytic

**瀛樺偍缁撴瀯鍙樻洿**锛?
22. 璁＄畻鏁版嵁锛歚data/{device}/vllm_ascend/{version}/`
23. 閫氫俊鏁版嵁锛歚data/{device}/hccl/{cann_version}/`锛堣法 vLLM 鐗堟湰澶嶇敤锛?
24. 宸ュ叿锛歚tools/perf_data_collection/`

**鍐呭鍙樻洿**锛?
25. 2.1 鑺傦細淇"鍐呮牳鍚姩寮€閿€"涓?tiling/鍒╃敤鐜?锛屽垹闄ゆ棤鏉ユ簮鐨?瀹炴祴44-95%"鏁版嵁
26. 鏁版嵁搴撴瀯寤猴細涓ょ骇绛栫暐 鈫?涓夋璧帮紙Profiling 鈫?microbenchmark 缃戞牸 鈫?楠岃瘉锛?
27. 閫氫俊 microbenchmark锛歅ython 鑴氭湰缁熶竴璁＄畻+閫氫俊锛孒CCL Test 浣滀负浜ゅ弶楠岃瘉
28. 9.1 鑺傦細鏇存柊铻嶅悎 Gap 宸插叧闂?浠嶅紑鏀剧姸鎬侊紙鍩轰簬 gitcode/develop 鍒嗘瀽锛?
29. 閬楃暀闂 1锛氫腑鏈熸柟妗堬紙铻嶅悎 Pass 瀵归綈锛変负姝ｅ紡鏂规锛岄暱鏈熸柟妗堬紙FX graph + 渚靛叆寮?DispatchMode锛夌Щ鑷?Future Work
30. 閬楃暀闂 2锛欶usedAttention microbenchmark 涓哄綋鍓嶆柟妗堬紝涓ゆ潯闀挎湡璺緞鍦?Future Work

**鏂板绀轰緥鏂囦欢**锛?
31. `docs/perf_database/examples/op_mapping_example.yaml`锛氬畬鏁?op_mapping.yaml 绀轰緥锛垀25 鏉＄畻瀛愭槧灏?+ torch_npu_reference锛?
32. `docs/perf_database/examples/comm_config_example.yaml`锛氬畬鏁?comm_config.yaml 绀轰緥锛堟嫇鎵戞弿杩?+ 閫氫俊绠楀瓙鏄犲皠锛?

**鏂板闄勫綍**锛?
33. 闄勫綍 A锛氬皬 Batch 鍦烘櫙 Roofline 鍋忓樊鍒嗘瀽
34. 闄勫綍 B锛欶RACTAL_NZ 甯冨眬鍒嗘瀽涓庨獙璇佹暟鎹?
35. 闄勫綍 C锛欶usedAttention KV Cache 缁村害鍒嗘瀽
36. 闄勫綍 D锛歷LLM 鑷畾涔夌畻瀛愯鐩栧垎鏋?
37. 闄勫綍 E锛歰p-plugin Type 鈫?torch_npu 鏄犲皠鍒嗘瀽
38. 闄勫綍 F锛欰IConfigurator 閫氫俊鏂规瀵规瘮
39. 闄勫綍 G锛欰I 杈呭姪寮€鍙戝疄璺靛缓璁?
40. 闄勫綍 H锛氳瀺鍚堢畻瀛愯涔変竴鑷存€ц缁嗗垎鏋?

**寮€鍙戣鍒?*锛?
41. 涓ゅ洟闃熺粨鏋勪繚鎸侊紝鐩爣鏇存柊涓?3.23
42. 涓夐樁娈碉細Phase 1 璁＄畻绠楀瓙绌垮埡锛堚啋3.14锛夆啋 Phase 2 閫氫俊绠楀瓙鎺ュ叆+鎻掑€硷紙鈫?.20锛夆啋 Phase 3 闆嗘垚楠岃瘉锛堚啋3.23锛?

### v1.0 鈫?v1.1

1. 浠庡崟涓€ `ProfilingPerformanceModel`锛堝唴鍚?Roofline 鍏滃簳锛夐噸鏋勪负涓変釜鐙珛 PerformanceModel 骞跺垪锛孯oofline 鍏滃簳涓婄Щ鑷?Model 灞?`fallback_model`
2. `PerfDatabase` 浠?Profiling 涓撶敤鍖呮彁鍗囦负鍏变韩鏁版嵁灞傦紝鍚屾椂涓?Profiling锛堝彧璇伙級鍜?Empirical锛堣鍐欐寔涔呭寲缂撳瓨锛夋湇鍔?
3. 鏂板 `OperatorKey` 鍏变韩鎶借薄锛屽皝瑁?`OpInvokeInfo 鈫?鏌ヨ key` 杞崲閫昏緫
4. `PerfDatabase` API 浠?`(system, backend, version)` 涓夊厓缁勬敼涓?`data_path` 璺緞鐩翠紶锛屽垹闄?`VersionManager`
5. Schema 绮掑害浠庢寜鎶借薄绠楀瓙绫诲瀷鍒嗙粍锛垀6 绫伙級鏀逛负鎸?kernel_details.csv 纭欢鍐呮牳涓€涓€瀵归綈锛垀17 涓級
6. YAML 鏄犲皠鏍煎紡浠庢寜 Schema 鍒嗙粍鏀逛负鎵佸钩鐨?`tensorcast_op_to_schema`锛?:1锛?
7. QueryEngine 浠?4 绾ч檷绾э紙鍚?Roofline锛夌簿绠€涓?3 绾э紙绮剧‘鈫掓彃鍊尖啋澶栨帹锛?
8. 瀛樺偍鏍煎紡浠?Parquet 涓轰富鏀逛负 CSV 涓轰富锛涙暟鎹噰闆嗘墿灞曚负涓夌鍙€夋柟妗?
9. 鍩轰簬 DeepSeekV3/Qwen3 瀹炴祴 Profiling 鏁版嵁淇浜嗗椤圭畻瀛愭槧灏?
10. 鏁版嵁閲囬泦娴佹按绾夸粠 TensorCast 鍖呭唴绉昏嚦浠撳簱椤跺眰鐙珛瀛愮郴缁燂紱寮€鍙戣鍒掍粠 5 闃舵鍗曞洟闃熸敼涓?3 闃舵鍙屽洟闃熷垎宸?

