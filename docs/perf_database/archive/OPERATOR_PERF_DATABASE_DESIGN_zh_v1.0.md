# TensorCast 绠楀瓙鎬ц兘鏁版嵁搴擄細鎶€鏈璁℃枃妗?
**鐗堟湰**: 1.0  
**鏃ユ湡**: 2026 骞?2 鏈? 
**鑼冨洿**: 闈㈠悜鍗庝负鏄囪吘 A3 鐨?LLM 浠跨湡鍙墿灞?Profiling  costmodel  
**鍒濇湡鐩爣妯″瀷**: DeepSeek-V3銆丵wen3-32B

---

## 1. 鎶€鏈垎鏋?
### 1.1 闂闄堣堪
TensorCast 褰撳墠閲囩敤**鍩轰簬 Roofline 鐨勮В鏋愭ā鍨?*锛坄AnalyticPerformanceModel`锛変及绠楃畻瀛愭墽琛岃€楁椂銆傝妯″瀷鍩轰簬娴偣杩愮畻閲忥紙FLOPs锛変笌璁垮瓨瀛楄妭鏁拌绠?`max(璁＄畻鑰楁椂, 璁垮瓨鑰楁椂)`锛屼笌 Atlas 800 A3 纭欢涓婄殑瀹為檯 VLLM-Ascend Profiling 缁撴灉瀵规瘮瀛樺湪绯荤粺鎬у亸宸€?

**鍋忓樊鏍瑰洜鍒嗘瀽**锛?

| 鍋忓樊鏉ユ簮 | 鍏蜂綋鎻忚堪 | 褰卞搷绋嬪害 |
| --- | --- | --- |
| **绠楀瓙铻嶅悎** | VLLM 灏嗗涓畻瀛愯瀺鍚堜负鍗曚竴鍐呮牳鎵ц锛堝 `DequantSwigluQuant`銆乣AddRmsNorm`锛?| 10-30% 鏃堕棿宸紓 |
| **纭欢鍒╃敤鐜?* | 瀹為檯 Cube 鍒╃敤鐜囩害 44-68%锛岃€岄潪 Roofline 鍋囪鐨?100% | 楂樹及璁＄畻瀵嗛泦鍨嬬畻瀛?|
| **Shape 鐩稿叧寮€閿€** | 灏?batch 鍦烘櫙涓嬪唴鏍稿惎鍔ㄥ紑閿€鏈缓妯?| 浣庝及 Decode 闃舵鑰楁椂 |
| **鐗堟湰鐩稿叧浼樺寲** | VLLM-Ascend 鍚勭増鏈紩鍏ヤ笉鍚岀殑铻嶅悎鍐呮牳 | 妯″瀷鍦ㄧ増鏈凯浠ｄ腑澶辨晥 |


### 1.2 Profiling 鏁版嵁鍒嗘瀽 锛圱ODO锛氭牴鎹渶鏂扮殑鑷姩鍒嗘瀽鍒蜂竴閬嶆暟鎹?02鏈?6鏃@HXW](https://www.yuque.com/hxw02477402)锛?
鍩轰簬瀹為檯鏄囪吘 Profiler 杈撳嚭锛坄kernel_details.csv`銆乣op_statistic.csv`锛夛細

**Qwen3-32B锛堝叡 41 涓嫭绔嬬畻瀛愶級**锛?

| 鏍稿績绠楀瓙 | 鑰楁椂鍗犳瘮 | 鏍稿績绫诲瀷 |
| --- | --- | --- |
| MatMulV2 | 42.4% | AI_CORE |
| FusedInferAttentionScore | 18.2% | MIX_AIC |
| TensorMove | 10.7% | AI_VECTOR_CORE |
| AddRmsNorm | 8.0% | AI_VECTOR_CORE |
| split_qkv_rmsnorm_rope_kernel | 5.2% | MIX_AIC |
| SwiGlu | 4.8% | AI_VECTOR_CORE |


**DeepSeek-V3锛堝叡 39 涓嫭绔嬬畻瀛愶紝鍚?MoE 鐗规湁绠楀瓙锛?*锛?

| 鏍稿績绠楀瓙 | 鑰楁椂鍗犳瘮 | 鏍稿績绫诲瀷 |
| --- | --- | --- |
| GroupedMatmul | 20.9% | AI_CORE |
| FusedInferAttentionScore | 18.5% | MIX_AIC |
| QuantBatchMatmulV3 | 16.9% | AI_CORE |
| MoeDistributeDispatch/Combine | 11.8% | MIX_AIC |
| AscendQuantV2 | 4.4% | AI_VECTOR_CORE |
| TransposeBatchMatMul | 4.2% | AI_CORE |


**鍏抽敭鍙戠幇**锛氱害 15 涓畻瀛愯础鐚簡瓒呰繃 95% 鐨勬墽琛屾椂闂淬€傚叾浣欑害 65 涓畻瀛愶紙绠楁湳銆佺储寮曘€侀€昏緫杩愮畻锛夎础鐚笉瓒?5%锛屽彲閲囩敤 Roofline 鍏滃簳浼扮畻銆?

### 1.3 鐗堟湰褰卞搷鍒嗘瀽
VLLM-Ascend 涓嶅悓鐗堟湰瀵硅瀺鍚堝唴鏍哥殑绉嶇被鍙婂叾鎬ц兘鏈夋樉钁楀奖鍝嶏細

| 鐗堟湰 | 鏂板铻嶅悎鍐呮牳 | 鎬ц兘褰卞搷 |
| --- | --- | --- |
| v0.14.0 | Triton RoPE 鍐呮牳銆丮atMul-AllReduce-RMSNorm 铻嶅悎 | 浜х敓鍏ㄦ柊铻嶅悎妯″紡 |
| v0.12.0 | AddRmsNormQuant 铻嶅悎銆佸ぇ閲?Triton 鍐呮牳 | 绠楀瓙鎵ц鍥惧彂鐢熷彉鍖?|
| v0.10.x | MLP 寮犻噺骞惰銆丄llGather-Expert 铻嶅悎 | 涓撳璺敱鏂瑰紡鍙樻洿 |


CANN 鐗堟湰鍚屾牱褰卞搷鍐呮牳鎵ц鏁堢巼锛堜笉鍚岀紪璇戜紭鍖栫瓥鐣ワ級锛屼袱鑰呭潎闇€绾冲叆鐗堟湰杩借釜銆?

### 1.4 鐜版湁鍩虹璁炬柦
`msmodeling-profiling_compare`锛堝凡瀹炵幇锛夛細

+ 铻嶅悎鎰熺煡鐨勭畻瀛愭槧灏勶紙YAML 閰嶇疆锛?00+ 鏄犲皠瑙勫垯锛?
+ VLLM kernel_details.csv 瑙ｆ瀽鍣紝鏀寔闃舵妫€娴?
+ TensorCast 浠跨湡閫傞厤鍣?
+ Excel 瀵规瘮鎶ュ憡
+ 浠ｇ爜浣嶇疆锛歚tensor_cast/scripts/profiling_comparison/`

`EmpiricalPerformanceModel`锛堝凡瀹炵幇锛屼粎鏀寔 JIT 鍩哄噯娴嬭瘯锛夛細

+ `tensor_cast/performance_model/empirical.py` 鈥?鍦ㄥ疄闄呰澶囦笂杩愯鍩哄噯娴嬭瘯
+ 浣跨敤 `OpBenchmark` 绫?
+ 浠呭湪鐗╃悊璁惧鍙敤鏃跺伐浣?

### 1.5 AI Configurator 鍙傝€冿紙NVIDIA锛?
浠?`/home/horacehxw/Projects/aiconfigurator` 椤圭洰涓彁鐐肩殑鏍稿績璁捐妯″紡锛?

+ **宓屽瀛楀吀绱㈠紩**锛氬宸?Profiling 鐨?Shape 瀹炵幇 O(1) 绮剧‘鏌ユ壘
+ **scipy.griddata 鎻掑€?*锛氬缁?Shape 绌洪棿鍖归厤
+ **DatabaseMode 闄嶇骇绛栫暐**锛歋ILICON 鈫?HYBRID 鈫?EMPIRICAL 鈫?SOL
+ **PerformanceResult(float)**锛氬悗鍚戝吋瀹圭殑缁撴灉绫诲瀷
+ **CSV 瀛樺偍**锛氬彲璇绘€у己銆佷究浜庣増鏈鐞?
+ **寤惰繜鍔犺浇涓庣紦瀛?*锛氭ā鍧楃骇缂撳瓨閬垮厤閲嶅鍔犺浇

---

## 2. 绯荤粺鏋舵瀯
### 2.1 鏁翠綋鏋舵瀯
**TensorCast Runtime 閫氳繃 `PerformanceModel` 鎻掍欢鍖栨灦鏋勬帴鍏?Profiling 鏁版嵁搴擄紝閫夋嫨鍝寤烘ā搴旇鏄彲浠ラ厤缃殑锛岄兘蹇呴』鏀寔銆?*

```plain
鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
鈹?                          TensorCast Runtime                                  鈹?
鈹?                                                                              鈹?
鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?    鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹? 鈹?
鈹? 鈹? Runtime            鈹?    鈹? ProfilingPerformanceModel锛堟柊澧烇級          鈹? 鈹?
鈹? 鈹? (TorchDispatchMode)鈹傗攢鈹€鈹€鈹€鈻垛攤  鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?  鈹? 鈹?
鈹? 鈹?                    鈹?    鈹? 鈹?1. 鍖归厤绠楀瓙 鈫?OperatorSchema        鈹?  鈹? 鈹?
鈹? 鈹? 鎷︽埅鎵€鏈夌畻瀛愯皟鐢?  鈹?    鈹? 鈹?2. 浠?OpInvokeInfo 鎻愬彇 Shape       鈹?  鈹? 鈹?
鈹? 鈹? 鐢熸垚 OpInvokeInfo  鈹?    鈹? 鈹?3. 鏌ヨ PerfDatabase               鈹?  鈹? 鈹?
鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?    鈹? 鈹?4. 闄嶇骇鑷?AnalyticModel             鈹?  鈹? 鈹?
鈹?                             鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?  鈹? 鈹?
鈹?                             鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹? 鈹?
鈹?                                        鈹?                                    鈹?
鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹尖攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
                                          鈹?
                    鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹尖攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
                    鈹?                    鈻?                    鈹?
                    鈹?             PerfDatabase                 鈹?
                    鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹?
                    鈹? 鈹? QueryEngine                      鈹?   鈹?
                    鈹? 鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?鈹?   鈹?
                    鈹? 鈹? 鈹?绮剧‘鍖归厤 鈫?鎻掑€间及绠?         鈹?鈹?   鈹?
                    鈹? 鈹? 鈹?鈫?澶栨帹浼扮畻 鈫?Roofline 鍏滃簳   鈹?鈹?   鈹?
                    鈹? 鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?鈹?   鈹?
                    鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹?
                    鈹?                    鈹?                    鈹?
                    鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹粹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?鈹?
                    鈹? 鈹?OperatorSchema 娉ㄥ唽琛?              鈹?鈹?
                    鈹? 鈹?鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹?鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?鈹屸攢鈹€鈹€鈹?鈹屸攢鈹€鈹€鈹€鈹€鈹?鈹?鈹?
                    鈹? 鈹?鈹?GEMM 鈹?鈹侫ttention鈹?鈹侻oE鈹?鈹侳used鈹?鈹?鈹?
                    鈹? 鈹?鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹?鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?鈹斺攢鈹€鈹€鈹?鈹斺攢鈹€鈹€鈹€鈹€鈹?鈹?鈹?
                    鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?鈹?
                    鈹?                    鈹?                    鈹?
                    鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹粹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?鈹?
                    鈹? 鈹?瀛樺偍灞傦紙鎸夌増鏈垎鐩綍锛孭arquet/CSV锛? 鈹?鈹?
                    鈹? 鈹?atlas_a3/vllm_ascend/0.14.0/        鈹?鈹?
                    鈹? 鈹?  gemm.parquet, attention.parquet    鈹?鈹?
                    鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?鈹?
                    鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?

鏁版嵁閲囬泦娴佹按绾匡紙绂荤嚎鎵ц锛?
鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
鈹? Level 1: 鍏ㄦā鍨?Profiling                                    鈹?
鈹? VLLM serve + bench 鈫?kernel_details.csv 鈫?瑙ｆ瀽 鈫?鍏ュ簱       鈹?
鈹溾攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
鈹? Level 2: 鍗曠畻瀛愬井鍩哄噯娴嬭瘯                                    鈹?
鈹? torch_npu 鑴氭湰 鈫?鐩存帴娴嬮噺 鈫?鍏ュ簱                            鈹?
鈹溾攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
鈹? 鏍″噯锛氫互 Level 1 鏁版嵁涓哄熀鍑嗭紝鏍℃ Level 2 娴嬮噺鍋忓樊           鈹?
鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
```

### 2.2 妯″潡缁撴瀯
```plain
tensor_cast/perf_database/              # 鏂板鍖?
鈹溾攢鈹€ __init__.py
鈹溾攢鈹€ core/
鈹?  鈹溾攢鈹€ database.py                     # PerfDatabase 涓荤被
鈹?  鈹溾攢鈹€ query.py                        # QueryEngine锛堝惈鎻掑€奸€昏緫锛?
鈹?  鈹溾攢鈹€ storage.py                      # Parquet/CSV IO 鍚庣
鈹?  鈹斺攢鈹€ versioning.py                   # 鐗堟湰瑙ｆ瀽涓庣鐞?
鈹溾攢鈹€ operators/
鈹?  鈹溾攢鈹€ __init__.py                     # 閫氳繃 importlib 鑷姩鍙戠幇
鈹?  鈹溾攢鈹€ base.py                         # OperatorSchema 鎶借薄鍩虹被 + 娉ㄥ唽琛?
鈹?  鈹溾攢鈹€ gemm.py                         # GEMM/MatMul 绠楀瓙 Schema
鈹?  鈹溾攢鈹€ attention.py                    # MHA銆丟QA銆丮LA 绠楀瓙 Schema
鈹?  鈹溾攢鈹€ moe.py                          # MoE 璺敱 + 涓撳璁＄畻 Schema
鈹?  鈹溾攢鈹€ communication.py                # allreduce銆乤llgather銆乤lltoall
鈹?  鈹溾攢鈹€ normalization.py                # RMSNorm銆丄ddRmsNorm
鈹?  鈹溾攢鈹€ fused.py                        # DequantSwigluQuant 绛夎瀺鍚堢畻瀛?
鈹?  鈹斺攢鈹€ elementwise.py                  # Cast銆佹縺娲诲嚱鏁般€侀€愬厓绱犺繍绠?
鈹溾攢鈹€ shape_generators/
鈹?  鈹溾攢鈹€ base.py                         # ShapeGridGenerator 鎶借薄鍩虹被
鈹?  鈹溾攢鈹€ model_driven.py                 # 浠?HuggingFace 閰嶇疆鑷姩鎻愬彇
鈹?  鈹斺攢鈹€ universal.py                    # 2 鐨勫箓娆?+ 甯歌 Shape 妯℃澘
鈹溾攢鈹€ interpolation/
鈹?  鈹溾攢鈹€ base.py                         # InterpolationStrategy 鎶借薄鍩虹被
鈹?  鈹斺攢鈹€ linear.py                       # 鍩轰簬 scipy 鐨勭嚎鎬ф彃鍊?
鈹溾攢鈹€ mappings/vllm_ascend/
鈹?  鈹溾攢鈹€ v0.10.yaml                      # VLLM 鍐呮牳 鈫?Schema 鏄犲皠
鈹?  鈹溾攢鈹€ v0.12.yaml
鈹?  鈹斺攢鈹€ v0.14.yaml
鈹溾攢鈹€ data/systems/                       # Profiling 鏁版嵁瀛樺偍锛坓itignore锛?
鈹?  鈹斺攢鈹€ ATLAS_800_A3_752T_128G_DIE/
鈹?      鈹斺攢鈹€ vllm_ascend/{version}/
鈹?          鈹溾攢鈹€ metadata.yaml
鈹?          鈹溾攢鈹€ gemm.parquet
鈹?          鈹溾攢鈹€ attention.parquet
鈹?          鈹斺攢鈹€ ...
鈹斺攢鈹€ scripts/
    鈹溾攢鈹€ collect_full_model.py           # Level 1: 鍩轰簬 VLLM 鐨?Profiling
    鈹溾攢鈹€ collect_microbench.py           # Level 2: torch_npu 寰熀鍑嗘祴璇?
    鈹溾攢鈹€ parse_ascend_output.py          # 瑙ｆ瀽 kernel_details.csv
    鈹溾攢鈹€ generate_shape_grid.py          # 鏍规嵁妯″瀷閰嶇疆鐢熸垚 Shape 缃戞牸
    鈹溾攢鈹€ calibrate.py                    # L2 瀵归綈 L1 鐨勬牎鍑?
    鈹斺攢鈹€ database validation tool                     # 鏁版嵁搴撶簿搴﹂獙璇?

tensor_cast/performance_model/
鈹斺攢鈹€ profiling.py                        # ProfilingPerformanceModel锛堟柊澧烇級
```

---

## 3. 鏍稿績妯″潡璁捐
### 3.1 OperatorSchema锛堟彃浠跺寲鏋舵瀯锛?
姣忕绠楀瓙绫诲瀷瀹氫箟鍏?Shape 绌洪棿銆佹彁鍙栭€昏緫鍜?Roofline 鍏滃簳浼扮畻锛?

```python
# tensor_cast/perf_database/operators/base.py

class DimensionType(Enum):
    BATCH = auto()      # 闅忛儴缃插彉鍖栵紙batch_size銆乶um_tokens锛?
    SEQUENCE = auto()   # 闅忚緭鍏ュ彉鍖栵紙query_len銆乧ontext_len锛?
    FEATURE = auto()    # 妯″瀷鍥烘湁鍙傛暟锛坔idden_size銆乭ead_dim锛?
    EXPERT = auto()     # MoE 涓撴湁锛坣um_experts銆乼op_k锛?
    DEVICE = auto()     # 骞惰绛栫暐鐩稿叧锛坣um_devices锛?

@dataclass
class DimensionSpec:
    name: str
    dim_type: DimensionType
    typical_range: Tuple[int, int]
    typical_values: Optional[List[int]] = None
    is_required: bool = True

class OperatorSchema(ABC):
    """
    绠楀瓙鎬ц兘 Schema 鍩虹被銆?
    閲囩敤娉ㄥ唽琛ㄦā寮忚嚜鍔ㄥ彂鐜帮紙涓?DeviceProfile 璁捐涓€鑷达級銆?
    """
    _registry: ClassVar[Dict[str, Type["OperatorSchema"]]] = {}

    @classmethod
    def register(cls, op_name: str):
        """娉ㄥ唽瑁呴グ鍣?""
        def decorator(schema_cls):
            cls._registry[op_name] = schema_cls
            return schema_cls
        return decorator

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def dimensions(self) -> List[DimensionSpec]: ...

    @property
    @abstractmethod
    def tensorcast_ops(self) -> List[str]:
        """璇?Schema 澶勭悊鐨?TensorCast 绠楀瓙鍚嶇О鍒楄〃"""
        ...

    @abstractmethod
    def extract_shape_from_op(self, op_invoke_info: OpInvokeInfo) -> Dict[str, Any]:
        """浠庢嫤鎴埌鐨勭畻瀛愯皟鐢ㄤ腑鎻愬彇 Shape 缁村害"""
        ...

    @abstractmethod
    def compute_roofline_estimate(self, shape, device_profile) -> float:
        """璁＄畻 Roofline锛圫peed-of-Light锛変及绠楀€硷紝鍗曚綅寰"""
        ...

    def get_interpolation_dimensions(self) -> List[str]:
        """杩斿洖閫傜敤浜庤繛缁彃鍊肩殑缁村害"""
        return [d.name for d in self.dimensions
                if d.dim_type in (DimensionType.BATCH, DimensionType.SEQUENCE)]
```

**鍏蜂綋 Schema 涓€瑙?*锛?

| Schema | 鍏抽敭缁村害 | 瀵瑰簲 TensorCast 绠楀瓙 |
| --- | --- | --- |
| `GEMMSchema` | m, n, k, quant_mode | `aten.mm`銆乣static_quant_linear`銆乣fp8_linear`銆乣grouped_matmul_quant` |
| `AttentionSchema` | batch, query_len, context_len, num_heads, num_kv_heads, head_dim, kv_lora_rank | `attention`銆乣attention_quant`銆乣multihead_latent_attention` |
| `MoESchema` | num_tokens, num_experts, top_k, hidden_size, intermediate_size | `grouped_matmul`銆乣permute_tokens`銆乣unpermute_tokens` |
| `NormalizationSchema` | num_tokens, hidden_size | `rms_norm`銆乣add_rms_norm` |
| `FusedSchema` | 锛堥殢铻嶅悎鍐呮牳鑰屽紓锛?| 鏄犲皠鑷崇粍鎴愮畻瀛?|
| `CommunicationSchema` | op_type, num_devices, message_size | `all_reduce`銆乣all_gather`銆乣all_to_all` |


### 3.2 PerfDatabase
```python
# tensor_cast/perf_database/core/database.py

class PerfDatabase:
    """鎬ц兘鏁版嵁搴撲富绫汇€傚姞杞?Profiling 鏁版嵁锛屾彁渚涙煡璇㈡帴鍙ｃ€?""

    def __init__(
        self,
        system: str = "ATLAS_800_A3_752T_128G_DIE",
        backend: str = "vllm_ascend",
        version: str = "latest",
        data_root: Optional[Path] = None,
    ):
        self.version_mgr = VersionManager(data_root)
        self.resolved_version, self.data_path = self.version_mgr.resolve(system, backend, version)
        self.query_engine = QueryEngine(self)
        self._data_cache: Dict[str, pd.DataFrame] = {}

    def query(self, schema_name: str, shape: Dict, mode: QueryMode = QueryMode.HYBRID) -> QueryResult:
        """涓绘煡璇㈠叆鍙ｏ紝濮旀墭缁?QueryEngine銆?""
        return self.query_engine.query(schema_name, shape, mode)

    def get_data(self, schema_name: str, quant_mode: str) -> Optional[pd.DataFrame]:
        """寤惰繜鍔犺浇骞剁紦瀛樼畻瀛?Parquet 鏁版嵁"""
        cache_key = f"{schema_name}_{quant_mode}"
        if cache_key not in self._data_cache:
            path = self.data_path / f"{schema_name}.parquet"
            if not path.exists():
                return None
            df = pd.read_parquet(path)
            self._data_cache[cache_key] = df[df["quant_mode"] == quant_mode]
        return self._data_cache[cache_key]
```

### 3.3 QueryEngine锛堟彃鍊?+ 闄嶇骇绛栫暐锛?
```python
# tensor_cast/perf_database/core/query.py

class QueryMode(Enum):
    EXACT = auto()
    INTERPOLATE = auto()
    HYBRID = auto()       # 绮剧‘鍖归厤 鈫?鎻掑€?鈫?Roofline
    ROOFLINE = auto()

class QuerySource(Enum):
    MEASURED = auto()          # 绮剧‘ Profiling 鏁版嵁锛堢疆淇″害: 1.0锛?
    INTERPOLATED = auto()      # scipy 鎻掑€硷紙缃俊搴? 0.7-0.95锛?
    EXTRAPOLATED = auto()      # 鍑稿寘澶栨帹锛堢疆淇″害: 0.3-0.6锛?
    ROOFLINE = auto()          # 瑙ｆ瀽妯″瀷锛堢疆淇″害: 0.4锛?
    ROOFLINE_CALIBRATED = auto()  # Roofline 脳 缁忛獙鏁堢巼绯绘暟锛堢疆淇″害: 0.6锛?

@dataclass
class QueryResult:
    latency_us: float
    confidence: float
    source: QuerySource
    details: Dict[str, Any] = field(default_factory=dict)

class QueryEngine:
    """澶勭悊绮剧‘鏌ユ壘銆佹彃鍊间及绠楀拰闄嶇骇绛栫暐銆?""

    def query(self, schema_name, shape, mode=QueryMode.HYBRID) -> QueryResult:
        schema = OperatorSchema.get_schema(schema_name)

        if mode == QueryMode.HYBRID:
            # 鎸変紭鍏堢骇渚濇灏濊瘯锛氱簿纭尮閰?鈫?鎻掑€?鈫?Roofline 鍏滃簳
            result = self._try_exact(schema, shape)
            if result: return result

            result = self._try_interpolate(schema, shape)
            if result: return result

            # TODO: try cover all ops, avoid to fallback
            return self._roofline_fallback(schema, shape)

    def _try_interpolate(self, schema, shape) -> Optional[QueryResult]:
        """鍩轰簬 scipy.interpolate.LinearNDInterpolator 鐨勫缁存彃鍊?""
        # 鑾峰彇閫傜敤浜庤繛缁彃鍊肩殑缁村害
        interp_dims = schema.get_interpolation_dimensions()
        # 浠庡凡鏈夋祴閲忔暟鎹瀯寤虹偣浜?
        # 鎵ц鎻掑€艰绠楋紝鏍规嵁鍒版渶杩戞祴閲忕偣鐨勮窛绂讳及绠楃疆淇″害
        ...
```

### 3.4 绠楀瓙鍒?Schema 鐨勬槧灏勶紙YAML 閰嶇疆锛?
```yaml
# tensor_cast/perf_database/mappings/vllm_ascend/v0.14.yaml
version: "0.14"
device: ATLAS_800_A3_752T_128G_DIE

operator_schemas:
  gemm:
    vllm_kernels:
      - name: "MatMulV2"
        shape_parser: "matmul_v2"
      - name: "GroupedMatmul"
        shape_parser: "grouped_matmul"
      - name: "QuantBatchMatmulV3"
        shape_parser: "quant_batch_matmul"
      - name: "TransposeBatchMatMul"
        shape_parser: "transpose_batch_matmul"
    tensorcast_ops:
      - "aten.mm.default"
      - "tensor_cast.static_quant_linear.default"
      - "tensor_cast.static_quant_linear_int4.default"
      - "tensor_cast.fp8_linear.default"
      - "tensor_cast.grouped_matmul_quant.default"

  attention:
    vllm_kernels:
      - name: "FusedInferAttentionScore"
        shape_parser: "fused_attention"
    tensorcast_ops:
      - "tensor_cast.attention.default"
      - "tensor_cast.attention_quant.default"
      - "tensor_cast.multihead_latent_attention.default"
      - "tensor_cast.multihead_latent_attention_quant.default"

  moe_routing:
    vllm_kernels:
      - name: "MoeDistributeDispatchV2"
      - name: "MoeDistributeCombineV2"
      - name: "MoeGatingTopK"
    tensorcast_ops:
      - "tensor_cast.permute_tokens.default"
      - "tensor_cast.unpermute_tokens.default"

  fused:
    vllm_kernels:
      - name: "DequantSwigluQuant"
        components: [dequantize, silu, mul, quantize]
      - name: "AddRmsNorm"
        components: [add, rms_norm]
      - name: "InplaceAddRmsNorm"
        components: [add, rms_norm]
      - name: "split_qkv_rmsnorm_rope_kernel"
        components: [rms_norm, qkv_split, apply_rope]
      - name: "KvRmsNormRopeCache"
        components: [rms_norm, apply_rope, reshape_and_cache]
```

---

## 4. 鎺ュ彛璁捐锛歍ensorCast 闆嗘垚
### 4.1 ProfilingPerformanceModel
杩欐槸涓?TensorCast 闆嗘垚鐨勬牳蹇冨叆鍙ｃ€傚畠缁ф壙鑷?`PerformanceModel`锛堜笌 `AnalyticPerformanceModel`銆乣EmpiricalPerformanceModel` 鍏辩敤鍚屼竴鎶借薄鍩虹被锛夈€?

```python
# tensor_cast/performance_model/profiling.py

class ProfilingPerformanceModel(PerformanceModel):
    """
    鍩轰簬 Profiling 鏁版嵁鐨勬€ц兘妯″瀷銆?

    涓?TensorCast 鐨勯泦鎴愭柟寮忥細
    - 缁ф壙 PerformanceModel锛坱ensor_cast/performance_model/__init__.py:169锛?
    - 瀹炵幇 process_op(OpInvokeInfo) 鈫?PerformanceModel.Result
    - 閫氳繃 perf_models 鍒楄〃鎺ュ叆 Runtime锛坱ensor_cast/runtime.py:41锛?
    - 鐢?Runtime 鑷姩鍖呰涓?CachingPerformanceModel
    - 鏀寔 get_classifiers() 鐢ㄤ簬鐡堕鍒嗙被缁熻
    """

    def __init__(
        self,
        device_profile: DeviceProfile,            # 鏉ヨ嚜 tensor_cast/device.py
        database_mode: QueryMode = QueryMode.HYBRID,
        vllm_version: str = "latest",
    ):
        super().__init__("profiling", device_profile)  # name="profiling"
        self.db = PerfDatabase(
            system=self._device_to_system(device_profile),
            version=vllm_version,
        )
        self.query_mode = database_mode
        self.fallback = AnalyticPerformanceModel(device_profile)  # 鐜版湁瑙ｆ瀽妯″瀷
        self._schema_cache: Dict[str, Optional[OperatorSchema]] = {}

    def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        """
        鏍稿績鎺ュ彛鏂规硶銆?
        鐢?Runtime.__torch_dispatch__ 鍦ㄦ瘡娆＄畻瀛愭嫤鎴椂璋冪敤銆?

        鍙傛暟:
            op_invoke_info: 鍖呭惈 func銆乤rgs銆乲wargs銆乷ut銆乧ache_key
                           锛堝畾涔変簬 tensor_cast/performance_model/__init__.py:52锛?

        杩斿洖:
            PerformanceModel.Result锛屽寘鍚?execution_time_s 鍜?statistics 瀛楀吀
            锛堝畾涔変簬 tensor_cast/performance_model/__init__.py:175锛?
        """
        schema = self._get_schema(op_invoke_info.func)

        if schema is not None:
            try:
                shape = schema.extract_shape_from_op(op_invoke_info)
                query_result = self.db.query(schema.name, shape, self.query_mode)
                return PerformanceModel.Result(
                    execution_time_s=query_result.latency_us * 1e-6,
                    statistics={
                        "source": query_result.source.name,
                        "confidence": query_result.confidence,
                    }
                )
            except Exception:
                pass  # 闄嶇骇鑷宠В鏋愭ā鍨?

        return self.fallback.process_op(op_invoke_info)

    def get_classifiers(self) -> List[PerformanceModel.OpClassifier]:
        return self.fallback.get_classifiers()
```

### 4.2 Runtime 闆嗘垚
`Runtime` 绫绘棤闇€浠讳綍淇敼銆傚畠宸叉敮鎸佹帴鏀?`PerformanceModel` 瀹炰緥鍒楄〃锛?

```python
# tensor_cast/runtime.py:41-56锛堢幇鏈変唬鐮侊紝鏃犻渶鏀瑰姩锛?
class Runtime(TorchDispatchMode):
    def __init__(
        self,
        perf_models: Union[PerformanceModel, List[PerformanceModel]],
        device_profile: DeviceProfile,
        memory_tracker: Optional[MemoryTracker] = None,
    ):
        self.perf_models = [
            CachingPerformanceModel(m) if not isinstance(m, CachingPerformanceModel) else m
            for m in (perf_models if isinstance(perf_models, list) else [perf_models])
        ]
```

浣跨敤 Profiling 妯″瀷鍙渶灏嗗叾浣滀负 perf_model 浼犲叆锛?

```python
# 鍦?model_runner.py 鎴?config_resolver.py 涓細
perf_model = ProfilingPerformanceModel(device_profile, vllm_version="0.14.0")
runtime = Runtime(perf_models=perf_model, device_profile=device_profile)
```

### 4.3 CLI 鎺ュ彛
鍦?`tensor_cast/scripts/text_generate.py` 涓柊澧炰互涓嬪弬鏁帮細

```python
parser.add_argument("--performance-model", choices=["analytic", "profiling", "empirical"],
                    default="analytic", help="鎬ц兘妯″瀷绫诲瀷")
parser.add_argument("--database-mode", choices=["exact", "interpolate", "hybrid", "roofline"],
                    default="hybrid", help="鏁版嵁搴撴煡璇㈡ā寮?)
parser.add_argument("--database-version", type=str, default="latest",
                    help="Profiling 鏁版嵁搴撶増鏈彿")
```

### 4.4 鏁版嵁娴?
```plain
鐢ㄦ埛 CLI                     TensorCast                    鏁版嵁搴?
鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€                     鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€                    鈹€鈹€鈹€鈹€鈹€鈹€
text_generate.py
  --performance-model profiling
  --database-version 0.14.0
       鈹?
       鈻?
  ModelRunner 鍒涘缓
  ProfilingPerformanceModel
       鈹?
       鈻?
  Runtime.__torch_dispatch__
  鎷︽埅姣忎釜绠楀瓙璋冪敤
       鈹?
       鈻?
  OpInvokeInfo(func, args, kwargs, out)
       鈹?
       鈻?
  ProfilingPerformanceModel.process_op()
       鈹?
       鈹溾攢鈹€鈻?鍖归厤绠楀瓙鍒?OperatorSchema
       鈹?   锛堜緥濡?aten.mm 鈫?GEMMSchema锛?
       鈹?
       鈹溾攢鈹€鈻?schema.extract_shape_from_op()
       鈹?   鈫?{m: 136, n: 4096, k: 5120, quant: "int8"}
       鈹?
       鈹溾攢鈹€鈻?db.query("gemm", shape, mode=HYBRID)
       鈹?        鈹?
       鈹?        鈹溾攢鈹€鈻?gemm.parquet 涓簿纭尮閰嶏紵 鈫?杩斿洖
       鈹?        鈹溾攢鈹€鈻?浠庨偦杩戠偣鎻掑€硷紵 鈫?杩斿洖
       鈹?        鈹斺攢鈹€鈻?Roofline 鍏滃簳 鈫?杩斿洖
       鈹?
       鈹斺攢鈹€鈻?杩斿洖 PerformanceModel.Result(execution_time_s=...)
```

---

## 5. 鑷姩鍖?Profiling 娴佹按绾?
### 5.1 涓ょ骇閲囬泦绛栫暐
#### Level 1: 鍏ㄦā鍨?Profiling锛堣幏鍙栧熀鍑嗙湡鍊硷級
鍚敤鏄囪吘 Profiler 杩愯 VLLM 鎺ㄧ悊锛屾崟鑾风湡瀹炶瀺鍚堝唴鏍哥殑鎵ц鑰楁椂銆?

**瑙﹀彂鏂瑰紡**锛堝弬鑰冨疄闄?Profiling 鑴氭湰锛夛細

```bash
# 鐜鍙橀噺閰嶇疆
export VLLM_TORCH_PROFILER_DIR=/path/to/output
export PROFILING_SAVE_PATH=/path/to/output

# 鍚姩 VLLM 鏈嶅姟锛堥珮寮€閿€锛氭ā鍨嬪姞杞介渶鏁板垎閽燂級
vllm serve $MODEL --tensor-parallel-size $TP --dtype bfloat16 ...

# 鍙戦€佸彈鎺ц姹傝繘琛屽熀鍑嗘祴璇曪紙浣庡紑閿€锛氭瘡杞粎鏁扮锛?
vllm bench serve --profile \
    --dataset-name random \
    --random-input-len $INPUT_LEN \
    --random-output-len $OUTPUT_LEN \
    --max-concurrency $BATCH_SIZE
```

**鏁堢巼浼樺寲**锛氭瘡涓湇鍔″疄渚嬪彧闇€鍚姩涓€娆★紝闅忓悗鍙戦€佸杞笉鍚?Shape 閰嶇疆鐨勫熀鍑嗘祴璇曡姹傘€?

**鍗曞疄渚嬪唴 Shape 閬嶅巻鏂规**锛?

```python
# Decode 鍦烘櫙锛坬uery_len=1锛?
for batch in [1, 8, 16, 32, 64, 128, 256]:
    bench(input_len=1, output_len=1, concurrency=batch)

# Prefill 鍦烘櫙
for isl in [256, 512, 1024, 2048, 4096, 8192]:
    for batch in [1, 8, 32, 64, 128]:
        bench(input_len=isl, output_len=1, concurrency=batch)
```

**闇€瑕佺嫭绔嬫湇鍔″疄渚嬬殑閰嶇疆**锛?

```python
SERVER_CONFIGS = [
    {"model": "Qwen3-32B", "tp": 4, "quant": "none"},
    {"model": "Qwen3-32B", "tp": 8, "quant": "W8A8"},
    {"model": "Qwen3-32B", "tp": 16, "quant": "W4A8"},
    {"model": "DeepSeek-V3", "tp": 4, "dp": 8, "ep": True, "quant": "W8A8"},
    {"model": "DeepSeek-V3", "tp": 4, "dp": 8, "ep": True, "quant": "W4A8"},
]
```

**杈撳嚭瑙ｆ瀽**锛?

```python
# 瑙ｆ瀽 ASCEND_PROFILER_OUTPUT/kernel_details.csv
# 鍒楀瓧娈碉細Name, Duration(us), Input Shapes, cube_utilization(%)
# 浠?FusedInferAttentionScore 浣滀负姝ラ杈圭晫杩涜鍒嗙粍
# 鎻愬彇姣忎釜 Shape 閰嶇疆涓嬪悇绠楀瓙鐨勮仛鍚堣€楁椂
```

#### Level 2: 鍗曠畻瀛愬井鍩哄噯娴嬭瘯锛圫hape 缃戞牸濉厖锛?
閫氳繃 `torch_npu` 鐩存帴瀵瑰崟涓畻瀛愯繘琛岀粏绮掑害 Shape 瑕嗙洊娴嬭瘯锛?

```python
# scripts/collect_microbench.py
import torch, torch_npu

def benchmark_matmul(m, n, k, dtype, warmup=5, runs=20):
    a = torch.randn(m, k, dtype=dtype, device='npu')
    b = torch.randn(k, n, dtype=dtype, device='npu')
    # 棰勭儹
    for _ in range(warmup):
        torch.mm(a, b)
    torch.npu.synchronize()
    # 璁℃椂
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()
    for _ in range(runs):
        torch.mm(a, b)
    end_event.record()
    torch.npu.synchronize()
    return start_event.elapsed_time(end_event) / runs * 1000  # 杩斿洖寰
```

#### Level 1 涓?Level 2 瀵规瘮
| 缁村害 | Level 1锛堝叏妯″瀷 Profiling锛?| Level 2锛堝井鍩哄噯娴嬭瘯锛?|
| --- | --- | --- |
| 铻嶅悎鍐呮牳鎹曡幏 | 瀹屾暣鎹曡幏锛堝熀鍑嗙湡鍊硷級 | 閮ㄥ垎鎹曡幏 |
| Shape 鎺у埗 | 闂存帴鎺у埗锛坆atch銆乻eq_len锛?| 鐩存帴鎺у埗锛坢銆乶銆乲锛?|
| 鎵ц寮€閿€ | 楂橈紙鏈嶅姟鍚姩鑰楁椂闀匡級 | 浣庯紙浠呭唴鏍告墽琛岋級 |
| 瑕嗙洊鑼冨洿 | 姣忎釜妯″瀷绾?100 绉嶉厤缃?| 姣忎釜绠楀瓙绾?2500 绉?Shape |
| 閫傜敤鍦烘櫙 | 楠岃瘉涓庢牎鍑?| Shape 缃戞牸澶ц妯″～鍏?|


**鏍″噯鏂规硶**锛氬湪 Shape 閲嶅彔鍖哄煙瀵规瘮 Level 1 鍜?Level 2 鐨勬暟鎹紝璁＄畻姣忕被绠楀瓙鐨勪慨姝ｇ郴鏁帮紝浣?Level 2 鏁版嵁瀵归綈瀹為檯閮ㄧ讲鍦烘櫙銆?

### 5.2 Profiling 杈撳嚭瑙ｆ瀽鍣?
```python
# scripts/parse_ascend_output.py

class AscendProfilerParser:
    def parse_kernel_details(self, csv_path: Path) -> pd.DataFrame:
        """瑙ｆ瀽 kernel_details.csv 鈫?缁撴瀯鍖?DataFrame"""
        # 澶勭悊鐏垫椿鐨勫垪鍚嶆牸寮忓拰缂哄け瀛楁
        # 杩斿洖: name, duration_us, input_shapes, data_types,
        #       aicore_time_us, aiv_time_us, cube_utilization_pct

    def extract_single_step(self, df: pd.DataFrame) -> pd.DataFrame:
        """浠?Attention 鍐呮牳涓鸿竟鐣屾彁鍙栧崟涓?Decode/Prefill 姝ラ"""
        # 澶嶇敤 profiling_compare 涓殑閫昏緫: kernel_details_parser.py

    def map_to_schema(self, kernel_name: str, input_shapes: str) -> Tuple[str, Dict]:
        """灏?VLLM 鍐呮牳鍚嶆槧灏勫埌 (schema_name, shape_dict)"""
        # "MatMulV2" + "136,4096; 4096,4096" 鈫?("gemm", {m:136, k:4096, n:4096})
        # 浣跨敤鐗堟湰鐩稿叧鐨?YAML 鏄犲皠閰嶇疆
```

### 5.3 鏂版ā鍨嬬畻瀛愬彂鐜版満鍒?
```python
# scripts/operator coverage check tool

def discover_operators(profiling_output: Path, mapping_yaml: Path) -> Dict:
    """鍙戠幇 Profiling 涓瓨鍦ㄤ絾褰撳墠鏄犲皠琛ㄤ腑缂哄け鐨勭畻瀛?""
    parser = AscendProfilerParser()
    kernels = parser.parse_kernel_details(profiling_output / "kernel_details.csv")

    known_ops = load_yaml_mappings(mapping_yaml)
    unknown = []

    for kernel_name in kernels["name"].unique():
        if not any(kernel_name in schema_kernels for schema_kernels in known_ops.values()):
            # 鍩轰簬鍐呮牳鍚嶇О妯″紡鑷姩鍒嗙被
            suggested_schema = auto_classify(kernel_name)
            unknown.append({"kernel": kernel_name, "suggested_schema": suggested_schema})

    return {"known": len(kernels) - len(unknown), "unknown": unknown}
```

---

## 6. 鍏ㄩ潰绠楀瓙瑕嗙洊绛栫暐
### 6.1 绠楀瓙鍒嗙骇
| 灞傜骇 | 鍒ゅ畾鏍囧噯 | 澶勭悊鏂瑰紡 | 绠楀瓙鏁伴噺 |
| --- | --- | --- | --- |
| **Tier 1** | 鎵ц鑰楁椂鍗犳瘮 >2% | 蹇呴』浣跨敤瀹屾暣 Shape 缃戞牸杩涜 Profiling | 绾?11 涓?|
| **Tier 2** | 鎵ц鑰楁椂鍗犳瘮 0.5-2% | 浣跨敤绮剧畝 Shape 缃戞牸杩涜 Profiling | 绾?8 涓?|
| **Tier 3** | 鎵ц鑰楁椂鍗犳瘮 <0.5% | 閲囩敤 Roofline 鍏滃簳浼扮畻 | 60+ 涓?|


### 6.2 Tier 1 瀹屾暣绠楀瓙鍒楄〃
| VLLM 鍐呮牳鍚嶇О | 鏁版嵁搴?Schema | Qwen3 鍗犳瘮 | DSV3 鍗犳瘮 |
| --- | --- | --- | --- |
| MatMulV2 | gemm | 42.4% | - |
| GroupedMatmul | gemm | - | 20.9% |
| QuantBatchMatmulV3 | gemm | - | 16.9% |
| FusedInferAttentionScore | attention | 18.2% | 18.5% |
| TensorMove | memory_move | 10.7% | ~3% |
| AddRmsNorm/InplaceAddRmsNorm | normalization | 8.0% | 2.0% |
| split_qkv_rmsnorm_rope_kernel | fused | 5.2% | - |
| SwiGlu/DequantSwigluQuant | fused | 4.8% | 2.7% |
| ReshapeAndCacheNdKernel | cache | 3.3% | - |
| MoeDistributeDispatch/Combine | moe_routing | - | 11.8% |
| AscendQuantV2/DynamicQuant | quantization | - | 5.5% |
| TransposeBatchMatMul | gemm | - | 4.2% |
| InterleaveRope | rope | - | 2.8% |


### 6.3 Shape 缃戞牸锛堟贩鍚堢瓥鐣ワ細妯″瀷椹卞姩 + 閫氱敤缃戞牸锛?
**绗竴姝?*锛氫粠 HuggingFace 妯″瀷閰嶇疆鑷姩鎻愬彇缁村害鍙傛暟锛?

```python
PRIORITY_MODELS = [
    "Qwen/Qwen3-32B", "Qwen/Qwen3-235B-A22B",
    "deepseek-ai/DeepSeek-V3", "moonshotai/Kimi-K2-Instruct",
    "meta-llama/Llama-3.1-70B", "meta-llama/Llama-3.1-405B",
    "Qwen/Qwen2.5-72B",
]
# 鑷姩鎻愬彇锛歨idden_size銆乶um_heads銆乶um_kv_heads銆乭ead_dim銆?
#           intermediate_size銆乶um_experts銆乲v_lora_rank 绛?
```

**绗簩姝?*锛氫互閫氱敤鐨?2 鐨勫箓娆＄綉鏍艰ˉ鍏呮ā鍨嬮棿鐨勬彃鍊奸棿闅欍€?

**棰勪及 Shape 鎬婚噺**锛氭瘡涓増鏈害 5,000 涓?

---

## 7. 寮€鍙戣鍒?
### 闃舵涓€锛氭牳蹇冨熀纭€璁炬柦
+ 鍒涘缓 `tensor_cast/perf_database/` 鍖呯粨鏋?
+ 瀹炵幇 `OperatorSchema` 鍩虹被鍙婃敞鍐岃〃鏈哄埗
+ 瀹炵幇 `GEMMSchema` 鍜?`AttentionSchema`
+ 瀹炵幇 `PerfDatabase`锛圥arquet 鍔犺浇锛?
+ 瀹炵幇绮剧‘鍖归厤鏌ヨ

### 闃舵浜岋細鎻掑€间笌闄嶇骇绛栫暐
+ 瀹炵幇 `QueryEngine`锛堝熀浜?scipy 鐨勫缁存彃鍊硷級
+ 娣诲姞缃俊搴﹁瘎鍒嗘満鍒?
+ 闆嗘垚 Roofline 鍏滃簳浼扮畻
+ 瀹炵幇鐗堟湰瑙ｆ瀽閫昏緫锛坄VersionManager`锛?

### 闃舵涓夛細鏁版嵁閲囬泦娴佹按绾?
+ 瀹炵幇 `generate_shape_grid.py`锛堟ā鍨嬮┍鍔?+ 閫氱敤缃戞牸锛?
+ 瀹炵幇 `collect_full_model.py`锛圠evel 1 缂栨帓鑴氭湰锛?
+ 瀹炵幇 `collect_microbench.py`锛圠evel 2 torch_npu 寰熀鍑嗘祴璇曪級
+ 瀹炵幇 `parse_ascend_output.py`锛坘ernel_details.csv 瑙ｆ瀽鍣級
+ 瀹炵幇 `calibrate.py`锛圠1 涓?L2 鏍″噯瀵归綈锛?
+ 瀹屾垚 GEMM 鍜?Attention 鍦?A3 涓婄殑鍒濆鏁版嵁閲囬泦

### 闃舵鍥涳細TensorCast 闆嗘垚
+ 瀹炵幇 `ProfilingPerformanceModel`
+ 鍦?`text_generate.py` 涓坊鍔?CLI 鍙傛暟
+ 鍦?`model_runner.py` / `config_resolver.py` 涓帴鍏?
+ 浣跨敤 Qwen3-32B 鍜?DeepSeek-V3 杩涜绔埌绔祴璇?

### 闃舵浜旓細鎵╁睍绠楀瓙涓庣簿搴﹂獙璇?
+ 鏂板 MoE銆佸綊涓€鍖栥€佽瀺鍚堢畻瀛愩€佺紦瀛樼畻瀛愮殑 Schema
+ 涓庡疄闄?VLLM Profiling 杩涜绔埌绔簿搴﹂獙璇?
+ 娴嬭瘯 PD 鑱氬悎涓庡垎绂伙紙Aggregation/Disaggregation锛夊満鏅?
+ 瀹屽杽鏂版ā鍨嬬畻瀛愬彂鐜版祦姘寸嚎
+ 缂栧啓浣跨敤鏂囨。

### 楠岃瘉鏍囧噯
| 鎸囨爣 | 鐩爣鍊?|
| --- | --- |
| 绔埌绔€楁椂璇樊 | 涓庡疄闄?VLLM 瀵规瘮 <15% |
| 鍗曠畻瀛愯宸紙宸插尮閰嶇畻瀛愶級 | <20% |
| 鏃堕棿瑕嗙洊鐜?| 瑕嗙洊 VLLM 鎵ц鏃堕棿鐨?>90% |


### 娴嬭瘯鐢ㄤ緥
1. Qwen3-32B Prefill锛?36 璇锋眰 x 4096 tokens锛孴P=16
2. Qwen3-32B Decode锛?36 璇锋眰 x 1 token锛宑ontext=4096锛孴P=16
3. DeepSeek-V3 Prefill锛?8 璇锋眰 x 4096 tokens锛孴P=4锛孌P=8锛孍P
4. DeepSeek-V3 Decode锛?8 璇锋眰 x 1 token锛宑ontext=4096锛孴P=4锛孌P=8锛孍P
5. PD 鑱氬悎/鍒嗙妯″紡锛氬垎鍒祴璇曚袱绉嶆ā寮?

---

## 8. 渚濊禆椤?
**鏂板**锛歚scipy`锛堟彃鍊硷級銆乣pyarrow`锛圥arquet锛夈€乣packaging`锛堢増鏈В鏋愶級  
**鐜版湁**锛歚pandas`銆乣numpy`銆乣pyyaml`銆乣torch`

---

## 9. 鍙傝€冭祫鏂?
+ [vLLM Ascend GitHub](https://github.com/vllm-project/vllm-ascend)
+ [vLLM Ascend 鍙戝竷璇存槑](https://docs.vllm.ai/projects/ascend/en/main/user_guide/release_notes.html)
+ [vLLM Ascend Profiling 鎸囧崡](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/service_profiling_guide.html)
+ [鍗庝负鏄囪吘 Profiler 鏂囨。](https://support.huaweicloud.com/intl/en-us/bestpractice-modelarts/modelarts_llm_infer_5906034.html)
+ [Intel NPU Cost Model](https://github.com/intel/npu-nn-cost-model)
+ NVIDIA AI Configurator锛堝唴閮ㄥ弬鑰冿紝浣嶄簬 `/home/horacehxw/Projects/aiconfigurator`锛?
+ 鐜版湁 Profiling 瀵规瘮宸ュ叿(寮€鍙戜腑)锛歚tensor_cast/scripts/profiling_comparison/`


