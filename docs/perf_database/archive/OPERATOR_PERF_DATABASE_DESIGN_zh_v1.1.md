# TensorCast 绠楀瓙鎬ц兘鏁版嵁搴擄細鎶€鏈璁℃枃妗?

**鐗堟湰**: 1.1
**鏃ユ湡**: 2026.2.12
**鑼冨洿**: 闈㈠悜 LLM 浠跨湡鐨勫彲鎵╁睍 Profiling Cost Model锛屼笉缁戝畾鍏蜂綋绠楀姏鍗★紝鏀寔鍩轰簬瀹炴祴 Profiling 鏁版嵁鐨勭畻瀛愭€ц兘浼扮畻銆?
**鍒濇湡鐩爣妯″瀷**: DeepSeek-V3銆丵wen3-32B

浣滆€咃細HXW
瀹℃牳浜猴細GJ

---

## 1. 鍔熻兘姒傝堪

### 1.1 鐩爣

涓?TensorCast 浠跨湡鍣ㄦ瀯寤哄熀浜庡疄娴嬫暟鎹殑绠楀瓙鎬ц兘浼扮畻绯荤粺銆傛柊澧炵嫭绔嬬殑 `ProfilingPerformanceModel`锛屼笌鐜版湁鐨?`EmpiricalPerformanceModel`锛圝IT 鍩哄噯娴嬭瘯锛夊拰 `AnalyticPerformanceModel`锛圧oofline锛夊苟鍒楋紝浣滀负鐢ㄦ埛鍙€夌殑绗笁绉嶆€ц兘妯″瀷銆?

`ProfilingPerformanceModel` 鍜?`EmpiricalPerformanceModel`鍙叡浜?**`PerfDatabase` 鏁版嵁灞?* 浣滀负鏍囧噯鍖栫殑绠楀瓙鎬ц兘鏁版嵁瀛樺偍涓庢煡璇㈡帴鍙ｃ€?

### 1.2 鏍稿績鍔熻兘锛堟湰鏂规鑼冨洿锛?

1. **绠楀瓙鎬ц兘鏁版嵁搴擄紙PerfDatabase锛夛紙鏂板锛?*锛氬畾涔夋爣鍑嗗寲鏁版嵁鏍煎紡鍜屾煡璇㈡帴鍙ｏ紝鏀寔鎸夌畻瀛愮被鍨嬨€丼hape 缁村害銆侀噺鍖栨ā寮忚繘琛屾€ц兘鏌ヨ锛屼笌鏁版嵁鏉ユ簮瑙ｈ€︼紙鏀寔 micro-benchmark銆佸叏妯″瀷 Profiling 绛夊绉嶆暟鎹簮锛?
2. **ProfilingPerformanceModel锛堟柊澧烇級**锛氱嫭绔嬬殑 `PerformanceModel` 瀛愮被锛屼粠棰勬瀯寤虹殑 `PerfDatabase` 涓煡璇㈢畻瀛愯€楁椂锛涘鏈敹褰曠畻瀛愬唴閮ㄥ厹搴曞洖閫€鑷?`AnalyticPerformanceModel`
3. **EmpiricalPerformanceModel 澧炲己**锛氬彲閫夋帴鍏?`PerfDatabase` 浣滀负 JIT benchmark 鐨勬寔涔呭寲缂撳瓨锛屽疄鐜拌法 session 鐨勭粨鏋滃鐢?
4. **鏌ヨ寮曟搸**锛氬疄鐜扮簿纭尮閰?鈫?鎻掑€间及绠?鈫?澶栨帹浼扮畻鐨勫绾ф煡璇㈢瓥鐣?
5. **鏁版嵁閲囬泦娴佹按绾?*锛堢嫭绔嬪瓙绯荤粺锛屼笉鍦?TensorCast 鍖呭唴锛夛細鑷姩鍖栨墽琛屽叏妯″瀷 Profiling 鍜屽崟绠楀瓙寰熀鍑嗘祴璇曪紝瑙ｆ瀽杈撳嚭骞舵瀯寤烘€ц兘鏁版嵁搴?

### 1.3 涓嶅湪鏈柟妗堣寖鍥村唴锛堝缓璁悗缁敮鎸侊級

- **CompositePerformanceModel 椤跺眰璋冨害鍣?*锛氱粺涓€缂栨帓澶氱 PerformanceModel锛屽疄鐜板彲閰嶇疆鐨勯檷绾ф垨鑰呯粍鍚堢瓥鐣ワ紙璇﹁绗?9 鑺傦級
- **璺ㄧ‖浠舵硾鍖?*锛氬綋鍓嶄粎鏀寔鏄囪吘 A3锛屽叾浠栫‖浠堕渶鐙珛閲囬泦鏁版嵁
- **鑷姩鍖栨寔缁泦鎴?*锛氶殢 VLLM-Ascend / CANN 鐗堟湰鍙戝竷鑷姩瑙﹀彂鏁版嵁閲囬泦

### 1.4 鍒濇湡鐩爣 锛堝緟瀵归綈锛?

- **鐩爣妯″瀷**锛欴eepSeek-V3銆丵wen3-32B
- **鐩爣纭欢**锛欰tlas 800 A3锛?52T锛?28G DIE锛?
- **鐩爣鍚庣**锛歷llm-0.13.0 (鍐呴儴闀滃儚)
- **绮惧害鐩爣**锛氱鍒扮浠跨湡璇樊 <15%锛堝姣斿疄闄?VLLM Profiling锛?
- **浜や粯鏃堕棿**锛歈1锛?026.3.20锛夊畬鍏ㄨ窇閫氬苟瀹屾垚鍒濆鏁版嵁閲囬泦鍜岄泦鎴愭祴璇?

---

## 2. 鎶€鏈垎鏋?

### 2.1 闂闄堣堪

TensorCast 褰撳墠閲囩敤**鍩轰簬 Roofline 鐨勮В鏋愭ā鍨?*锛坄AnalyticPerformanceModel`锛変及绠楃畻瀛愭墽琛岃€楁椂銆傝妯″瀷鍩轰簬娴偣杩愮畻閲忥紙FLOPs锛変笌璁垮瓨瀛楄妭鏁拌绠?`max(璁＄畻鑰楁椂, 璁垮瓨鑰楁椂)`锛屼富瑕佺敤浣滅悊璁烘€ц兘涓婄嚎璇勪及锛屽疄闄呰€楁椂鍙兘瀛樺湪鍋忓樊銆?

`EmpiricalPerformanceModel`锛坄tensor_cast/performance_model/empirical.py`锛夊凡鎻愪緵瀵规帴瀹炴祴鏁版嵁鐨勫垵姝ユ鏋讹紝褰撳墠浠呴€氳繃 `OpBenchmark` 绫诲湪鐗╃悊璁惧涓?JIT 鎵ц骞惰鏃?

**鏈柟妗堢殑鏍稿績鎬濊矾**锛氭柊澧炵嫭绔嬬殑 `ProfilingPerformanceModel`锛屼笓鑱屼粠棰勬瀯寤虹殑鎬ц兘鏁版嵁搴撲腑鏌ヨ绠楀瓙鑰楁椂銆傚悓鏃舵娊鍙?`PerfDatabase` 浣滀负鍏变韩鏁版嵁灞傦紝渚?`ProfilingPerformanceModel`锛堝彧璇绘煡璇級鍜?`EmpiricalPerformanceModel`锛堣鍐欑紦瀛橈級鍏卞悓浣跨敤銆備笁绉?PerformanceModel 淇濇寔鐙珛锛岀敤鎴烽€氳繃 CLI 閰嶇疆閫夋嫨銆?

**鍋忓樊鏍瑰洜鍒嗘瀽**锛?

| 鍋忓樊鏉ユ簮 | 鍏蜂綋鎻忚堪 | 褰卞搷绋嬪害 |
|---------|---------|---------|
| **绠楀瓙铻嶅悎** | VLLM 灏嗗涓畻瀛愯瀺鍚堜负鍗曚竴鍐呮牳鎵ц锛堝 `DequantSwigluQuant`銆乣AddRmsNorm`锛?| 10-30% 鏃堕棿宸紓 |
| **纭欢鍒╃敤鐜?* | 瀹為檯 Cube 鍒╃敤鐜囩害 44-68%锛岃€岄潪 Roofline 鍋囪鐨?100% | 楂樹及璁＄畻瀵嗛泦鍨嬬畻瀛?|
| **Shape 鐩稿叧寮€閿€** | 灏?batch 鍦烘櫙涓嬪唴鏍稿惎鍔ㄥ紑閿€鏈缓妯?| 浣庝及 Decode 闃舵鑰楁椂 |
| **鐗堟湰鐩稿叧浼樺寲** | VLLM-Ascend 鍚勭増鏈紩鍏ヤ笉鍚岀殑铻嶅悎鍐呮牳 | 妯″瀷鍦ㄧ増鏈凯浠ｄ腑澶辨晥 |

### 2.2 Profiling 鏁版嵁鍒嗘瀽

鍩轰簬瀹為檯鏄囪吘 Profiler 杈撳嚭锛坄kernel_details.csv`銆乣op_statistic.csv`锛夛細

> **娉ㄦ剰**锛氫互涓嬪崰姣斾粎缁熻**璁＄畻鍐呮牳**锛屼笉鍚€氫俊鍐呮牳锛堝 AllReduce锛夈€備笌绗?7.2 鑺傚寘鍚€氫俊鐨勫叏閲忓唴鏍稿崰姣斾笉鍚屻€?

**Qwen3-32B锛堝叡 41 涓嫭绔嬬畻瀛愶級**锛?
| 鏍稿績绠楀瓙 | 鑰楁椂鍗犳瘮 | 鏍稿績绫诲瀷 |
|---------|---------|---------|
| MatMulV2 | 42.4% | AI_CORE |
| FusedInferAttentionScore | 18.2% | MIX_AIC |
| TensorMove | 10.7% | AI_VECTOR_CORE |
| AddRmsNorm | 8.0% | AI_VECTOR_CORE |
| split_qkv_rmsnorm_rope_kernel | 5.2% | MIX_AIC |
| SwiGlu | 4.8% | AI_VECTOR_CORE |

**DeepSeek-V3锛堝叡 39 涓嫭绔嬬畻瀛愶紝鍚?MoE 鐗规湁绠楀瓙锛?*锛?
| 鏍稿績绠楀瓙 | 鑰楁椂鍗犳瘮 | 鏍稿績绫诲瀷 |
|---------|---------|---------|
| GroupedMatmul | 20.9% | AI_CORE |
| FusedInferAttentionScore | 18.5% | MIX_AIC |
| QuantBatchMatmulV3 | 16.9% | AI_CORE |
| MoeDistributeDispatch/Combine | 11.8% | MIX_AIC |
| AscendQuantV2 | 4.4% | AI_VECTOR_CORE |
| TransposeBatchMatMul | 4.2% | AI_CORE |

**鍏抽敭鍙戠幇**锛氱害 15 涓畻瀛愯础鐚簡瓒呰繃 95% 鐨勬墽琛屾椂闂淬€傚叾浣欑害 65 涓畻瀛愶紙绠楁湳銆佺储寮曘€侀€昏緫杩愮畻锛夎础鐚笉瓒?5%锛屽彲閲囩敤 Roofline 鍏滃簳浼扮畻銆?

### 2.3 鐗堟湰褰卞搷鍒嗘瀽

VLLM-Ascend 涓嶅悓鐗堟湰瀵硅瀺鍚堝唴鏍哥殑绉嶇被鍙婂叾鎬ц兘鏈夋樉钁楀奖鍝嶃€備互涓嬩俊鎭潵鑷?[vLLM-Ascend 鍙戝竷璇存槑](https://docs.vllm.ai/projects/ascend/en/main/user_guide/release_notes.html) 鍜?[GitHub Releases](https://github.com/vllm-project/vllm-ascend/releases)锛?

| 鐗堟湰 | 鍙戝竷鏃ユ湡 | 鍏抽敭铻嶅悎鍐呮牳鍙樺寲 |
|-----|---------|---------------|
| v0.13.0锛堟渶鏂扮ǔ瀹氱増锛?| 2026.02.06 | AddRmsnormQuant 铻嶅悎+SP 鏀寔銆乫used matmul/reduce-scatter 鍐呮牳銆乀riton chunk_gated_delta_rule锛圦wen3-Next锛夈€乀riton RoPE 浼樺寲 |
| v0.14.0rc1锛堥鍙戝竷锛?| 2026.01.26 | MatMul-AllReduce-RMSNorm 铻嶅悎 Pass锛堥粯璁ゅ叧闂級銆?10P 鍩虹鏀寔 |
| v0.12.0rc1 | 2025.12.13 | 澶ч噺 Triton 鍐呮牳锛圦wen3-Next, DeepSeek 3.2锛夈€丗ull Decode-Only Graph Mode锛堝疄楠屾€э級銆乄4A4 閲忓寲鏀寔 |
| v0.11.0 | 2025.12.16 | W8A16 閲忓寲銆丗ull Graph Mode (ACLGraph) + GQA銆佸 Token 棰勬祴 + Chunked Prefill |

CANN 鐗堟湰鍚屾牱褰卞搷鍐呮牳鎵ц鏁堢巼锛堝 CANN 8.5 閽堝 FIA 绠楀瓙鍋氫簡 flash decoding 浼樺寲锛夛紝涓よ€呯増鏈彿鍧囬渶绾冲叆鏁版嵁搴撶増鏈拷韪€?

### 2.4 鐜版湁鍩虹璁炬柦

**`EmpiricalPerformanceModel`**锛堝凡瀹炵幇锛屼綅浜?`tensor_cast/performance_model/empirical.py`锛夛細褰撳墠浠?20 琛屼唬鐮侊紝缁ф壙 `PerformanceModel`锛岄€氳繃 `OpBenchmark` 鍦ㄧ墿鐞嗚澶囦笂 JIT 鎵ц骞惰鏃躲€?

`OpBenchmark`锛坄op_benchmark.py`锛夋彁渚涗簡鍏抽敭鎵╁睍鐐癸細
- `OpBenchmarkBase` 鎶借薄鍩虹被瀹氫箟浜?`benchmark()` 鎺ュ彛
- 鏀寔 meta tensor 鈫?real tensor 鐨勮嚜鍔ㄨ浆鎹?
- `register_op_impl` 娉ㄥ唽琛ㄤ负 TensorCast 鑷畾涔夌畻瀛愭彁渚涜澶囩壒瀹氬疄鐜?

### 2.5 AI Configurator 鍙傝€?

鍙傝€?[AI Configurator](https://github.com/ai-dynamo/aiconfigurator) 椤圭洰锛圢VIDIA 鐨?LLM 鎺ㄧ悊鎬ц兘棰勪及宸ュ叿锛夋彁鐐肩殑鏍稿績璁捐妯″紡锛?

- **宓屽瀛楀吀绱㈠紩 + 娣峰悎鎻掑€?*锛氬宸?Profiling 鐨?Shape 瀹炵幇 O(1) 绮剧‘鏌ユ壘锛堝 `gemm_data[quant_mode][m][n][k]`锛夛紱閲囩敤 2D+1D 娣峰悎鎻掑€肩瓥鐣ワ紙鍏堝涓や釜缁村害鍋氬弻绾挎€ф彃鍊硷紝鍐嶅绗笁涓淮搴﹀仛 1D 鎻掑€硷級锛屾敮鎸?sqrt 鍙樻崲澶勭悊 O(n虏) 澶嶆潅搴︾殑 Attention 搴忓垪缁村害
- **DatabaseMode 闄嶇骇绛栫暐**锛歋ILICON锛堜粎瀹炴祴鏁版嵁锛夆啋 HYBRID锛堝疄娴嬩紭鍏堬紝鍥為€€缁忛獙鍊硷級鈫?EMPIRICAL锛圧oofline 脳 鏁堢巼绯绘暟锛夆啋 SOL锛堢函鐞嗚宄板€硷級
- **CSV 瀛樺偍 + 寤惰繜鍔犺浇**锛氬彲璇绘€у己銆丟it-friendly锛屾寜 `systems/{device}/{backend}/{version}/` 鍒嗙洰褰曞瓨鍌紱鍏ㄥ眬 DB 缂撳瓨 + LRU 鏌ヨ缂撳瓨
- **SOL 鏁版嵁鏍℃**锛氱敤鐞嗚涓嬬晫鏍℃寮傚父娴嬮噺鍊硷紝纭繚 `measured >= SOL`

---

## 3. 绯荤粺鏋舵瀯

### 3.1 鏁翠綋鏋舵瀯

```
鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
鈹?                          TensorCast Runtime                                鈹?
鈹?                                                                            鈹?
鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?    鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹? 鈹?
鈹? 鈹? Runtime            鈹?    鈹? 鐢ㄦ埛鍙厤缃€夋嫨浠ヤ笅浠讳竴 PerformanceModel   鈹? 鈹?
鈹? 鈹? (TorchDispatchMode)鈹傗攢鈹€鈹€鈹€鈻垛攤                                            鈹? 鈹?
鈹? 鈹?                    鈹?    鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹? 鈹?
鈹? 鈹? 鎷︽埅鎵€鏈夌畻瀛愯皟鐢?  鈹?    鈹? 鈹?ProfilingPerformanceModel锛堟柊澧烇級  鈹?   鈹? 鈹?
鈹? 鈹? 鐢熸垚 OpInvokeInfo  鈹?    鈹? 鈹? PerfDatabase 鍙鏌ヨ             鈹?   鈹? 鈹?
鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?    鈹? 鈹? 鏈懡涓?鈫?fallback (Analytic)      鈹?   鈹? 鈹?
鈹?                             鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹? 鈹?
鈹?                             鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹? 鈹?
鈹?                             鈹? 鈹?EmpiricalPerformanceModel锛堢幇鏈夛級  鈹?   鈹? 鈹?
鈹?                             鈹? 鈹? 鍙€? PerfDatabase 鎸佷箙鍖栫紦瀛?    鈹?   鈹? 鈹?
鈹?                             鈹? 鈹? 鏌ョ紦瀛?鈫?JIT benchmark 鈫?鍐欑紦瀛? 鈹?   鈹? 鈹?
鈹?                             鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹? 鈹?
鈹?                             鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹? 鈹?
鈹?                             鈹? 鈹?AnalyticPerformanceModel锛堢幇鏈夛級   鈹?   鈹? 鈹?
鈹?                             鈹? 鈹? Roofline 鐞嗚妯″瀷                 鈹?   鈹? 鈹?
鈹?                             鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹? 鈹?
鈹?                             鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹? 鈹?
鈹?                                          鈹?                                 鈹?
鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹尖攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
                                            鈹?
                      鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹尖攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
                      鈹?                    鈻?                    鈹?
                      鈹?    PerfDatabase锛堝叡浜暟鎹眰锛?            鈹?
                      鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹?
                      鈹? 鈹? QueryEngine                      鈹?   鈹?
                      鈹? 鈹? 绮剧‘鍖归厤 鈫?鎻掑€?鈫?澶栨帹            鈹?   鈹?
                      鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹?
                      鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹?
                      鈹? 鈹? OperatorSchema 娉ㄥ唽琛?            鈹?   鈹?
                      鈹? 鈹? OperatorKey 鎻愬彇閫昏緫              鈹?   鈹?
                      鈹? 鈹?鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹?鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?鈹屸攢鈹€鈹€鈹?鈹屸攢鈹€鈹€鈹?鈹?   鈹?
                      鈹? 鈹?鈹?GEMM 鈹?鈹侫ttention鈹?鈹侻oE鈹?鈹?..鈹?鈹?   鈹?
                      鈹? 鈹?鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹?鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?鈹斺攢鈹€鈹€鈹?鈹斺攢鈹€鈹€鈹?鈹?   鈹?
                      鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹?
                      鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹?
                      鈹? 鈹? 瀛樺偍灞傦紙鎸夌増鏈垎鐩綍锛孋SV/Parquet锛夆攤    鈹?
                      鈹? 鈹? data/{device}/{backend}/{version}鈹?   鈹?
                      鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹?
                      鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?

鏁版嵁閲囬泦娴佹按绾匡紙绂荤嚎鎵ц锛岀嫭绔嬪瓙绯荤粺锛?
鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
鈹? 鏂规 A锛堥粯璁わ級: 鍏ㄦā鍨?Profiling + 寰熀鍑嗘祴璇?              鈹?
鈹?   VLLM serve + bench 鈫?kernel_details.csv 鈫?瑙ｆ瀽 鈫?鍏ュ簱    鈹?
鈹?   torch_npu 寰熀鍑嗘祴璇?鈫?鐩存帴娴嬮噺 鈫?鍏ュ簱                    鈹?
鈹?   鏍″噯锛氫互 Level 1 鏁版嵁涓哄熀鍑嗭紝鏍℃ Level 2 娴嬮噺鍋忓樊        鈹?
鈹溾攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
鈹? 鏂规 B: 浠跨湡椹卞姩 + 寰熀鍑嗘祴璇?                              鈹?
鈹?   VLLM 鎷夊彇涓€娆?鈫?鑾峰彇绠楀瓙鍒楄〃鍜屽悕绉?                       鈹?
鈹?   TensorCast 浠跨湡 鈫?鏋氫妇涓嶅悓閰嶇疆涓嬬殑 Input Shape            鈹?
鈹?   torch_npu 寰熀鍑嗘祴璇?鈫?鎸夐渶閲囬泦杩欎簺 Shape 鐨勬€ц兘         鈹?
鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
```

### 3.2 妯″潡缁撴瀯

鍏变韩鏁版嵁灞?`PerfDatabase` 鏀惧湪 `tensor_cast/performance_model/perf_database/` 涓嬶紝涓?`analytic.py`銆乣empirical.py` 骞跺垪銆傛柊澧?`profiling.py` 浣滀负 `ProfilingPerformanceModel` 鐨勫疄鐜版枃浠躲€傛暟鎹噰闆嗘祦姘寸嚎浣滀负鐙珛瀛愮郴缁熸斁鍦ㄤ粨搴撻《灞?`perf_database/` 鐩綍銆?

```
tensor_cast/performance_model/
鈹溾攢鈹€ __init__.py                        # PerformanceModel, OpInvokeInfo, CachingPerformanceModel锛堢幇鏈夛級
鈹溾攢鈹€ analytic.py                        # AnalyticPerformanceModel锛堢幇鏈夛級
鈹溾攢鈹€ profiling.py                       # ProfilingPerformanceModel锛堟柊澧烇級
鈹溾攢鈹€ empirical.py                       # EmpiricalPerformanceModel锛堝寮猴細鍙€?cache_db锛?
鈹溾攢鈹€ comm_analytic.py                   # CommAnalyticModel锛堢幇鏈夛級
鈹溾攢鈹€ op_benchmark.py                    # OpBenchmark锛堢幇鏈?JIT 鍩哄噯娴嬭瘯锛?
鈹溾攢鈹€ memory_tracker.py                  # MemoryTracker锛堢幇鏈夛級
鈹斺攢鈹€ perf_database/                     # 鏂板锛氬叡浜暟鎹眰
    鈹溾攢鈹€ __init__.py
    鈹溾攢鈹€ database.py                    # PerfDatabase 涓荤被锛堝姞杞姐€佺紦瀛樸€佹煡璇㈠鎵橈級
    鈹溾攢鈹€ operator_key.py                # OperatorKey锛圤pInvokeInfo 鈫?鏌ヨ key 杞崲锛?
    鈹溾攢鈹€ query_engine.py                # QueryEngine锛堢簿纭尮閰?+ 鎻掑€?+ 澶栨帹锛?
    鈹溾攢鈹€ storage.py                     # CSV/Parquet IO 鍚庣
    鈹斺攢鈹€ schemas/                       # OperatorSchema 娉ㄥ唽琛?
        鈹溾攢鈹€ __init__.py                # 閫氳繃 importlib 鑷姩鍙戠幇
        鈹溾攢鈹€ base.py                    # OperatorSchema 鎶借薄鍩虹被 + 娉ㄥ唽琛?
        鈹溾攢鈹€ matmul.py                  # MatMulSchema, GroupedMatMulSchema, QuantBatchMatMulSchema, TransposeBatchMatMulSchema
        鈹溾攢鈹€ attention.py               # FusedAttentionSchema
        鈹溾攢鈹€ rope.py                    # InterleaveRopeSchema
        鈹溾攢鈹€ moe.py                     # MoeDispatchSchema, MoeCombineSchema, MoeGatingSchema
        鈹溾攢鈹€ communication.py           # AllReduceSchema, AllGatherSchema, AllToAllSchema
        鈹溾攢鈹€ normalization.py           # AddRmsNormSchema
        鈹溾攢鈹€ activation.py              # SwiGluSchema
        鈹溾攢鈹€ quantization.py            # AscendQuantSchema
        鈹溾攢鈹€ cache.py                   # ReshapeAndCacheSchema
        鈹斺攢鈹€ data_movement.py           # TensorMoveSchema

perf_database/                         # 鐙珛瀛愮郴缁燂細鏁版嵁閲囬泦娴佹按绾?
鈹溾攢鈹€ scripts/
鈹?  鈹溾攢鈹€ collect_full_model.py          # Level 1: 鍩轰簬 VLLM 鐨勫叏妯″瀷 Profiling
鈹?  鈹溾攢鈹€ collect_microbench.py          # Level 2: torch_npu 寰熀鍑嗘祴璇?
鈹?  鈹溾攢鈹€ parse_ascend_output.py         # 瑙ｆ瀽 kernel_details.csv
鈹?  鈹溾攢鈹€ generate_shape_grid.py         # 鏍规嵁妯″瀷閰嶇疆 / 浠跨湡缁撴灉鐢熸垚 Shape 缃戞牸
鈹?  鈹溾攢鈹€ calibrate.py                   # L1 涓?L2 鏍″噯瀵归綈
鈹?  鈹溾攢鈹€ database validation tool                    # 鏁版嵁搴撶簿搴﹂獙璇?
鈹?  鈹斺攢鈹€ operator coverage check tool          # 鏂版ā鍨嬬畻瀛愬彂鐜?
鈹溾攢鈹€ mappings/                          # VLLM 鍐呮牳 鈫?TensorCast 绠楀瓙鏄犲皠琛?
鈹?  鈹斺攢鈹€ vllm_ascend/
鈹?      鈹溾攢鈹€ v0.11.yaml
鈹?      鈹溾攢鈹€ v0.12.yaml
鈹?      鈹斺攢鈹€ v0.13.yaml
鈹斺攢鈹€ data/                              # 鎬ц兘鏁版嵁瀛樺偍锛?gitignore锛?
    鈹斺攢鈹€ ATLAS_800_A3_752T_128G_DIE/
        鈹斺攢鈹€ vllm_ascend/{version}/
            鈹溾攢鈹€ metadata.yaml          # 閲囬泦鐜淇℃伅
            鈹溾攢鈹€ matmul.csv
            鈹溾攢鈹€ fused_attention.csv
            鈹斺攢鈹€ ...
```

---

## 4. 鏍稿績妯″潡璁捐

### 4.1 PerfDatabase锛堝叡浜暟鎹眰锛?

鎬ц兘鏁版嵁搴撲綔涓哄叡浜暟鎹眰锛屼负 `ProfilingPerformanceModel`锛堝彧璇绘煡璇級鍜?`EmpiricalPerformanceModel`锛堣鍐欑紦瀛橈級鎻愪緵缁熶竴鐨勬暟鎹瓨鍌ㄤ笌鏌ヨ鑳藉姏銆?

**鏍稿績璁捐鍘熷垯锛氫笌鏁版嵁鏉ユ簮瑙ｈ€?*銆傛暟鎹簱瀹氫箟鏍囧噯鍖栫殑鏁版嵁鏍煎紡鍜屾煡璇㈡帴鍙ｏ紝鍙互鎺ュ叆鏉ヨ嚜 micro-benchmark銆佸叏妯″瀷 Profiling 鎴栧叾浠栦换浣曞伐鍏蜂骇鍑虹殑鏁版嵁銆?

```python
# tensor_cast/performance_model/perf_database/database.py

class PerfDatabase:
    """
    鎬ц兘鏁版嵁搴撲富绫汇€?

    鑱岃矗锛?
    - 鎸?data_path 瀹氫綅鏁版嵁鐩綍锛屼粠 metadata.yaml 璇诲彇鐗堟湰淇℃伅鍜?YAML 鏄犲皠琛ㄨ矾寰?
      metadata.yaml 鍖呭惈 device銆乥ackend銆乿ersion銆乵apping_yaml锛堟寚鍚?YAML 鏄犲皠琛ㄧ殑璺緞锛夈€乧ollection_date 绛夊瓧娈?
    - 寤惰繜鍔犺浇骞剁紦瀛樼畻瀛愭€ц兘鏁版嵁锛圕SV/Parquet锛?
    - 鎻愪緵 lookup() / store() / save() 鎺ュ彛
    - 濮旀墭 QueryEngine 鎵ц鏌ヨ锛堢簿纭尮閰?+ 鎻掑€硷級

    鏁版嵁鏍煎紡绾﹀畾锛?
    - 姣忎釜 Schema 瀵瑰簲涓€涓暟鎹枃浠讹紙濡?matmul.csv銆乫used_attention.csv锛?
    - CSV 鍒?= Schema.dimensions 瀹氫箟鐨勭淮搴﹀悕 + "latency_us" 鍒?
    - 绂绘暎缁村害锛坬uant_mode 绛夛級浣滀负杩囨护鏉′欢
    - 杩炵画缁村害锛坢銆乥atch 绛夛級鐢ㄤ簬鎻掑€?
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        writable: bool = False,
    ):
        self.data_path = Path(data_path)
        self.query_engine = QueryEngine()
        self.writable = writable
        self._data_cache: Dict[str, Dict] = {}  # schema_name 鈫?nested dict
        # writable=True 鏃惰嫢 data_path 涓嶅瓨鍦ㄦ垨鏃?metadata.yaml锛岃嚜鍔ㄥ垱寤虹洰褰曞拰榛樿 metadata
        if self.writable and not (self.data_path / "metadata.yaml").exists():
            self.data_path.mkdir(parents=True, exist_ok=True)
            self._create_default_metadata()
        self.metadata = self._load_metadata()
        self.op_mapping: Dict[str, str] = self._load_op_mapping()

    def lookup(self, key: "OperatorKey") -> Optional[QueryResult]:
        """鏌ヨ绠楀瓙鎬ц兘鏁版嵁銆傝繑鍥?None 琛ㄧず鏃犳暟鎹€?""
        data = self._load_data(key.schema_name)
        if data is None:
            return None
        schema = OperatorSchema._registry[key.schema_name]
        try:
            return self.query_engine.query(schema, data, key.shape)
        except PerfDataNotAvailableError:
            return None

    def store(self, key: "OperatorKey", result: QueryResult) -> None:
        """鍐欏叆绠楀瓙鎬ц兘鏁版嵁锛堜粎 writable=True 鏃跺彲鐢級銆?""
        if not self.writable:
            raise RuntimeError("PerfDatabase is read-only")
        ...

    def save(self, output_path: Optional[Path] = None) -> None:
        """灏嗗唴瀛樹腑鐨勬暟鎹寔涔呭寲鍒扮鐩樸€?""
        ...
```

**绂绘暎缁村害璁捐璇存槑**锛?

鏁版嵁搴撶殑绂绘暎缁村害搴斾笌 TensorCast 鐨勬劅鐭ヨ兘鍔涘榻愩€傚弬鑰?[AI Configurator](https://github.com/ai-dynamo/aiconfigurator) 鐨勮璁★細

| 缁村害绫诲瀷 | 鏄惁绾冲叆 | 鐞嗙敱 |
|---------|---------|------|
| **quant_mode** | 鏄紙绂绘暎锛?| TensorCast 閫氳繃涓嶅悓绠楀瓙鍚嶅尯鍒嗛噺鍖栨ā寮忥紙濡?`static_quant_linear` vs `fp8_linear`锛夛紝鏄犲皠鍒版暟鎹簱鐨?`quant_mode` 鍒椼€侫I Configurator 鍚屾牱浠?`quant_mode` 浣滀负宓屽瀛楀吀鐨勭涓€绾?key |
| **BF16 vs FP16** | 缁熶竴涓?`fp16` | AI Configurator 涓嶅尯鍒?BF16/FP16銆傛槆鑵剧‖浠朵笂 BF16/FP16 鐨?Cube 绠楀姏鐩稿悓锛屽疄娴嬪樊寮傚彲蹇界暐 |
| **stride / layout** | 涓嶇撼鍏?| TensorCast 浣跨敤 meta tensor锛屼笉鎰熺煡鍐呭瓨鎺掑竷銆侫I Configurator 鍚屾牱涓嶈拷韪?layout |
| **kv_cache_quant_mode** | 鏄紙绂绘暎锛屼粎 Attention锛?| AI Configurator 瀵?Attention 鍒嗗埆杩借釜 `fmha_quant_mode` 鍜?`kv_cache_quant_mode`銆傛湰鏂规鍦?`fused_attention` Schema 涓鍔?`kv_cache_dtype` 绂绘暎缁村害 |
| **head_size** | 鏄紙绂绘暎锛屼粎 Attention锛?| 涓嶅悓 head_dim 鐨?Attention 鍐呮牳琛屼负宸紓澶э紙濡?128 vs 576 for MLA锛夛紝搴斾綔涓虹鏁ｇ淮搴﹁€岄潪鎻掑€?|

> **娉ㄦ剰**锛歚quant_mode` 鐨勫叿浣撳彇鍊奸渶涓?TensorCast 鐨勯噺鍖栭厤缃紙`QuantConfig`锛夊拰 YAML 鏄犲皠琛ㄤ腑鐨勭畻瀛愬垎绫讳繚鎸佷竴鑷淬€備緥濡?`tensor_cast.static_quant_linear.default` 瀵瑰簲 `quant_mode=int8`锛宍tensor_cast.fp8_linear.default` 瀵瑰簲 `quant_mode=fp8`銆?

**鏁版嵁鏂囦欢鏍煎紡绀轰緥锛坢atmul.csv锛?*锛?
```csv
quant_mode,m,n,k,latency_us
fp16,1,4096,5120,12.5
fp16,8,4096,5120,13.2
...
```

### 4.2 OperatorKey锛堝叡浜煡璇㈤敭锛?

`OperatorKey` 灏佽浠?`OpInvokeInfo` 鍒?schematized 鏌ヨ key 鐨勫叡浜浆鎹㈤€昏緫锛屼緵 `ProfilingPerformanceModel` 鍜?`EmpiricalPerformanceModel` 鍏卞悓浣跨敤銆?

```python
# tensor_cast/performance_model/perf_database/operator_key.py

@dataclass
class OperatorKey:
    """
    鏍囧噯鍖栫殑绠楀瓙鏌ヨ閿€?

    涓?CachingPerformanceModel 鐨?SHA256 cache_key 鐨勫尯鍒細
    - CachingPerformanceModel: session 绾х簿纭尮閰嶏紝func + 瀹屾暣 tensor shape/stride/dtype
    - OperatorKey: 璺?session 璇箟鍖归厤锛屾寜 Schema 瀹氫箟鐨勬€ц兘鐩稿叧缁村害鎻愬彇
    """
    schema_name: str          # Schema 鍚嶇О锛堝 "matmul"銆?fused_attention"锛?
    shape: Dict[str, Any]     # 鎬ц兘鐩稿叧缁村害锛堝 {m: 136, n: 4096, k: 5120, quant_mode: "int8"}锛?

    @classmethod
    def from_op_invoke_info(
        cls, op_invoke_info: OpInvokeInfo, op_mapping: Dict[str, str]
    ) -> Optional["OperatorKey"]:
        """
        浠?OpInvokeInfo 鎻愬彇 OperatorKey銆?
        op_mapping: YAML 鍔犺浇鐨?TensorCast 绠楀瓙 鈫?Schema 鍚嶇О鏄犲皠琛紙瑙?4.6 鑺傦級
        杩斿洖 None 琛ㄧず璇ョ畻瀛愪笉鍦ㄤ换浣?Schema 鐨勮鐩栬寖鍥村唴銆?
        """
        op_name = str(op_invoke_info.func)
        schema_name = op_mapping.get(op_name)
        if schema_name is None:
            return None
        schema = OperatorSchema._registry.get(schema_name)
        if schema is None:
            return None
        shape = schema.extract_shape_from_op(op_invoke_info)
        return cls(schema_name=schema_name, shape=shape)
```

### 4.3 ProfilingPerformanceModel锛堟柊澧烇級

鐙珛鐨?`PerformanceModel` 瀛愮被锛屼笓鑱屼粠棰勬瀯寤虹殑 `PerfDatabase` 涓煡璇㈢畻瀛愯€楁椂銆?

```python
# tensor_cast/performance_model/profiling.py

class ProfilingPerformanceModel(PerformanceModel):
    """
    鍩轰簬棰勯噰闆?Profiling 鏁版嵁鐨勬€ц兘妯″瀷銆?

    - 鍙鏌ヨ PerfDatabase锛屼笉杩涜浠讳綍鍐欏叆
    - 鏈懡涓椂鍐呴儴 fallback 鑷?AnalyticPerformanceModel
    - 閫氳繃 perf_models 鍒楄〃鎺ュ叆 Runtime锛坱ensor_cast/runtime.py:41锛?
    - 鐢?Runtime 鑷姩鍖呰涓?CachingPerformanceModel
    """

    def __init__(
        self,
        device_profile: DeviceProfile,
        database: PerfDatabase,
        fallback_model: Optional[PerformanceModel] = None,
    ):
        super().__init__("profiling", device_profile)
        self.database = database
        self.fallback_model = fallback_model or AnalyticPerformanceModel(device_profile)

    def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        key = OperatorKey.from_op_invoke_info(op_invoke_info, self.database.op_mapping)
        if key is not None:
            result = self.database.lookup(key)
            if result is not None:
                return PerformanceModel.Result(
                    execution_time_s=result.latency_us * 1e-6,
                    statistics={"source": result.source.name, "confidence": result.confidence}
                )
        # 鏈懡涓細鍥為€€鑷?Roofline
        return self.fallback_model.process_op(op_invoke_info)

    def get_classifiers(self) -> List[PerformanceModel.OpClassifier]:
        return self.fallback_model.get_classifiers()
```

### 4.4 EmpiricalPerformanceModel 澧炲己

澧炲己鐜版湁 `EmpiricalPerformanceModel`锛屽彲閫夋帴鍏?`PerfDatabase` 浣滀负璺?session 鐨勬寔涔呭寲缂撳瓨銆備笉浼?`cache_db` 鏃朵笌鐜版湁琛屼负瀹屽叏涓€鑷达紙鍚庡悜鍏煎锛夈€?

```python
# tensor_cast/performance_model/empirical.py锛堝寮哄悗锛?

class EmpiricalPerformanceModel(PerformanceModel):
    """
    鍩轰簬 JIT 瀹炴祴鐨勬€ц兘妯″瀷锛屽彲閫夋帴鍏?PerfDatabase 浣滀负鎸佷箙鍖栫紦瀛樸€?
    娴佺▼锛氭煡缂撳瓨锛堣嫢鏈夛級 鈫?JIT benchmark 鈫?鍐欑紦瀛橈紙鑻ユ湁锛?
    """

    def __init__(
        self,
        device_profile: DeviceProfile,
        cache_db: Optional[PerfDatabase] = None,
    ):
        super().__init__("empirical", device_profile)
        self.op_benchmark = OpBenchmark(device_profile)
        self.cache_db = cache_db

    def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        # 1. 鑻ユ湁鎸佷箙鍖栫紦瀛橈紝鍏堟煡缂撳瓨
        key = None
        if self.cache_db is not None:
            key = OperatorKey.from_op_invoke_info(op_invoke_info, self.cache_db.op_mapping)
            if key is not None:
                cached = self.cache_db.lookup(key)
                if cached is not None:
                    return PerformanceModel.Result(
                        execution_time_s=cached.latency_us * 1e-6,
                        statistics={"source": "cached_benchmark"}
                    )

        # 2. JIT benchmark锛堢幇鏈夐€昏緫锛?
        result = self.op_benchmark.benchmark(op_invoke_info)

        # 3. 灏嗙粨鏋滃啓鍏ユ寔涔呭寲缂撳瓨
        if self.cache_db is not None and key is not None:
            self.cache_db.store(key, QueryResult(
                latency_us=result.execution_time_s * 1e6,
                confidence=1.0,
                source=QuerySource.MEASURED
            ))

        return result
```

### 4.5 OperatorSchema锛堟彃浠跺寲鏋舵瀯锛?

姣忕 Schema **涓?kernel_details.csv 涓殑纭欢鍐呮牳绫诲瀷涓€涓€瀵瑰簲**锛岃礋璐ｏ細
1. 澹版槑璇ョ‖浠跺唴鏍哥殑 Shape 缁村害绌洪棿锛堝摢浜涚淮搴﹀奖鍝嶆€ц兘锛?
2. 鎻愪緵浠?`OpInvokeInfo` 鎻愬彇 Shape 鐨勯€昏緫锛堣繛鎺?TensorCast 杩愯鏃讹級
3. 澹版槑璇?Schema 澶勭悊鐨?TensorCast 绠楀瓙鍚嶇О鍒楄〃锛堥€氳繃鐗堟湰鐩稿叧鐨?YAML 鏄犲皠琛ㄩ厤缃級

> **璁捐鍘熷垯**锛歋chema 鐨勭矑搴︿笌纭欢鍐呮牳瀵归綈锛岃€岄潪涓?TensorCast 鎶借薄绠楀瓙瀵归綈銆備笉鍚岀‖浠跺唴鏍稿嵆浣垮姛鑳界浉浼硷紙濡?`MatMulV2` 鍜?`GroupedMatmul` 閮芥槸鐭╅樀涔橈級锛屼篃鏈変笉鍚岀殑瀹炵幇鍜屾€ц兘鏇茬嚎锛屽繀椤诲垎寮€寤烘ā銆?

```python
# tensor_cast/performance_model/perf_database/schemas/base.py

class DimensionSpec:
    """
    缁村害瑙勬牸瀹氫箟銆?
    - 鍙彃鍊肩淮搴︼細鏁板€艰繛缁紝鏀寔鎻掑€间及绠楋紙濡?m銆乶um_tokens銆乥atch_size锛?
    - 绂绘暎鍖归厤缁村害锛氭灇涓惧€硷紝蹇呴』绮剧‘鍖归厤浣滀负杩囨护鏉′欢锛堝 quant_mode銆乨type锛?
    """
    name: str
    interpolatable: bool = True   # True=鍙彃鍊? False=绂绘暎鍖归厤
    typical_range: Optional[Tuple[int, int]] = None

class OperatorSchema(ABC):
    """
    绠楀瓙鎬ц兘 Schema 鍩虹被銆?
    姣忎釜 Schema 瀵瑰簲 kernel_details.csv 涓殑涓€绉嶇‖浠跺唴鏍哥被鍨嬨€?
    閲囩敤娉ㄥ唽琛ㄦā寮忚嚜鍔ㄥ彂鐜帮紙涓?DeviceProfile銆丱pInvokeInfo.register_op_properties 璁捐涓€鑷达級銆?
    """
    _registry: ClassVar[Dict[str, "OperatorSchema"]] = {}

    def __init_subclass__(cls, **kwargs):
        """鑷姩娉ㄥ唽瀛愮被瀹炰緥銆?""
        super().__init_subclass__(**kwargs)
        if not getattr(cls, '__abstractmethods__', None):
            instance = cls()
            OperatorSchema._registry[instance.name] = instance

    @property
    @abstractmethod
    def name(self) -> str:
        """Schema 鍚嶇О锛屽搴旀暟鎹枃浠跺悕鍜岀‖浠跺唴鏍哥被鍨嬶紙濡?"matmul" 鈫?matmul.csv锛夈€?""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> List[DimensionSpec]:
        """璇ョ‖浠跺唴鏍哥殑 Shape 缁村害瀹氫箟銆?""
        ...

    @abstractmethod
    def extract_shape_from_op(self, op_invoke_info: OpInvokeInfo) -> Dict[str, Any]:
        """浠庢嫤鎴埌鐨勭畻瀛愯皟鐢ㄤ腑鎻愬彇 Shape 缁村害瀛楀吀銆?""
        ...

    def get_interpolation_dimensions(self) -> List[str]:
        return [d.name for d in self.dimensions if d.interpolatable]

    def get_discrete_dimensions(self) -> List[str]:
        return [d.name for d in self.dimensions if not d.interpolatable]
```

**鍏蜂綋 Schema 涓€瑙?*锛堜笌 kernel_details.csv 纭欢鍐呮牳瀵归綈锛夛細

| Schema | 鍚嶇О | 瀵瑰簲纭欢鍐呮牳 | 鍏抽敭缁村害 | 鏁版嵁鏂囦欢 |
|--------|------|------------|---------|---------|
| `MatMulSchema` | `matmul` | `MatMulV2` | quant_mode\*, m, n, k | `matmul.csv` |
| `GroupedMatMulSchema` | `grouped_matmul` | `GroupedMatmul` | quant_mode\*, num_tokens, hidden_size, inter_size, num_experts | `grouped_matmul.csv` |
| `QuantBatchMatMulSchema` | `quant_batch_matmul` | `QuantBatchMatmulV3` | quant_mode\*, m, n, k | `quant_batch_matmul.csv` |
| `TransposeBatchMatMulSchema` | `transpose_batch_matmul` | `TransposeBatchMatMul` | batch, m, n, k | `transpose_batch_matmul.csv` |
| `FusedAttentionSchema` | `fused_attention` | `FusedInferAttentionScore` | batch, query_len, context_len, num_heads, num_kv_heads, head_dim\*, kv_cache_dtype\* | `fused_attention.csv` |
| `InterleaveRopeSchema` | `interleave_rope` | `InterleaveRope` | num_tokens, num_heads, head_dim | `interleave_rope.csv` |
| `AddRmsNormSchema` | `add_rms_norm` | `AddRmsNorm` / `InplaceAddRmsNorm` | num_tokens, hidden_size | `add_rms_norm.csv` |
| `MoeDispatchSchema` | `moe_dispatch` | `MoeDistributeDispatch` | num_tokens, num_experts, hidden_size | `moe_dispatch.csv` |
| `MoeCombineSchema` | `moe_combine` | `MoeCombine` | num_tokens, num_experts, hidden_size | `moe_combine.csv` |
| `MoeGatingSchema` | `moe_gating` | `MoeGatingTopK` | num_tokens, num_experts | `moe_gating.csv` |
| `SwiGluSchema` | `swiglu` | `SwiGlu` / `DequantSwigluQuant` | num_tokens, hidden_size | `swiglu.csv` |
| `AscendQuantSchema` | `ascend_quant` | `AscendQuantV2` / `DynamicQuant` | num_tokens, hidden_size | `ascend_quant.csv` |
| `ReshapeAndCacheSchema` | `reshape_and_cache` | `ReshapeAndCacheNdKernel` | num_tokens, num_kv_heads, head_dim | `reshape_and_cache.csv` |
| `TensorMoveSchema` | `tensor_move` | `TensorMove` | total_bytes | `tensor_move.csv` |
| `AllReduceSchema` | `all_reduce` | `AllReduce` | num_devices, message_bytes | `all_reduce.csv` |
| `AllGatherSchema` | `all_gather` | `AllGather` | num_devices, message_bytes | `all_gather.csv` |
| `AllToAllSchema` | `all_to_all` | `AllToAll` | num_devices, message_bytes | `all_to_all.csv` |

> **娉ㄦ剰**锛氫互涓婁负鍒濆鍙傝€冨垪琛ㄣ€傛爣 `*` 鐨勭淮搴︿负绂绘暎鍖归厤缁村害锛坄interpolatable=False`锛夛紝鍏朵綑涓哄彲鎻掑€肩淮搴︺€傚叿浣撶殑 Schema 鍚嶇О銆佺淮搴﹀畾涔夊拰鏄犲皠鍏崇郴闇€鏍规嵁瀹為檯 kernel_details.csv 鐨勫唴鏍稿悕绉扮敱涓撳纭銆備笉鍚?vLLM-Ascend 鐗堟湰鐨勫唴鏍稿悕鍙兘涓嶅悓锛岄€氳繃鐗堟湰鐩稿叧鐨?YAML 鏄犲皠琛ㄧ鐞嗐€?

**MatMul Schema 绀轰緥瀹炵幇**锛?
```python
# tensor_cast/performance_model/perf_database/schemas/matmul.py

class MatMulSchema(OperatorSchema):
    """瀵瑰簲纭欢鍐呮牳 MatMulV2锛團P16 鏍囧噯绋犲瘑鐭╅樀涔橈級銆?""

    @property
    def name(self) -> str:
        return "matmul"

    @property
    def dimensions(self) -> List[DimensionSpec]:
        return [
            DimensionSpec("quant_mode", interpolatable=False),
            DimensionSpec("m", interpolatable=True),
            DimensionSpec("n", interpolatable=True),
            DimensionSpec("k", interpolatable=True),
        ]

    def extract_shape_from_op(self, op_invoke_info: OpInvokeInfo) -> Dict[str, Any]:
        x, w = op_invoke_info.args[0], op_invoke_info.args[1]
        return {"quant_mode": "fp16", "m": x.shape[0], "k": x.shape[1], "n": w.shape[1]}
```

### 4.6 TensorCast 绠楀瓙 鈫?纭欢鍐呮牳 Schema 鏄犲皠

鏄犲皠琛ㄨВ鍐?*鍚嶇О缈昏瘧**闂锛歍ensorCast dispatch 鐨勭畻瀛愬悕涓?Profiling 鐨勭‖浠跺唴鏍稿悕涓嶅悓锛堝 `aten.mm.default` 鈫?`MatMulV2`锛夛紝闇€瑕佹槧灏勮〃灏?TensorCast 绠楀瓙瀵瑰簲鍒版纭殑 Schema銆?

**鏍稿績鍓嶆彁锛氶€氳繃缂栬瘧 Pass 瀹炵幇 1:1 瀵归綈**銆俆ensorCast 宸叉湁鎴愮啛鐨?`PatternMatcherPass` 鍩虹璁炬柦锛坄tensor_cast/compilation/patterns/` 涓嬪凡瀹炵幇 13+ 绉?RMSNorm 铻嶅悎鍜?2 绉?RoPE 铻嶅悎锛夛紝瀵逛簬 vLLM 纭欢铻嶅悎鍐呮牳涓?TensorCast dispatch trace 涓嶄竴鑷寸殑鎯呭喌锛?*缁熶竴閫氳繃鏂板缂栬瘧 Pass 瑙ｅ喅**锛屼娇 dispatch trace 涓庣‖浠跺唴鏍稿垪琛ㄤ繚鎸?1:1 瀵归綈銆傚綋鍓嶄粛闇€琛ュ厖鐨勭紪璇?Pass 璇﹁绗?9.1 鑺傘€?

#### 鏄犲皠琛ㄦ牸寮?

鏄犲皠鍏崇郴閫氳繃**鐗堟湰鐩稿叧鐨?YAML 閰嶇疆**绠＄悊銆俙tensorcast_op_to_schema` 涓烘墎骞崇殑 1:1 鏄犲皠锛氭瘡涓?TensorCast 绠楀瓙鏄犲皠鍒颁竴涓‖浠跺唴鏍?Schema銆傚悓绫荤畻瀛愮殑涓嶅悓閲忓寲鍙樹綋鍙兘鏄犲皠鍒颁笉鍚岀殑纭欢鍐呮牳锛堝 `aten.mm` 鈫?`matmul`锛宍static_quant_linear` 鈫?`quant_batch_matmul`锛夈€?

```yaml
# perf_database/mappings/vllm_ascend/v0.13.yaml
version: "0.13"
device: ATLAS_800_A3_752T_128G_DIE

tensorcast_op_to_schema:
  # GEMM
  "aten.mm.default": matmul
  "tensor_cast.static_quant_linear.default": quant_batch_matmul
  "tensor_cast.static_quant_linear_int4.default": quant_batch_matmul
  "tensor_cast.fp8_linear.default": quant_batch_matmul
  "tensor_cast.mxfp4_linear.default": quant_batch_matmul

  # Grouped MatMul (MoE)
  "tensor_cast.grouped_matmul.default": grouped_matmul
  "tensor_cast.grouped_matmul_quant.default": grouped_matmul
  "tensor_cast.grouped_matmul_fp8.default": grouped_matmul

  # Attention
  "tensor_cast.attention.default": fused_attention
  "tensor_cast.attention_quant.default": fused_attention
  "tensor_cast.multihead_latent_attention.default": fused_attention
  "tensor_cast.multihead_latent_attention_quant.default": fused_attention

  # RoPE
  "tensor_cast.apply_rope.default": interleave_rope

  # Normalization锛堝凡鏈夌紪璇?Pass 铻嶅悎锛岃 compilation/patterns/rms_norm.py锛?
  "tensor_cast.add_rms_norm.default": add_rms_norm
  "tensor_cast.add_rms_norm_quant.default": add_rms_norm
  "tensor_cast.add_rms_norm_dynamic_quant_symmetric.default": add_rms_norm

  # Activation锛堥渶鏂板缂栬瘧 Pass锛岃 9.1 鑺傦級
  "tensor_cast.swiglu.default": swiglu                  # 寰呭疄鐜?
  "tensor_cast.dequant_swiglu_quant.default": swiglu     # 寰呭疄鐜?

  # MoE 璺敱
  "tensor_cast.permute_tokens.default": moe_dispatch
  "tensor_cast.unpermute_tokens.default": moe_combine
  "aten.topk.default": moe_gating  # 娉ㄦ剰锛歛ten.topk 涔熺敤浜庨噰鏍凤紝姝ゅ浠呰繎浼煎鐞?MoE 鍦烘櫙锛屽緟 9.1 鑺傞€傞厤鍚庢敼鐢ㄤ笓鐢ㄧ畻瀛?

  # Communication
  "tensor_cast.all_reduce.default": all_reduce
  "tensor_cast.all_gather.default": all_gather
  "tensor_cast.all_to_all.default": all_to_all

  # 閲忓寲锛坙inear 鍓嶇殑杈撳叆閲忓寲锛岀嫭绔?dispatch锛?
  "tensor_cast.quantize.default": ascend_quant
  "tensor_cast.dynamic_quantize_symmetric.default": ascend_quant
  "tensor_cast.dynamic_quantize_asymmetric.default": ascend_quant
  "tensor_cast.dynamic_quantize_mxfp4.default": ascend_quant

  # Cache
  "tensor_cast.reshape_and_cache.default": reshape_and_cache
```

> **娉ㄦ剰**锛氫互涓婃槧灏勪负鍒濆绀轰緥銆傛爣娉?寰呭疄鐜?鐨勭畻瀛愰渶鍏堝畬鎴愮 9.1 鑺備腑鐨勭紪璇?Pass 閫傞厤銆傚叿浣撴槧灏勫叧绯婚渶鏍规嵁瀹為檯 kernel_details.csv 鐢变笓瀹堕厤缃紝涓嶅悓鐗堟湰鐨?vLLM-Ascend 鏄犲皠琛ㄤ細涓嶅悓銆?

### 4.7 QueryEngine锛堢簿纭尮閰?+ 鎻掑€硷級

`PerfDatabase` 鍐呴儴鐨?QueryEngine **浠呭鐞嗗疄娴嬫暟鎹?*锛氱簿纭尮閰?鈫?鎻掑€?鈫?澶栨帹銆傛湭鏀跺綍绠楀瓙涓嶅湪 QueryEngine 鍐呭仛 Roofline 鍏滃簳锛岃€屾槸杩斿洖鏌ヨ澶辫触锛岀敱 `ProfilingPerformanceModel` 鐨?`fallback_model` 澶勭悊銆?

```python
# tensor_cast/performance_model/perf_database/query_engine.py

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

class QueryEngine:
    """澶勭悊绮剧‘鏌ユ壘鍜屾彃鍊间及绠椼€?""

    def query(self, schema: OperatorSchema, data: Dict, shape: Dict) -> QueryResult:
        # 鍒嗙绂绘暎缁村害鍜岃繛缁淮搴?
        discrete_dims = schema.get_discrete_dimensions()
        interp_dims = schema.get_interpolation_dimensions()

        # Step 1: 鎸夌鏁ｇ淮搴﹁繃婊ゅ埌瀛愰泦
        # Step 2: 绮剧‘鍖归厤
        # Step 3: 鎻掑€?
        # Step 4: 澶栨帹锛堟渶杩戦偦 + 绾挎€у鎺級
```

> **鎻掑€肩瓥鐣ュ弬鑰?*锛氬弬鑰?[AI Configurator](https://github.com/ai-dynamo/aiconfigurator) 鐨?2D+1D 娣峰悎鎻掑€兼柟娉曪紙瑙佺 2.5 鑺傦級锛氬厛瀵逛袱涓淮搴﹀仛鍙岀嚎鎬ф彃鍊硷紝鍐嶅绗笁涓淮搴﹀仛 1D 鎻掑€笺€傚 Attention 绛?O(n虏) 澶嶆潅搴︾殑绠楀瓙锛屽湪鎻掑€煎墠瀵瑰簭鍒楃淮搴﹀仛 sqrt 鍙樻崲浠ユ彁楂樻嫙鍚堢簿搴︺€傜疆淇″害璇勫垎鍩轰簬鏌ヨ鐐瑰埌鏈€杩戝疄娴嬫暟鎹偣鐨勮窛绂昏绠椼€?

---

## 5. TensorCast 闆嗘垚鎺ュ彛

### 5.1 Runtime 闆嗘垚

`Runtime` 绫绘棤闇€淇敼銆傚畠宸叉敮鎸佹帴鏀?`PerformanceModel` 瀹炰緥鍒楄〃锛屽苟鑷姩鍖呰涓?`CachingPerformanceModel`銆備笁绉?PerformanceModel 鐨勪娇鐢ㄦ柟寮忥細

```python
# 鏂瑰紡 1: ProfilingPerformanceModel锛堟煡棰勬瀯寤烘暟鎹簱锛屾棤闇€鐗╃悊璁惧锛?
db = PerfDatabase("perf_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/v0.13.0")
perf_model = ProfilingPerformanceModel(device_profile, database=db)

# 鏂瑰紡 2: EmpiricalPerformanceModel + 鎸佷箙鍖栫紦瀛橈紙鏈夌墿鐞嗚澶囷級
cache_db = PerfDatabase("./my_benchmark_cache", writable=True)
perf_model = EmpiricalPerformanceModel(device_profile, cache_db=cache_db)

# 鏂瑰紡 3: AnalyticPerformanceModel锛圧oofline锛岀幇鏈夐粯璁よ涓猴級
perf_model = AnalyticPerformanceModel(device_profile)

# 缁熶竴浼犲叆 Runtime
runtime = Runtime(perf_models=perf_model, device_profile=device_profile)
```

### 5.2 CLI 鎺ュ彛

鍦?`tensor_cast/scripts/text_generate.py` 涓柊澧炲弬鏁帮細

```python
parser.add_argument("--performance-model",
                    choices=["analytic", "profiling", "empirical"],
                    default="analytic", help="鎬ц兘妯″瀷绫诲瀷")
parser.add_argument("--perf-database", type=str, default=None,
                    help="鎬ц兘鏁版嵁搴撹矾寰勶紙profiling / empirical 妯″紡鐢熸晥锛夛紝"
                         "鎸囧悜鍖呭惈 metadata.yaml 鍜?CSV 鏁版嵁鏂囦欢鐨勭洰褰?)
```

| CLI 閫夐」 | 鍒涘缓鐨?PerformanceModel | 鏄惁闇€瑕佺墿鐞嗚澶?| 鏄惁闇€瑕佹暟鎹簱 |
|---------|------------------------|----------------|-------------|
| `--performance-model analytic` | `AnalyticPerformanceModel` | 鍚?| 鍚?|
| `--performance-model profiling` | `ProfilingPerformanceModel` | 鍚?| 鏄紙`--perf-database`锛?|
| `--performance-model empirical` | `EmpiricalPerformanceModel` | 鏄?| 鍙€夛紙`--perf-database` 浣滅紦瀛橈級 |

### 5.3 鏁版嵁娴侊紙浠?ProfilingPerformanceModel 涓轰緥锛?

```
text_generate.py
  --performance-model profiling
  --perf-database ./perf_database/data/atlas_a3/.../v0.13.0/
       鈹?
       鈻?
  鍒涘缓 ProfilingPerformanceModel + PerfDatabase(data_path)
       鈹?
       鈻?
  Runtime.__torch_dispatch__ 鈫?鎷︽埅绠楀瓙 鈫?OpInvokeInfo
       鈹?
       鈻?
  ProfilingPerformanceModel.process_op()
       鈹溾攢鈹€鈻?OperatorKey.from_op_invoke_info()
       鈹?   op_mapping["aten.mm.default"] 鈫?"matmul" 鈫?MatMulSchema
       鈹?   鈫?extract_shape 鈫?{m: 136, n: 4096, k: 5120, quant_mode: "fp16"}
       鈹?
       鈹溾攢鈹€鈻?database.lookup(key)
       鈹?   绮剧‘鍖归厤 / 鎻掑€?/ 澶栨帹 鈫?QueryResult
       鈹?
       鈹溾攢鈹€鈻?鑻?lookup 杩斿洖 None 鈫?fallback_model.process_op()
       鈹?
       鈹斺攢鈹€鈻?杩斿洖 PerformanceModel.Result(execution_time_s=...)
```

---

## 6. 鑷姩鍖?Profiling 娴佹按绾?

> **娉ㄦ剰**锛氭湰鑺傛弿杩扮殑鏁版嵁閲囬泦娴佹按绾夸綔涓虹嫭绔嬪瓙绯荤粺锛屼唬鐮佷綅浜庝粨搴撻《灞?`perf_database/` 鐩綍锛屼笉鍖呭惈鍦?TensorCast 鍖呭唴銆?

### 6.1 鏁版嵁搴撴瀯寤虹瓥鐣?

| 绛栫暐 | 閫傜敤鍦烘櫙 | 浼樺娍 | 鍔ｅ娍 |
|------|---------|------|------|
| **鏂规 A**锛堥粯璁わ級锛氬叏妯″瀷 Profiling + 寰熀鍑嗘祴璇?| 鏈夊畬鏁?VLLM 閮ㄧ讲鐜 | 绮剧‘鎹曡幏铻嶅悎鍐呮牳锛屾牎鍑嗘暟鎹彲闈?| VLLM 鏈嶅姟鍚姩寮€閿€澶?|
| **鏂规 B**锛氫豢鐪熼┍鍔?+ 寰熀鍑嗘祴璇?| 浠呮湁瑁告満 NPU 鐜 | 鏃犻渶 VLLM 閮ㄧ讲锛岄噰闆嗘晥鐜囬珮 | 鏃犳硶瑕嗙洊铻嶅悎鍐呮牳 |
| **鏂规 C**锛氱函 Profiling 瀵煎叆 | 宸叉湁鐜版垚鐨?Profiling 鏁版嵁 | 闆堕澶栭噰闆嗘垚鏈?| Shape 瑕嗙洊鍙楅檺 |

### 6.2 涓ょ骇閲囬泦绛栫暐锛堟柟妗?A锛?

#### Level 1: 鍏ㄦā鍨?Profiling锛堣幏鍙栧熀鍑嗙湡鍊硷級

鍚敤鏄囪吘 Profiler 杩愯 VLLM 鎺ㄧ悊锛屾崟鑾风湡瀹炶瀺鍚堝唴鏍哥殑鎵ц鑰楁椂銆傛瘡涓湇鍔″疄渚嬪彧闇€鍚姩涓€娆★紝闅忓悗鍙戦€佸杞笉鍚?Shape 閰嶇疆鐨勫熀鍑嗘祴璇曡姹傘€?

```bash
export VLLM_TORCH_PROFILER_DIR=/path/to/output
export PROFILING_SAVE_PATH=/path/to/output
vllm serve $MODEL --tensor-parallel-size $TP --dtype bfloat16 ...
vllm bench serve --profile --dataset-name random \
    --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
    --max-concurrency $BATCH_SIZE
```

**鍗曞疄渚嬪唴 Shape 閬嶅巻鏂规**锛?
```python
# Decode 鍦烘櫙锛坬uery_len=1锛屼笉鍚?context_len 褰卞搷 KV Cache 闀垮害鍜?Attention 鎬ц兘锛?
for context_len in [256, 512, 1024, 2048, 4096, 8192]:
    for batch in [1, 8, 16, 32, 64, 128, 256]:
        bench(input_len=1, output_len=1, concurrency=batch,
              max_model_len=context_len)

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

#### Level 2: 鍗曠畻瀛愬井鍩哄噯娴嬭瘯锛圫hape 缃戞牸濉厖锛?

閫氳繃 `torch_npu` 鐩存帴瀵瑰崟涓畻瀛愯繘琛岀粏绮掑害 Shape 瑕嗙洊娴嬭瘯锛?

```python
# perf_database/scripts/collect_microbench.py
def benchmark_matmul(m, n, k, dtype, warmup=5, runs=20):
    """浣跨敤 NPU Event 璁℃椂鐨勫崟绠楀瓙寰熀鍑嗘祴璇曪紝杩斿洖骞冲潎鑰楁椂锛堝井绉掞級銆?""
    a = torch.randn(m, k, dtype=dtype, device='npu')
    b = torch.randn(k, n, dtype=dtype, device='npu')
    for _ in range(warmup):
        torch.mm(a, b)
    torch.npu.synchronize()
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
|-----|--------------------------|---------------------|
| 铻嶅悎鍐呮牳鎹曡幏 | 瀹屾暣鎹曡幏锛堝熀鍑嗙湡鍊硷級 | 鏃犳硶鎹曡幏 |
| Shape 鎺у埗 | 闂存帴鎺у埗锛坆atch銆乻eq_len锛?| 鐩存帴鎺у埗锛坢銆乶銆乲锛?|
| 鎵ц寮€閿€ | 楂橈紙鏈嶅姟鍚姩鑰楁椂闀匡級 | 浣庯紙浠呭唴鏍告墽琛岋級 |
| 瑕嗙洊鑼冨洿 | 姣忎釜妯″瀷绾?100 绉嶉厤缃?| 姣忎釜绠楀瓙绾?2500 绉?Shape |
| 閫傜敤鍦烘櫙 | 楠岃瘉涓庢牎鍑?| Shape 缃戞牸澶ц妯″～鍏?|

**鏍″噯鏂规硶**锛氬湪 Shape 閲嶅彔鍖哄煙瀵规瘮 Level 1 鍜?Level 2 鐨勬暟鎹紝璁＄畻姣忕被绠楀瓙鐨勪慨姝ｇ郴鏁帮紝浣?Level 2 鏁版嵁瀵归綈瀹為檯閮ㄧ讲鍦烘櫙銆?

### 6.3 Profiling 杈撳嚭瑙ｆ瀽鍣?

```python
# perf_database/scripts/parse_ascend_output.py

class AscendProfilerParser:
    def parse_kernel_details(self, csv_path: Path) -> pd.DataFrame:
        """瑙ｆ瀽 kernel_details.csv 鈫?缁撴瀯鍖?DataFrame銆?
        杩斿洖: name, duration_us, input_shapes, data_types,
              aicore_time_us, aiv_time_us, cube_utilization_pct"""

    def extract_single_step(self, df: pd.DataFrame) -> pd.DataFrame:
        """浠?Attention 鍐呮牳涓鸿竟鐣屾彁鍙栧崟涓?Decode/Prefill 姝ラ銆?""

    def map_to_schema(self, kernel_name: str, input_shapes: str,
                      mapping_yaml: Path) -> Tuple[str, Dict]:
        """灏?VLLM 鍐呮牳鍚嶆槧灏勫埌 (schema_name, shape_dict)銆?
        濡?"MatMulV2" + "136,4096; 4096,4096" 鈫?("matmul", {m:136, k:4096, n:4096})"""
```

### 6.4 鏂版ā鍨嬬畻瀛愬彂鐜版満鍒?

```python
# perf_database/scripts/operator coverage check tool

def discover_operators(profiling_output: Path, mapping_yaml: Path) -> Dict:
    """鍙戠幇 Profiling 涓瓨鍦ㄤ絾褰撳墠鏄犲皠琛ㄤ腑缂哄け鐨勭畻瀛愩€?
    杩斿洖 known 鍜?unknown 璁℃暟锛寀nknown 鍖呭惈 kernel 鍚嶇О鍜屽缓璁?Schema銆?""
    parser = AscendProfilerParser()
    kernels = parser.parse_kernel_details(profiling_output / "kernel_details.csv")
    known_ops = load_yaml_mappings(mapping_yaml)
    unknown = []
    for kernel_name in kernels["name"].unique():
        if not any(kernel_name in schema_kernels for schema_kernels in known_ops.values()):
            unknown.append({"kernel": kernel_name, "suggested_schema": auto_classify(kernel_name)})
    return {"known": len(kernels) - len(unknown), "unknown": unknown}
```

---

## 7. 鍏ㄩ潰绠楀瓙瑕嗙洊绛栫暐

### 7.1 绠楀瓙鍒嗙骇

| 灞傜骇 | 鍒ゅ畾鏍囧噯 | 澶勭悊鏂瑰紡 | 绠楀瓙鏁伴噺 |
|-----|---------|---------|---------|
| **Tier 1** | 鎵ц鑰楁椂鍗犳瘮 >2%锛堟垨璁＄畻鍐呮牳涓?>5%锛?| 蹇呴』浣跨敤瀹屾暣 Shape 缃戞牸杩涜 Profiling | 绾?14 涓?|
| **Tier 2** | 鎵ц鑰楁椂鍗犳瘮 0.5-2% | 浣跨敤绮剧畝 Shape 缃戞牸杩涜 Profiling | 绾?9 涓?|
| **Tier 3** | 鎵ц鑰楁椂鍗犳瘮 <0.5% | 閲囩敤 Roofline 鍏滃簳浼扮畻 | 60+ 涓?|

### 7.2 Tier 1 瀹屾暣绠楀瓙鍒楄〃

> **鏁版嵁鏉ユ簮**锛氫互涓嬪崰姣斿潎鍩轰簬 kernel_details.csv 鐨?*鍏ㄩ噺鍐呮牳鑰楁椂锛堝寘鍚€氫俊鍐呮牳锛?*锛屼笌绗?2.2 鑺備粎缁熻璁＄畻鍐呮牳鐨勫崰姣斿彛寰勪笉鍚屻€備緥濡?Qwen3 鐨?FusedInferAttentionScore 鍦ㄨ绠楀唴鏍镐腑鍗?18.2%锛堢 2.2 鑺傦級锛屼絾鍦ㄥ叏閲忓唴鏍革紙鍚€氫俊锛変腑浠呭崰 1.7%銆?
> - **Qwen3-32B**锛歅refill 妯″紡锛?6鍗★紝閫氫俊鍗?89.8%锛岃绠楀唴鏍稿崰姣旂浉搴斿亸灏?
> - **DeepSeekV3**锛欴ecode 妯″紡锛?2鍗★紝璁＄畻/閫氫俊姣斾緥鏇村潎琛?
>
> 鏌愬唴鏍稿湪浠讳竴妯″瀷涓秴杩?2% 鍗冲垪鍏?Tier 1銆?

| VLLM 鍐呮牳鍚嶇О | 鏁版嵁搴?Schema | Qwen3 鍗犳瘮 | DSV3 鍗犳瘮 |
|-------------|-------------|-----------|---------|
| hcom_allReduce\_ | `all_reduce` | **89.8%** | 8.1% |
| GroupedMatmul | `grouped_matmul` | - | **18.7%** |
| QuantBatchMatmulV3 | `quant_batch_matmul` | - | **18.3%** |
| FusedInferAttentionScore | `fused_attention` | 1.7% | **16.5%** |
| MoeDistributeDispatchV2 | `moe_dispatch` | - | **7.0%** |
| MatMulV2 | `matmul` | **4.1%** | 0.5% |
| AscendQuantV2 / DynamicQuant | `ascend_quant` | - | **3.9%** |
| TransposeBatchMatMul | `transpose_batch_matmul` | - | **3.8%** |
| MoeDistributeCombineV2 | `moe_combine` | - | **3.6%** |
| InterleaveRope | `interleave_rope` | - | **2.5%** |
| DequantSwigluQuant / SwiGlu | `swiglu` | 0.5% | **2.4%** |
| InplaceAddRmsNorm / AddRmsNorm | `add_rms_norm` | 0.8% | 1.8% |
| HcomAllGather | `all_gather` | 0.6% | 1.7% |

**Tier 2 鍙傝€冨垪琛?*锛?.5-2%锛夛細

| VLLM 鍐呮牳鍚嶇О | 鏁版嵁搴?Schema | Qwen3 鍗犳瘮 | DSV3 鍗犳瘮 |
|-------------|-------------|-----------|---------|
| MatMul锛圡LA latent proj锛?| `matmul` | - | 1.3% |
| MoeGatingTopK | `moe_gating` | - | 1.1% |
| TensorMove | `tensor_move` | 1.0% | 0.1% |
| RmsNorm | Roofline 鍏滃簳 | <0.1% | 0.9% |
| KvRmsNormRopeCache | 铻嶅悎鎷嗚В锛堣 4.6 鑺傦級 | - | 0.8% |
| Transpose | Roofline 鍏滃簳 | - | 0.8% |
| split\_qkv\_rmsnorm\_rope\_kernel | 铻嶅悎鎷嗚В锛堣 4.6 鑺傦級 | 0.5% | - |
| ReshapeAndCacheNdKernel | `reshape_and_cache` | 0.3% | - |
| SwiGlu | `swiglu` | 0.5% | - |

### 7.3 Shape 缃戞牸绛栫暐

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

**棰勪及 Shape 鎬婚噺**锛氭瘡涓増鏈害 5,000 涓€?

---

## 8. 寮€鍙戣鍒?

**鍥㈤槦鍒嗗伐**锛?
- **鍏，宸ュ叿鍥㈤槦**锛歍ensorCast 渚э紙PerformanceModel銆丱peratorSchema銆丆LI 闆嗘垚锛?
- **灏忓阀鐏靛洟闃?*锛氭暟鎹晶锛圥erfDatabase銆丵ueryEngine銆佹暟鎹噰闆嗘祦姘寸嚎锛?

**鍓嶇疆渚濊禆**锛歍ensorCast 绠楀瓙杩借釜瀵归綈锛堣瑙佺 9.1 鑺傦級锛岄渶鍦ㄩ泦鎴愭祴璇曞墠瀹屾垚銆?

**鐩爣**锛?026.3.20 瀹屾垚绔埌绔泦鎴愶紝DeepSeek-V3 / Qwen3-32B 鑳藉榻愮殑绠楀瓙瑕嗙洊绔埌绔?>90% 鐨勬椂闂淬€?

### 闃舵涓€锛氭牳蹇冩鏋朵笌绔埌绔┛鍒猴紙2 鍛級

鑱斿悎瀵归綈 Qwen3-32B 鍜?DeepSeek-V3 鍦ㄧ洰鏍囩増鏈笂鐨勭畻瀛愬垪琛ㄥ拰 YAML 鏄犲皠琛ㄣ€?

| 鍏，鍥㈤槦 | 灏忓阀鐏靛洟闃?|
|---------|-----------|
| `ProfilingPerformanceModel` + Roofline fallback | `PerfDatabase`锛堝姞杞姐€佹煡璇€佸瓨鍌級 |
| `OperatorSchema` 鍩虹被 + 鍒濆 Schema 闆嗗悎 | CSV 鏁版嵁鏍煎紡瑙勮寖 + 绀轰緥鏁版嵁 + 绮剧‘鍖归厤鏌ヨ |
| `OperatorKey` + CLI 闆嗘垚 | Profiling 瑙ｆ瀽鍣紙`parse_ascend_output.py`锛?|
| 鍩轰簬绀轰緥鏁版嵁搴撶┛鍒虹鍒扮浠跨湡 | 绌垮埡鏁版嵁閲囬泦鏂规 |

### 闃舵浜岋細鎻掑€煎紩鎿庝笌鏁版嵁閲囬泦锛? 鍛級

| 鍏，鍥㈤槦 | 灏忓阀鐏靛洟闃?|
|---------|-----------|
| `EmpiricalPerformanceModel` 澧炲己锛堝彲閫?`cache_db`锛?| `QueryEngine`锛堟彃鍊笺€佺疆淇″害璇勫垎锛?|
| 鎵╁睍 Schema 瑕嗙洊锛圡oE銆佸綊涓€鍖栥€佽瀺鍚堢畻瀛愮瓑锛?| 鏁版嵁閲囬泦鑴氭湰锛堝叏妯″瀷 Profiling + 寰熀鍑嗘祴璇曪級 |
| | A3 涓婂垵濮嬫暟鎹簱閲囬泦 |

### 闃舵涓夛細闆嗘垚娴嬭瘯涓庨獙璇侊紙1 鍛級

鑱斿悎浣跨敤 Qwen3-32B 鍜?DeepSeek-V3 杩涜绔埌绔簿搴﹂獙璇併€?

**楠岃瘉鏍囧噯**锛?

| 鎸囨爣 | 鐩爣鍊?|
|-----|-------|
| 绔埌绔€楁椂璇樊 | <15%锛堝姣斿疄闄?VLLM Profiling锛?|
| 鍗曠畻瀛愯宸紙宸插尮閰嶇畻瀛愶級 | <20% |
| 鏃堕棿瑕嗙洊鐜?| >90% |

---

## 9. 澶栭儴渚濊禆涓庢墿灞曞缓璁?

### 9.1 鍓嶇疆渚濊禆锛歍ensorCast 绠楀瓙杩借釜瀵归綈

鏈柟妗堣姹?TensorCast dispatch trace 涓?vLLM Profiling 鐨勭‖浠跺唴鏍稿垪琛ㄥ湪鍏抽敭绠楀瓙涓?1:1 瀵归綈锛堣 4.6 鑺傦級銆傚榻愭柟寮忎负**鏂板缂栬瘧 Pass**锛坄PatternMatcherPass`锛夛紝灏嗗涓?aten 绠楀瓙铻嶅悎涓哄崟涓€ TensorCast 鑷畾涔夌畻瀛愶紝涓庡凡鏈夌殑 13+ 绉?RMSNorm 铻嶅悎 Pass锛坄compilation/patterns/rms_norm.py`锛夊拰 RoPE 铻嶅悎 Pass锛坄compilation/patterns/rotary_embedding.py`锛変竴鑷淬€?

褰撳墠瀛樺湪鑻ュ共涓嶅尮閰嶉」闇€閫氳繃鍓嶇疆閫傞厤宸ヤ綔瑙ｅ喅锛?

| 涓嶅尮閰嶉」 | 鐜扮姸 | 鎵€闇€閫傞厤 | 瀵瑰簲纭欢鍐呮牳 |
|---------|------|---------|------------|
| **SwiGlu 铻嶅悎** | TensorCast dispatch `aten.silu` + `aten.mul` 涓や釜鐙珛绠楀瓙 | 鏂板 `PatternMatcherPass` 灏?silu+mul 铻嶅悎涓?`tensor_cast.swiglu`锛涘惈閲忓寲鍙樹綋 `tensor_cast.dequant_swiglu_quant` | `SwiGlu` / `DequantSwigluQuant` |
| **split_qkv_rmsnorm_rope 铻嶅悎** | TensorCast 鍒嗗埆 dispatch qkv_split銆乺ms_norm銆乤pply_rope | 鏂板缂栬瘧 Pass 铻嶅悎涓?`tensor_cast.split_qkv_rmsnorm_rope` | `split_qkv_rmsnorm_rope_kernel` |
| **KvRmsNormRopeCache 铻嶅悎** | TensorCast 鍒嗗埆 dispatch rms_norm銆乤pply_rope銆乺eshape_and_cache | 鏂板缂栬瘧 Pass 铻嶅悎涓?`tensor_cast.kv_rmsnorm_rope_cache` | `KvRmsNormRopeCache` |
| `multihead_latent_attention` | 鍗曚竴 dispatch 鑺傜偣锛屽寘鍚?TransposeBatchMatMul 脳 2 + FusedInferAttentionScore | 鎷嗗垎涓虹嫭绔?dispatch 绠楀瓙 |  |
| `aten.topk` / MoeGatingTopK | aten.topk 鍦ㄩ潪 MoE 鍦烘櫙涔熻璋冪敤 | 鍖哄垎 MoE 璺敱 topk 鍜岄噰鏍?topk |  |
| 鍏朵粬娼滃湪宸紓 | 闇€閫愭ā鍨嬫瘮瀵圭‘璁?| 閫愪竴浜哄伐閫傞厤 |  |

**寤鸿鏂瑰紡**锛氫互 DeepSeek-V3 鍜?Qwen3-32B 鐨?kernel_details.csv 涓哄熀鍑嗭紝瀵煎嚭 TensorCast dispatch trace 閫愪竴姣斿锛岄€氳繃鏂板 custom op 鍜岀紪璇?Pass 娑堥櫎宸紓銆係wiGlu 铻嶅悎浼樺厛绾ф渶楂橈紙Decode 闃舵鍗犳瘮 2.4%锛夈€?

### 9.2 CompositePerformanceModel 椤跺眰璋冨害鍣?

褰撳墠鏂规涓?`ProfilingPerformanceModel` 瀵规湭鏀跺綍绠楀瓙鍐呴儴鍏滃簳鍥為€€鑷?`AnalyticPerformanceModel`锛岃繖鏄?v1.1 鐨勫姟瀹炲仛娉曘€傞暱鏈熸潵鐪嬶紝寤鸿璁捐涓€涓?`CompositePerformanceModel`锛屾寜浼樺厛绾х粍鍚堝涓?PerformanceModel锛屽疄鐜板彲閰嶇疆鐨勯檷绾х瓥鐣ワ紙`Profiling 鈫?Empirical 鈫?Analytic`锛夈€傚紩鍏ュ悗 `ProfilingPerformanceModel` 涓嶅啀闇€瑕佸唴閮ㄦ寔鏈?`fallback_model`锛屼笁绉嶆ā鍨嬪畬鍏ㄨВ鑰︺€?

### 9.3 鍏朵粬鎵╁睍寤鸿

- **璺ㄧ‖浠舵硾鍖?*锛氭敮鎸佹洿澶?DeviceProfile锛堝 Atlas A2銆丟PU锛夛紝闇€涓烘瘡绉嶇‖浠剁嫭绔嬮噰闆嗘暟鎹?
- **鑷姩鍖?CI**锛氶殢 VLLM-Ascend / CANN 鐗堟湰鍙戝竷鑷姩瑙﹀彂鏁版嵁閲囬泦娴佹按绾?
- **棰勬彃鍊间紭鍖?*锛氬弬鑰?AI Configurator 鐨?`_extrapolate_data_grid`锛屽湪鏁版嵁搴撳姞杞芥椂棰勫～鍏呭父鐢ㄧ綉鏍肩偣
- **SOL 鏁版嵁鏍℃**锛氱敤 Roofline 鐞嗚涓嬬晫鏍℃寮傚父娴嬮噺鍊硷紝纭繚 `measured >= SOL`

---

## 10. 渚濊禆椤?

**鏂板**锛歚scipy`锛堟彃鍊硷級銆乣packaging`锛堢増鏈В鏋愶級
**鐜版湁**锛歚pandas`銆乣numpy`銆乣pyyaml`銆乣torch`
**鍙€?*锛歚pyarrow`锛圥arquet 鏍煎紡鏀寔锛屽垵鏈熷彲浠呯敤 CSV锛?

---

## 11. 鍙傝€冭祫鏂?

- [vLLM Ascend GitHub](https://github.com/vllm-project/vllm-ascend)
- [vLLM Ascend 鍙戝竷璇存槑](https://docs.vllm.ai/projects/ascend/en/main/user_guide/release_notes.html)
- [vLLM Ascend GitHub Releases](https://github.com/vllm-project/vllm-ascend/releases)
- [vLLM Ascend Profiling 鎸囧崡](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/service_profiling_guide.html)
- [鍗庝负鏄囪吘 Profiler 鏂囨。](https://support.huaweicloud.com/intl/en-us/bestpractice-modelarts/modelarts_llm_infer_5906034.html)
- [AI Configurator](https://github.com/ai-dynamo/aiconfigurator)锛圢VIDIA LLM 鎺ㄧ悊鎬ц兘棰勪及宸ュ叿锛?
- [Intel NPU Cost Model](https://github.com/intel/npu-nn-cost-model)

---

## 闄勫綍锛氭灦鏋勮璁″垎鏋?

### ProfilingPerformanceModel vs EmpiricalPerformanceModel

**闂**锛氬熀浜庢暟鎹簱鏌ヨ鐨?Profiling 妯″瀷鍜屽熀浜?JIT microbenchmark 鐨?Empirical 妯″瀷搴旇鍚堝苟杩樻槸鍒嗗紑锛?

**涓夌鍊欓€夋柟妗?*锛?
- **鏂规 A锛堝悎骞讹級**锛氬湪 EmpiricalPerformanceModel 鍐呭紩鍏?DataSource 鎶借薄灞傘€傜己鐐癸細绫诲鏉傚害鑶ㄨ儉锛屽唴閮?fallback 涓?CompositePerformanceModel 鑱岃矗閲嶅彔锛岃繚鍙嶅紑闂師鍒欍€?
- **鏂规 B锛堥噰绾筹細鍒嗗紑锛?*锛氫袱涓嫭绔?PerformanceModel + 鍏变韩 PerfDatabase 鏁版嵁灞傘€備紭鐐癸細鍗曚竴鑱岃矗锛岀嫭绔嬪彲娴嬭瘯锛孍mpiricalPerformanceModel 鏃犻渶鏀瑰姩鏍稿績閫昏緫銆?
- **鏂规 C锛堢户鎵匡級**锛欴atabasePerformanceModel 缁ф壙 EmpiricalPerformanceModel銆傜己鐐癸細缁ф壙鑰﹀悎锛屾棤娉曞崟鐙娇鐢?Database 鑰屼笉鐢?JIT銆?

**缁撹**锛氳€﹀悎鍦ㄦ暟鎹眰锛岃В鑰﹀湪妯″瀷灞傘€備袱鑰呯殑鍖哄埆浠呭湪浜庢暟鎹綍鏃朵骇鐢燂紙绂荤嚎 vs JIT锛夈€佽鍐欐ā寮忥紙鍙 vs 璇诲啓锛夈€佹暟鎹瘑搴︼紙绯荤粺鎬ч噰闆?vs 鎸夐渶瑙﹀彂锛夈€傛洿浼橀泤鐨勫仛娉曟槸鎶藉彇鍏变韩鐨?`PerfDatabase` 鏁版嵁灞傦紝鑰岄潪鍚堝苟涓や釜 Model銆?

---

## Change Log

### v1.0 鈫?v1.1

1. 浠庡崟涓€ `ProfilingPerformanceModel`锛堝唴鍚?Roofline 鍏滃簳锛夐噸鏋勪负涓変釜鐙珛 PerformanceModel 骞跺垪锛孯oofline 鍏滃簳涓婄Щ鑷?Model 灞?`fallback_model`
2. `PerfDatabase` 浠?Profiling 涓撶敤鍖呮彁鍗囦负鍏变韩鏁版嵁灞傦紝鍚屾椂涓?Profiling锛堝彧璇伙級鍜?Empirical锛堣鍐欐寔涔呭寲缂撳瓨锛夋湇鍔★紱`EmpiricalPerformanceModel` 鏂板鍙€?`cache_db` 鍙傛暟
3. 鏂板 `OperatorKey` 鍏变韩鎶借薄锛屽皝瑁?`OpInvokeInfo 鈫?鏌ヨ key` 杞崲閫昏緫
4. `PerfDatabase` API 浠?`(system, backend, version)` 涓夊厓缁勬敼涓?`data_path` 璺緞鐩翠紶锛屽垹闄?`VersionManager`
5. Schema 绮掑害浠庢寜鎶借薄绠楀瓙绫诲瀷鍒嗙粍锛垀6 绫伙級鏀逛负鎸?kernel_details.csv 纭欢鍐呮牳涓€涓€瀵归綈锛垀17 涓級锛宍DimensionSpec` 浠庢灇涓剧畝鍖栦负 `interpolatable: bool`
6. YAML 鏄犲皠鏍煎紡浠庢寜 Schema 鍒嗙粍鏀逛负鎵佸钩鐨?`tensorcast_op_to_schema`锛?:1锛?
7. QueryEngine 浠?4 绾ч檷绾э紙鍚?Roofline锛夌簿绠€涓?3 绾э紙绮剧‘鈫掓彃鍊尖啋澶栨帹锛?
8. 瀛樺偍鏍煎紡浠?Parquet 涓轰富鏀逛负 CSV 涓轰富锛涙暟鎹噰闆嗕粠鍗曚竴 L1+L2 鎵╁睍涓轰笁绉嶅彲閫夋柟妗堬紙鍏ㄦā鍨?Profiling / 浠跨湡椹卞姩 / 绾鍏ワ級
9. 鍩轰簬 DeepSeekV3/Qwen3 瀹炴祴 Profiling 鏁版嵁淇浜嗗椤圭畻瀛愭槧灏勶紙濡?`static_quant_linear` 鈫?`quant_batch_matmul`锛夛紝鏂板 TransposeBatchMatMul銆両nterleaveRope銆丄llGather銆丮oeGating 鐙珛 Schema
10. 鏁版嵁閲囬泦娴佹按绾夸粠 TensorCast 鍖呭唴绉昏嚦浠撳簱椤跺眰 `perf_database/` 鐙珛瀛愮郴缁燂紱寮€鍙戣鍒掍粠 5 闃舵鍗曞洟闃熸敼涓?3 闃舵鍙屽洟闃熷垎宸?

