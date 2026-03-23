# CHANGELOG 2026-03-12

璁捐鏂囨。 v1.3 鈫?v1.3.1 + 宸ヤ綔璁″垝 v3 鈫?v3.1

---

## 璁捐鏂囨。鍙樻洿 (v1.3 鈫?v1.3.1)

### 鏂板

- **搂2.2**: CANN 8.5 DispatchFFNCombine 瓒呯骇铻嶅悎璇存槑锛?5.3% DSV3 Decode锛?
- **搂3.3**: 瀹屾暣鐗堟湰瀛楃涓茬洰褰曞懡鍚嶇害瀹氾紙`vllm{ver}_torch{ver}_cann{ver}`锛?
- **搂4.2**: Input 鏁伴噺涓嶅尮閰嶅鐞嗚鍒欙紙QuantBatchMatmulV3 2鈫?, ReshapeAndCacheNdKernel 4鈫?, TensorMove 2鈫?锛?
- **搂4.7**: HCCL v8.5 鏁版嵁璐ㄩ噺娉ㄦ剰浜嬮」锛圝IT 鍒濆鍖栭棶棰?+ 鍗?session 淇鏂规锛?
- **搂9.1**: DispatchFFNCombine gap 椤癸紙CANN 8.5 鏂板锛?5.3% DSV3锛?
- **搂9.1**: TP Padding bug 宸蹭慨澶嶈褰曪紙59cf184锛?
- **搂9.1**: MC2 宸查獙璇佽褰曪紙3f82c2b锛孊F16 + W8A8锛?

### 鍙樻洿

- **搂1.4**: 鐩爣鍚庣 vllm-ascend 0.13.0 鈫?0.15.0锛圕ANN 8.5锛宼orch 2.9.0锛?
- **搂3.3**: 鏁版嵁瀛樺偍璺緞鏇存柊锛堝惈 CANN 8.3 legacy + CANN 8.5 production target锛?
- **搂4.2**: ProfilingDataSource 鏋勯€犲弬鏁?`comm_grid` 鈫?`device_profile`锛堢粺涓€纭欢鍙傛暟璁块棶锛?
- **搂4.5**: op_mapping.yaml cann_version 鏇存柊涓?8.5锛宑ommunication_data_ref 鏇存柊
- **搂9.1**: KvRmsNormRopeCache 鐘舵€?"浠嶅紑鏀? 鈫?"宸插叧闂?锛坢lapo 宸茶鐩栵級
- **搂9.1**: MC2 铻嶅悎鐘舵€?"浠嶅紑鏀? 鈫?"宸查獙璇?
- **搂9.2**: MLA 鍒嗚В鐘舵€佹洿鏂帮紙composite 鏌ヨ宸插疄鐜帮級

### 鏋舵瀯鍐崇瓥

1. **ProfilingDataSource 鎺ュ彛缁熶竴**锛氭墍鏈夋煡璇㈣矾寰勶紙璁＄畻 CSV + 閫氫俊 HCCL锛夐€氳繃 `device_profile` 瀵硅薄璁块棶锛屾浛浠ｅ垎鏁ｇ殑 `comm_grid` 鍙傛暟
2. **ModelRunnerMetrics 澶氭ā鍨嬫敮鎸?*锛歚execution_time_s: float` 鈫?`Dict[str, float]`锛涙柊澧?`tps_per_model: Dict[str, float]`
3. **Input 鏁伴噺涓嶅尮閰?*锛氶噰鐢?kernel_type 绾ц繃婊よ鍒欙紙闈為€氱敤鎴柇锛夛紝鍥?NPU kernel 瀹為檯鎺ユ敹鐨?input 鍙兘鍖呭惈 TC 灞備笉鍙鐨勫唴閮ㄥ弬鏁?

---

## 宸ヤ綔璁″垝鍙樻洿 (v3 鈫?v3.1)

### 浜哄憳鍙樺姩

- **XJT 鈫?LJW**锛歑JT 宸ヤ綔浜ゆ帴缁?LJW锛孡JW 3.17 璧峰叏鑱屾姇鍏?
- **XJT 宸插畬鎴愪氦浠?*锛欰1/A2/A3 + MC2 楠岃瘉 + KvRmsNormRopeCache 纭

### 鐩爣璋冩暣

- **E2E 楠岃瘉鐗堟湰**锛欳ANN 8.3 鈫?**CANN 8.5**锛坴llm 0.15.0 + torch 2.9.0锛?
- **鏁版嵁鐩綍**锛歚vllm0.15.0_torch2.9.0_cann8.5/`
- **鏁版嵁閲囬泦宸ュ叿閾?*锛? 涓?鈫?9 涓?

### Phase 1 杩涘睍锛堟埅鑷?3.12锛?

| 鐘舵€?| 浠诲姟 |
|------|------|
| 鉁?宸插畬鎴?| A1, A2, A3, B1, C1, C2, C3, C6, C7, C8, C9, D1 |
| 馃攧 閮ㄥ垎瀹屾垚 | B2(draft), C10(CSV 浜у嚭锛宼ier=0 缂哄け) |
| 馃攧 杩涜涓?| C4, D2, D3, D4 |

**鍏抽敭鍙戠幇**锛?
- TC compile 璺緞涓嶅彲琛岀敤浜?op_mapping 楠岃瘉锛圸ZY锛夛紝鏀圭敤 AI/skill 鏂规
- KvRmsNormRopeCache 琚?mlapo 瑕嗙洊锛孎1 scope 缂╁噺
- TP Padding bug 宸蹭慨澶嶏紙鍏ㄥ眬 input 鈫?MoE layer-local锛屾秷闄?DSV3 Decode 8x 楂樹及锛?
- Qwen3 BF16 楠岃瘉瑕嗙洊 97.01%锛孌SV3 W8A8 瑕嗙洊 98.02%

### 鏂板浠诲姟

- **C11**: DispatchFFNCombine 瀛愬唴鏍告暟鎹噰闆嗭紙HDY, C11-1 鎴 3.13, C11-2 鎴 3.18, C11-3 鎴 3.19锛?
- **F1 scope 鍙樻洿**: ~~KvRmsNormRopeCache pass~~ 鈫?DispatchFFNCombine 鍙鎬ц瘎浼帮紙LJW锛?
- **F2**: 鏉′欢鎬?DispatchFFNCombine pass 瀹炵幇

### 鏂板椋庨櫓

| # | 椋庨櫓 | 姒傜巼 |
|---|------|------|
| R10 | 鍗曠畻瀛愪笌鏁寸綉绠楀瓙鑰楁椂 gap锛圱CX 3.12 鎻愬嚭锛?| 楂?|
| R11 | CANN 8.5 DispatchFFNCombine 瑕嗙洊涓嶈冻锛?5.3% DSV3锛?| 楂?|
| R12 | LJW 涓婃墜鍛ㄦ湡锛?-3 澶╋級 | 涓?|

### 宸插叧闂闄?

- **R9**锛圶JT琚叾浠栭」鐩嫋浣忥級锛欰1/A2/A3 宸插畬鎴愶紝XJT鈫扡JW 浜ゆ帴

### 鏍稿績鍋囪鏂板

- CANN 8.5 DispatchFFNCombine 鍙€氳繃 composite 鍒嗚В鏌ヨ瑕嗙洊锛堥獙璇侊細C11 + Phase 2锛?

