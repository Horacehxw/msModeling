# MicroBench 涓庢暣缃?Profiling 宸紓鍒嗘瀽

## 鑼冨洿

- 鎵弿鐩綍: `G:\浠跨湡寮€鍙慭profiling\浠跨湡&瀹炴祴瀵规瘮`
- 鍙垎鏋?`tools/perf_data_collection/op_replay` 涓嬪凡缁忓瓨鍦ㄥ搴?`*_run.py` 鐨勭畻瀛?- 鏈夋晥姣旇緝鍓嶆彁: CSV 鍚屾椂鍖呭惈 `Average Duration(us)` 鍜?`Profiling Average Duration(us)`

鏈鍏卞尮閰嶅埌 451 鏉℃牱鏈紝瑕嗙洊 15 涓彲姣旇緝绠楀瓙銆?
浠ヤ笅 2 涓畻瀛愬湪璇ョ洰褰曚笅娌℃湁 `Average Duration(us)`锛屽洜姝ゆ棤娉曞仛 microbench vs profiling 瀵规瘮:

- `AddRmsNormBias`
- `RINGMLAPrefillBF16Kernel`

## 鍒ゅ畾鍙ｅ緞

灏嗏€滃樊璺濆緢澶р€濆畾涔変负鍚屾椂婊¤冻:

- `max(Profiling / MicroBench, MicroBench / Profiling) >= 1.5`
- `|Profiling Average Duration(us) - Average Duration(us)| >= 5`

杩欎釜鍙ｅ緞鍙互杩囨护鎺夊彧鏈夋瘮渚嬪ぇ浣嗙粷瀵瑰€煎緢灏忕殑鍣０鐐广€?
## 鎬讳綋缁撹

鏈€绐佸嚭鐨勫樊寮備富瑕侀泦涓湪涓ょ被绠楀瓙:

1. `AI_VECTOR_CORE` / `AI_VECTOR` 涓诲鐨勮瀛樺瀷绠楀瓙  
   浠ｈ〃: `GatherV2`銆乣MaskedFill`銆乣DynamicQuant`銆乣AscendQuantV2`銆乣RmsNorm`銆乣Add`
2. `AI_CORE` 鎴?`MIX_AIC` 鐨勮绠楀瀷绠楀瓙  
   浠ｈ〃: `MatMulV2`銆乣FusedInferAttentionScore`

涓€涓槑鏄炬ā寮忔槸:

- 瀵瑰ぇ閲?AIV 绠楀瓙锛宍MicroBench aicore_time(us)` 鍜?`Profiling Average aicore_time(us)` 閮芥槸 0
- 鐪熸鏀惧ぇ鐨勪笉鏄?AICORE锛岃€屾槸 `aiv_time(us)` 鎴栬€呮€绘椂寤朵腑鐨勯潪鏍告椿璺冩椂闂?- 杩欒鏄庡樊寮傛洿澶氭潵鑷瀛樸€佷笂涓嬫枃銆佹暟鎹竷灞€銆佺储寮?鎺╃爜褰㈡€併€佽皟搴﹀紑閿€锛岃€屼笉鏄€滅畻瀛愬叕寮忔湰韬彉浜嗏€?
## 鎸夌畻瀛愯仛鍚堢粨鏋?
| 绠楀瓙 | 鏍锋湰鏁?| 寮傚父鏍锋湰鏁?| 鏈€澶ф椂寤跺€嶇巼 | 鏈€澶х粷瀵瑰樊鍊?us) | 缁撹 |
| --- | ---: | ---: | ---: | ---: | --- |
| `DynamicQuant` | 39 | 9 | 2.339x | 12.850 | 鎸佺画鍋忓ぇ锛孉IV 涓诲 |
| `AscendQuantV2` | 42 | 9 | 1.982x | 15.451 | 鎸佺画鍋忓ぇ锛孉IV 涓诲 |
| `GatherV2` | 13 | 8 | 3.897x | 61.030 | 鏈€鏄庢樉鐨勮瀛?绱㈠紩鍨嬪亸宸?|
| `RmsNorm` | 26 | 6 | 2.131x | 15.356 | 鎸佺画鍋忓ぇ锛孉IV 涓诲 |
| `MaskedFill` | 18 | 5 | 4.072x | 23.932 | 鎬绘椂寤舵斁澶ф槑鏄撅紝浣嗘牳鏃堕棿鏀惧ぇ鏈夐檺 |
| `Add` | 44 | 4 | 2.655x | 17.248 | 灏?shape 涓嬩笂涓嬫枃寮€閿€鏁忔劅 |
| `MatMulV2` | 54 | 3 | 1.757x | 148.781 | AICORE 鏃堕棿鍚屾鏀惧ぇ |
| `InterleaveRope` | 16 | 2 | 1.908x | 9.061 | 灏忓紶閲忓満鏅亸鏁忔劅 |
| `FusedInferAttentionScore` | 4 | 2 | 1.757x | 20.942 | AICORE 鍜?AIV 鍚屾椂鏀惧ぇ |
| `Slice` | 26 | 1 | 1.825x | 8.716 | 灏戦噺 case 涓?profiling 鏇村揩 |
| `SoftmaxV2` | 5 | 0 | 1.137x | 5.382 | 鍩烘湰绋冲畾 |
| `Sort` | 5 | 0 | 1.080x | 12.350 | 鍩烘湰绋冲畾 |

## 閲嶇偣寮傚父绠楀瓙

### 1. `GatherV2`

浠ｈ〃 case:

- 琛屽彿: 14
- 杈撳叆: `"16160,7168;1548;1"`
- 杈撳嚭: `"1548,7168"`
- `Average Duration(us) = 18.780`
- `Profiling Average Duration(us) = 73.177`
- 鏃跺欢鍊嶇巼: `3.897x`
- `MicroBench aicore_time(us) = 0`
- `Profiling Average aicore_time(us) = 0`
- `MicroBench aiv_time(us) = 16.341`
- `Profiling Average aiv_time(us) = 69.489`

鍒嗘瀽:

- 杩欐槸鍏稿瀷鐨?AIV 璁垮瓨/绱㈠紩绠楀瓙锛屼笉鏄?AICORE 绠楀瓙
- 宸紓鍑犱箮鍏ㄩ儴浣撶幇鍦?`aiv_time`
- `16160 x 7168` 鐨勫ぇ婧?tensor 閰嶅悎 `1548` 闀垮害绱㈠紩锛屾暣缃戜腑鏇村鏄撳彈鍒版簮 tensor 瀹為檯椹荤暀浣嶇疆銆乧ache/TLB 鍛戒腑鐜囥€侀〉琛ㄨ闂拰鐪熷疄绱㈠紩鍒嗗竷褰卞搷
- microbench 鏇村儚鈥滅悊鎯冲寲鍗曠畻瀛愬洖鏀锯€濓紝瀵圭湡瀹炴暣缃戜腑鐨勫唴瀛樺帇鍔涖€佸墠鍚庣畻瀛愬共鎵般€佺储寮曞紶閲忕敓鎴愭柟寮忓鐜颁笉瓒?
鍙兘鍘熷洜:

- replay 杈撳叆铏界劧 shape 瀵归綈锛屼絾娌℃湁澶嶇幇鐪熷疄绱㈠紩鍊煎垎甯?- 鏁寸綉涓簮 tensor 鍙兘鏄墠搴忕畻瀛愯緭鍑猴紝鍐呭瓨灞€閮ㄦ€ф槑鏄惧樊浜庡崟绠楀瓙
- `GatherV2` 鏈韩鏋佸害鍚冨甫瀹藉拰鍦板潃璁块棶妯″紡锛屽洜姝ゅ涓婁笅鏂囨洿鏁忔劅

### 2. `MaskedFill`

浠ｈ〃 case:

- 琛屽彿: 5
- 杈撳叆: `"2,129280;2,129280;"`
- 杈撳嚭: `"2,129280"`
- `Average Duration(us) = 7.360`
- `Profiling Average Duration(us) = 29.973`
- 鏃跺欢鍊嶇巼: `4.072x`
- `MicroBench aicore_time(us) = 0`
- `Profiling Average aicore_time(us) = 0`
- `MicroBench aiv_time(us) = 5.832`
- `Profiling Average aiv_time(us) = 6.844`

鍒嗘瀽:

- 鎬绘椂寤舵斁澶т簡 4 鍊嶏紝浣?`aiv_time` 鍙粠 `5.832us` 澧炲姞鍒?`6.844us`
- 杩欒鏄庡樊寮備笉涓昏鏉ヨ嚜鏍告湰浣撴墽琛屾椂闂达紝鑰屾槸鏍稿寮€閿€
- 瀵?`MaskedFill` 杩欑鍖呭惈甯冨皵 mask 鐨勭畻瀛愶紝鏁寸綉閲屽父瑙侀澶栨垚鏈槸 mask 鐢熸垚/杞崲銆佸悓姝ャ€佽皟搴︾瓑寰呭拰涓婁笅娓镐緷璧?
鍙兘鍘熷洜:

- replay 娌℃湁澶嶇幇鐪熷疄 mask 鐨勬潵婧愰摼璺紝鍙鐜颁簡鏈€缁堝紶閲?shape
- 甯冨皵 mask 鍦ㄦ暣缃戜腑鍙兘浼撮殢棰濆 cast銆乥roadcast銆佷緷璧栫瓑寰?- 杩欑被灏忔牳鍦ㄦ暣缃戜腑瀵?launch 寮€閿€鍜岄槦鍒楃瓑寰呴潪甯告晱鎰?
### 3. `DynamicQuant`

浠ｈ〃 case:

- 琛屽彿: 32
- 杈撳叆: `"1030,2304"`
- 杈撳嚭: `"1030,2304;1030"`
- `Average Duration(us) = 15.320`
- `Profiling Average Duration(us) = 26.348`
- 鏃跺欢鍊嶇巼: `1.720x`
- `MicroBench aicore_time(us) = 0`
- `Profiling Average aicore_time(us) = 0`
- `MicroBench aiv_time(us) = 13.721`
- `Profiling Average aiv_time(us) = 22.504`

鍒嗘瀽:

- 杩欐槸灏?shape銆佸己璁垮瓨銆佸己缁熻鐗瑰緛渚濊禆鐨勯噺鍖栫畻瀛?- `aiv_time` 鍑犱箮鍚屾鏀惧ぇ鍒?3 鍊嶏紝璇存槑鐪熷疄鏁寸綉涓殑鍚戦噺璁垮瓨璺緞鏇撮噸
- 鍚岀被鐜拌薄鍦?`AscendQuantV2` 涓婁篃寰堟槑鏄撅紝璇存槑鈥滈噺鍖栫被灏忔牳鈥濇暣浣撴湁绯荤粺鎬т綆浼?
鍙兘鍘熷洜:

- replay 娌″鐜板墠搴忔縺娲诲€煎垎甯冿紝瀵艰嚧 scale / maxabs 鐩稿叧璺緞涓嶄竴鑷?- 灏忕煩闃甸噺鍖栧湪鏁寸綉涓洿鍙楄皟搴︺€乧ache 鍏变韩鍜屽墠搴忚緭鍑哄竷灞€褰卞搷
- microbench 涓嬫暟鎹洿杩炵画銆佹洿骞插噣锛岃瀛樺帇鍔涘亸灏?
### 4. `AscendQuantV2`

浠ｈ〃 case:

- 琛屽彿: 19
- 杈撳叆: `"9,2048;2048;2048"`
- 杈撳嚭: `"9,2048"`
- `Average Duration(us) = 6.520`
- `Profiling Average Duration(us) = 12.924`
- 鏃跺欢鍊嶇巼: `1.982x`
- `MicroBench aicore_time(us) = 0`
- `Profiling Average aicore_time(us) = 0`
- `MicroBench aiv_time(us) = 3.847`
- `Profiling Average aiv_time(us) = 9.225`

鍒嗘瀽:

- 涓?`DynamicQuant` 涓€鏍凤紝灞炰簬閲忓寲绫?AIV 绠楀瓙
- 宸紓涓昏钀藉湪 `aiv_time`
- 澶氫釜 shape 涓婇兘閲嶅鍑虹幇锛岃鏄庝笉鏄绔嬬偣

鍙兘鍘熷洜:

- 閲忓寲杈撳叆鏁版嵁鍒嗗竷涓?microbench 闅忔満杈撳叆宸紓杈冨ぇ
- 鏁寸綉涓婁笅鏂囦腑鐨勫紶閲忓竷灞€銆乧ache 姹℃煋銆侀槦鍒楃珵浜夋病鏈夎 replay 鎹曡幏

### 5. `RmsNorm`

浠ｈ〃 case:

- 琛屽彿: 27
- 杈撳叆: `"515,1536;1536"`
- 杈撳嚭: `"515,1536;515,1"`
- `Average Duration(us) = 13.580`
- `Profiling Average Duration(us) = 28.936`
- 鏃跺欢鍊嶇巼: `2.131x`
- `MicroBench aicore_time(us) = 0`
- `Profiling Average aicore_time(us) = 0`
- `MicroBench aiv_time(us) = 10.077`
- `Profiling Average aiv_time(us) = 24.274`

鍒嗘瀽:

- `RmsNorm` 鐨勫樊寮傛槸绋冲畾鑰屽箍娉涚殑锛屼笉鏄崟鐐瑰皷宄?- AIV 鏃堕棿杩戜技鍚屾鏀惧ぇ锛岃鏄庨棶棰樹粛鐒朵富瑕佸湪璁垮瓨鍜屽悜閲忓綊绾﹂摼璺?- `515/518/1030/1545` 杩欎簺涓嶈鍒?token 闀垮害涓婂亸宸挨鍏舵槑鏄?
鍙兘鍘熷洜:

- 涓嶈鍒?token 鏁板鑷村疄闄?kernel tile 鍒╃敤鐜囦笌 microbench 鍋囪涓嶄竴鑷?- replay 鏈鐜版暣缃戜腑鐨勫墠搴忓紶閲忛┗鐣欑姸鎬佸拰褰掔害鍓嶅悗渚濊禆

### 6. `MatMulV2`

浠ｈ〃 case:

- 琛屽彿: 29
- 杈撳叆: `"6,512;4096,512"`
- 杈撳嚭: `"6,4096"`
- `Average Duration(us) = 6.920`
- `Profiling Average Duration(us) = 12.157`
- 鏃跺欢鍊嶇巼: `1.757x`
- `MicroBench aicore_time(us) = 5.406`
- `Profiling Average aicore_time(us) = 9.527`

鍒嗘瀽:

- 杩欐槸灏戞暟鈥滄椂寤跺樊寮傚拰 `aicore_time` 宸紓鍚屾鈥濈殑鍏稿瀷渚嬪瓙
- 璇存槑闂涓嶅彧鏄?launch 鎴栧閮ㄨ皟搴︼紝鑰屾槸瀹為檯 AICORE 鎵ц璺緞鏈韩鏇撮噸
- 鍙︿竴鏉℃洿澶х殑缁濆宸€?case 鏄?`"3,7168;16160,7168" -> "3,16160"`锛屾€诲樊鍊艰揪鍒?`148.781us`锛宍aicore_time` 涔熷悓姝ュ鍔?`139.571us`

鍙兘鍘熷洜:

- microbench 澶嶇幇鐨勮緭鍏ユ牸寮忔垨鍐呴儴 format 鍙兘涓庢暣缃戠湡瀹炶矾寰勪笉瀹屽叏涓€鑷?- 鏁寸綉涓殑 MatMul 鍙兘鍛戒腑浜嗕笉鍚?tiling / transpose / pack 璺緞
- 灏?batch銆佸ぇ K/N 鍦烘櫙涓嬶紝鐪熷疄璋冨害绛栫暐鍜屽崟绠楀瓙鍥炴斁瀹规槗鍒嗗弶

### 7. `FusedInferAttentionScore`

浠ｈ〃 case:

- 琛屽彿: 4
- 杈撳叆鍖呭惈: `q=(4,16,1,512)`銆乣k/v=(1135,1,128,512)`銆乺ope 鐩稿叧杈撳叆
- 杈撳嚭: `"16,4,1,512;"`
- `Average Duration(us) = 27.680`
- `Profiling Average Duration(us) = 48.622`
- 鏃跺欢鍊嶇巼: `1.757x`
- `MicroBench aicore_time(us) = 23.203`
- `Profiling Average aicore_time(us) = 38.906`
- `MicroBench aiv_time(us) = 24.837`
- `Profiling Average aiv_time(us) = 45.319`

鍒嗘瀽:

- 杩欐槸 `MIX_AIC` 绠楀瓙锛孉ICORE 鍜?AIV 閮芥槑鏄惧彉鎱?- 璇存槑鏁寸綉涓?attention 鐩稿叧鏁版嵁缁勭粐銆乵ask銆乺ope銆乴ayout 鏇存帴杩戔€滃鏉傜湡瀹炶矾寰勨€?- replay 铏界劧 shape 澶嶇幇浜嗭紝浣嗗唴閮ㄦ暟鎹竷灞€鍜屼笂涓嬫枃渚濊禆鍙兘浠嶇劧鍋忕悊鎯冲寲

鍙兘鍘熷洜:

- mask / rope / seq-len / layout 缁嗚妭娌℃湁瀹屽叏璐磋繎鏁寸綉
- attention 鍓嶅悗閾捐矾鐨勭紦瀛樼姸鎬併€佸悓姝ュ拰娴佹按閲嶅彔鏃犳硶闈犲崟绠楀瓙 replay 瀹屾暣澶嶇幇

## 鐩稿绋冲畾鐨勭畻瀛?
浠ヤ笅绠楀瓙褰撳墠 replay 涓庢暣缃?profiling 宸茬粡姣旇緝鎺ヨ繎:

- `SoftmaxV2`: 鏈€澶у€嶇巼 `1.137x`
- `Sort`: 鏈€澶у€嶇巼 `1.080x`

璇存槑杩欎袱涓畻瀛愮殑 `*_run.py` 澶嶇幇璐ㄩ噺鐩稿鏇村ソ锛屾垨鑰呰繖绫荤畻瀛愭湰韬鏁寸綉涓婁笅鏂囦笉鏁忔劅銆?
## 鐗规畩鎯呭喌

### `Slice`

浠ｈ〃 case:

- 琛屽彿: 2
- 杈撳叆: `"2048,2112;2;2"`
- 杈撳嚭: `"2048,1536"`
- `Average Duration(us) = 19.280`
- `Profiling Average Duration(us) = 10.564`

杩欑被 case 鏄?profiling 鍙嶈€屾洿蹇紝璇存槑褰撳墠 `Slice_run.py` 寰堝彲鑳借蛋浜嗘洿淇濆畧鎴栨洿閫氱敤鐨?replay 璺緞锛屾病鏈夊畬鍏ㄥ懡涓暣缃戦噷鐨勯珮鏁?fast path銆?
## 寤鸿

### 浼樺厛绾ф渶楂?
1. 浼樺厛琛ュ己 `GatherV2`銆乣MaskedFill`銆乣DynamicQuant`銆乣AscendQuantV2`銆乣RmsNorm`
2. 瀵硅繖浜涚畻瀛愶紝涓嶈鍙鐜?shape锛屼紭鍏堝鐜?
   - 绱㈠紩鍊煎垎甯?   - mask 绋€鐤忔ā寮?   - 杈撳叆鏁板€煎垎甯?   - 涓婁笅娓稿紶閲忔槸鍚﹁繛缁€佹槸鍚﹀鐢ㄥ墠搴忚緭鍑?
### 瀵?AIV 绠楀瓙

1. 鎶ュ憡涓?`aicore_time=0` 鐨勭畻瀛愶紝涓嶈缁х画鎶婇棶棰樺綊鍥犲埌 AICORE
2. 鏇村簲鍏虫敞:
   - `aiv_time`
   - launch/dispatch 寮€閿€
   - 鐪熷疄绱㈠紩鍜屽竷灏?mask 鐨勬瀯閫犳垚鏈?   - layout / contiguous 鐘舵€?
### 瀵?AICORE / MIX 绠楀瓙

1. `MatMulV2` 鍜?`FusedInferAttentionScore` 寤鸿琛ュ厖:
   - 杈撳叆鏍煎紡涓?internal format 瀵归綈妫€鏌?   - 鏄惁缂哄け transpose / pack / rope / mask 鐨勭湡瀹炶矾寰?   - 鏄惁闇€瑕佹妸涓婃父涓棿鎬佷竴骞跺洖鏀撅紝鑰屼笉鏄粎鍥炴斁鏈€缁堣緭鍏?shape

## 缁撹

褰撳墠 replay 浣撶郴瀵?`SoftmaxV2`銆乣Sort` 杩欑被绠楀瓙宸茬粡姣旇緝鍙俊锛屼絾瀵瑰ぇ閲?AIV 涓诲鐨勫皬鏍稿拰璁垮瓨鍨嬬畻瀛愪粛鐒跺亸涔愯锛屽挨鍏舵槸 `GatherV2`銆乣MaskedFill`銆乣DynamicQuant`銆乣AscendQuantV2`銆乣RmsNorm`銆? 

杩欎簺宸紓澶у涓嶆槸 AICORE 绠楀姏鏈韩鐨勯棶棰橈紝鑰屾槸 replay 瀵圭湡瀹炴暣缃戜笂涓嬫枃澶嶇幇涓嶅锛屽鑷磋瀛樸€佺储寮曘€乵ask銆佽皟搴﹀拰 layout 鐩稿叧寮€閿€琚綆浼般€?
