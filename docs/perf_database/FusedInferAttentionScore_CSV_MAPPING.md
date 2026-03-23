# FusedInferAttentionScore CSV 鍙傛暟鏄犲皠璇存槑

鏈枃妗ｆ暣鐞?`FusedInferAttentionScore.csv` 涓?`Input Shapes / Input Data Types / Input Formats` 涓?
`torch_npu.npu_fused_infer_attention_score` 鎺ュ彛鍙傛暟鐨勫搴斿叧绯伙紝骞惰ˉ鍏呰繖浠?CSV 鐨勫疄闄?case 鍒嗙被銆?

閫傜敤瀵硅薄锛?
- `tensor_cast/performance_model/perf_database/data/.../FusedInferAttentionScore.csv`
- `tools/perf_data_collection/op_replay/FusedInferAttentionScore_run.py`

## 1. 缁撹姒傝

`FusedInferAttentionScore.csv` 閲岀殑杈撳叆涓嶆槸鈥滃彧璁板綍浜嗗疄闄呭嚭鐜扮殑鍙傛暟鈥濓紝鑰屾槸鎸?
`torch_npu.npu_fused_infer_attention_score` 鐨?tensor 鍙傛暟椤哄簭灞曞紑鎴愬浐瀹氭Ы浣嶃€?

- 鍏?31 涓?tensor 杈撳叆妲戒綅
- 绌哄瓧绗︿覆琛ㄧず璇ュ弬鏁板湪璇ヨ鏈紶鍏?
- 闈?tensor 鏍囬噺鍙傛暟涓嶅湪杩?31 涓Ы浣嶄腑锛岄渶瑕佽剼鏈澶栨帹鏂垨鏄惧紡浼犲叆

褰撳墠鍒嗘瀽涓昏渚濇嵁鍥涢儴鍒嗕氦鍙夌‘璁わ細
- `torch_npu.npu_fused_infer_attention_score` 瀹樻柟鏂囨。绛惧悕
- `FusedInferAttentionScore.csv` 鍚勮闈炵┖妲戒綅鍒嗗竷
- shape/dtype/format 鐨勮涔夌壒寰?
- 浠撳唴 attention / paged attention / MLA 鐩稿叧浠ｇ爜

## 2. CSV 31 涓緭鍏ユЫ浣嶄笌鎺ュ彛鍙傛暟鐨勫搴斿叧绯?

| Index | 鍙傛暟鍚?| 褰撳墠 CSV 鏄惁鍑虹幇 | 鍏稿瀷 Shape | 璇存槑 |
|---|---|---:|---|---|
| 0 | `query` | 鏄?| `3072,4,128` / `16,4,128` / `4,16,1,512` | 涓昏緭鍏?Q |
| 1 | `key` | 鏄?| `12235,128,128` / `892,1,128,512` / `41040,1,128` | 涓昏緭鍏?K锛宲aged 鍦烘櫙涓嬫槸 KV cache 褰㈡€?|
| 2 | `value` | 鏄?| `12235,128,128` / `892,1,128,512` / `41040,1,128` | 涓昏緭鍏?V |
| 3 | `pse_shift` | 鍚?| - | 褰撳墠搴撻噷鏈嚭鐜?|
| 4 | `atten_mask` | 鏄?| `2048,2048` | attention mask |
| 5 | `actual_seq_lengths` | 鏄?| `2` / `11` / `16` | 瀹為檯 Q 闀垮害鍒楄〃锛屽 TND 甯告寜绱闀垮害鐞嗚В |
| 6 | `actual_seq_lengths_kv` | 鏄?| `2` / `11` / `4` / `5` | 瀹為檯 KV 闀垮害鍒楄〃 |
| 7 | `dequant_scale1` | 鍚?| - | 鏈嚭鐜?|
| 8 | `quant_scale1` | 鍚?| - | 鏈嚭鐜?|
| 9 | `dequant_scale2` | 鍚?| - | 鏈嚭鐜?|
| 10 | `quant_scale2` | 鍚?| - | 鏈嚭鐜?|
| 11 | `quant_offset2` | 鍚?| - | 鏈嚭鐜?|
| 12 | `antiquant_scale` | 鍚?| - | 鏈嚭鐜?|
| 13 | `antiquant_offset` | 鍚?| - | 鏈嚭鐜?|
| 14 | `block_table` | 鏄?| `2,512` / `129,512` / `4,512` / `5,512` | page attention 鐨?block 鏄犲皠琛?|
| 15 | `query_padding_size` | 鍚?| - | 鏈嚭鐜?|
| 16 | `kv_padding_size` | 鍚?| - | 鏈嚭鐜?|
| 17 | `key_antiquant_scale` | 鍚?| - | 鏈嚭鐜?|
| 18 | `key_antiquant_offset` | 鍚?| - | 鏈嚭鐜?|
| 19 | `value_antiquant_scale` | 鍚?| - | 鏈嚭鐜?|
| 20 | `value_antiquant_offset` | 鍚?| - | 鏈嚭鐜?|
| 21 | `key_shared_prefix` | 鍚?| - | 鏈嚭鐜?|
| 22 | `value_shared_prefix` | 鍚?| - | 鏈嚭鐜?|
| 23 | `actual_shared_prefix_len` | 鍚?| - | 鏈嚭鐜?|
| 24 | `query_rope` | 鏄?| `4,16,1,64` / `5,16,1,64` | MLA/rope 鍦烘櫙涓嬬殑 query rope |
| 25 | `key_rope` | 鏄?| `892,1,128,64` | MLA/rope 鍦烘櫙涓嬬殑 key rope |
| 26 | `key_rope_antiquant_scale` | 鍚?| - | 鏈嚭鐜?|

璇存槑锛?
- 鏂囨。绛惧悕閲?`*` 涔嬪悗鐨勫弬鏁板潎涓?keyword 鍙傛暟锛屼絾 CSV 浠嶆寜鍥哄畾浣嶇疆灞曞紑
- 褰撳墠搴撲腑鍙嚭鐜颁簡鍓?27 涓?tensor 鍙傛暟涓殑閮ㄥ垎妲戒綅
- 鏍囬噺鍙傛暟濡?`num_heads`銆乣scale`銆乣input_layout`銆乣sparse_mode` 涓嶄細鍑虹幇鍦ㄨ繖寮犺〃閲?

## 3. 涓嶅湪 31 涓Ы浣嶄腑鐨勬爣閲忓弬鏁?

杩欎簺鍙傛暟涓嶅湪 CSV 鐨?`Input Shapes / Input Data Types / Input Formats` 31 妲戒綅涓紝闇€瑕佽剼鏈澶栨帹鏂垨鏄惧紡浼犲叆銆?

| 鍙傛暟鍚?| 鏄惁鍦?CSV 鐨?31 涓緭鍏ユЫ浣嶄腑 | 褰撳墠鑴氭湰濡備綍纭畾 |
|---|---:|---|
| `num_heads` | 鍚?| 浠?`query` shape 鎺ㄦ柇 |
| `scale` | 鍚?| 閫氬父鎸?`1 / sqrt(head_dim)` 鎺ㄦ柇 |
| `pre_tokens` | 鍚?| 鑴氭湰鍥哄畾浼犺緝澶у€?|
| `next_tokens` | 鍚?| 鑴氭湰鍥哄畾浼犺緝澶у€?|
| `input_layout` | 鍚?| 鏍规嵁 `query / key / query_rope` shape 妯″紡鎺ㄦ柇 |
| `num_key_value_heads` | 鍚?| 鏍规嵁 `key` shape 鍜屽満鏅帹鏂?|
| `sparse_mode` | 鍚?| 鏍规嵁 `atten_mask` 鏄惁瀛樺湪鍙婂舰鐘舵帹鏂?|
| `inner_precise` | 鍚?| 褰撳墠鑴氭湰鏈壒鍒墦寮€ |
| `block_size` | 鍚?| paged 鍦烘櫙涓嬬敱 `key` shape 鎺ㄦ柇 |
| `antiquant_mode` | 鍚?| 褰撳墠鏈娇鐢?|
| `softmax_lse_flag` | 鍚?| 鎸?`Output Shapes` 鏄惁瀛樺湪绗簩杈撳嚭鍒ゆ柇 |
| `key_antiquant_mode` | 鍚?| 褰撳墠鏈娇鐢?|
| `value_antiquant_mode` | 鍚?| 褰撳墠鏈娇鐢?|

## 4. 鎴戜滑鏄浣曠‘璁よ繖浜涙Ы浣嶆槧灏勭殑

### 4.1 鎸夋帴鍙ｇ鍚嶉『搴忓榻?

鏂囨。缁欏嚭鐨勭鍚嶅涓嬶細

```python
torch_npu.npu_fused_infer_attention_score(
    query,
    key,
    value,
    *,
    pse_shift=None,
    atten_mask=None,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    dequant_scale1=None,
    quant_scale1=None,
    dequant_scale2=None,
    quant_scale2=None,
    quant_offset2=None,
    antiquant_scale=None,
    antiquant_offset=None,
    block_table=None,
    query_padding_size=None,
    kv_padding_size=None,
    key_antiquant_scale=None,
    key_antiquant_offset=None,
    value_antiquant_scale=None,
    value_antiquant_offset=None,
    key_shared_prefix=None,
    value_shared_prefix=None,
    actual_shared_prefix_len=None,
    query_rope=None,
    key_rope=None,
    key_rope_antiquant_scale=None,
    num_heads=1,
    scale=1.0,
    pre_tokens=2147483647,
    next_tokens=2147483647,
    input_layout="BSH",
    num_key_value_heads=0,
    sparse_mode=0,
    inner_precise=0,
    block_size=0,
    antiquant_mode=0,
    softmax_lse_flag=False,
    key_antiquant_mode=0,
    value_antiquant_mode=0,
)
```

CSV 閲岀殑 31 涓?tensor 妲戒綅灏辨槸鎸夎繖閲岀殑 tensor 鍙傛暟椤哄簭灞曞紑鐨勩€?

### 4.2 鐢ㄩ潪绌?index 鍙嶆帹鍏蜂綋鍙傛暟

褰撳墠搴撲腑锛屽父瑙佽鐨勯潪绌?index 寰堢ǔ瀹氾紝渚嬪锛?

- `0,1,2` 鎭掗潪绌猴紝瀵瑰簲 `query/key/value`
- `4` 甯镐负 `2048,2048` 涓?dtype 鏄?`INT8`锛屾槑鏄剧鍚?`atten_mask`
- `5,6` 鏄崟鏁板瓧 shape 涓?dtype 鏄?`INT64`锛岀鍚?`actual_seq_lengths / actual_seq_lengths_kv`
- `14` 鏄簩缁?`INT32`锛屽 `129,512`锛岀鍚?`block_table`
- `24,25` 浠呭湪 MLA 鏍锋湰涓嚭鐜帮紝涓?dtype 鏄?`BF16`锛岀鍚?`query_rope / key_rope`

### 4.3 鐢?shape 璇箟鍋氫簩娆℃牎楠?

- 鏅€氳锛歚query` 甯镐负 `(T, N, D)`锛宍key/value` 甯镐负 `(block_num, block_size, D)` 鎴栬繎浼?page cache 甯冨眬锛岃鏄庢槸 TND + page attention 璺緞
- MLA 琛岋細`query` 涓?`(B, N, S, D)`锛屽悓鏃舵湁 `query_rope/key_rope`锛岃鏄庢槸 MLA rope 褰㈡€?
- `block_table` 绗簩缁村浐瀹氬儚 `max_blocks_per_seq`锛屽拰 page attention 鏂囨。涓€鑷?

### 4.4 鐢ㄤ粨鍐呬唬鐮佷氦鍙夐獙璇?

涓昏鍙傝€冿細
- `tensor_cast/ops/attention.py`
- `tensor_cast/ops/mla.py`
- `tensor_cast/core/input_generator.py`
- `tensor_cast/performance_model/__init__.py`

杩欎簺鏂囦欢甯姪纭浜嗭細
- `block_table` 鐨勮涔?
- `query_lens / seq_lens` 涓?TND/paged attention 鐨勫叧绯?
- rope 涓?MLA 鐩稿叧 shape 鐨勮涔?

## 5. 閽堝杩欎唤 CSV 鐨勫疄闄呬笁绉?case 鍒嗙被

褰撳墠杩欎唤 `FusedInferAttentionScore.csv` 瀹為檯鍙垎涓轰笁绫汇€?

### Case A: 鏅€?paged TND

杩欐槸褰撳墠搴撻噷鏁伴噺鏈€澶氱殑涓€绫汇€?

鍏稿瀷鐗瑰緛锛?
- `query`: 3D锛屽舰濡?`(T, N, D)`锛屼緥濡?`3072,4,128`
- `key/value`: 3D锛屽舰濡?`(block_num, block_size, D)`锛屼緥濡?`12235,128,128`
- `atten_mask`: 瀛樺湪锛岄€氬父鏄?`2048,2048`
- `actual_seq_lengths`: 瀛樺湪
- `actual_seq_lengths_kv`: 瀛樺湪
- `block_table`: 瀛樺湪
- `query_rope/key_rope`: 涓嶅瓨鍦?

瀵瑰簲妲戒綅锛?
- 蹇呭～锛歚0,1,2`
- 甯歌闈炵┖锛歚4,5,6,14`

鑴氭湰渚ф帹鏂細
- `input_layout = "TND"`
- `num_heads = query.shape[1]`
- `num_key_value_heads = 1`
- `block_size = key.shape[1]`
- `sparse_mode` 渚濇嵁 `atten_mask` 褰㈡€佹帹鏂?

澶囨敞锛?
- 杩欑被琛岀殑 `key/value` 棣栫淮鏇村儚鏁翠釜 KV cache 姹犲閲忥紝鑰屼笉鏄綋鍓?batch 鐨勭湡瀹炰笂涓嬫枃 block 鏁?
- 浠呭嚟 CSV 鐨?shape 鏃犳硶鎭㈠鐪熷疄 `block_table` 鍐呭鍜岀湡瀹?`actual_seq_lengths_kv` 鏁板€?

### Case B: 闈?paged TND

杩欑被琛屾暟閲忚緝灏戯紝浣嗕粛鏄櫘閫?attention锛屼笉甯?rope銆?

鍏稿瀷鐗瑰緛锛?
- `query`: 3D锛屽舰濡?`(T, N, D)`锛屼緥濡?`41040,4,128`
- `key/value`: 3D锛屽舰濡?`(T_kv, KV_N, D)`锛屼緥濡?`41040,1,128`
- `atten_mask`: 瀛樺湪
- `actual_seq_lengths`: 瀛樺湪
- `actual_seq_lengths_kv`: 瀛樺湪
- `block_table`: 涓嶅瓨鍦?
- `query_rope/key_rope`: 涓嶅瓨鍦?

瀵瑰簲妲戒綅锛?
- 蹇呭～锛歚0,1,2`
- 甯歌闈炵┖锛歚4,5,6`

鑴氭湰渚ф帹鏂細
- `input_layout = "TND"`
- `num_heads = query.shape[1]`
- `num_key_value_heads = key.shape[1]`
- `block_size = 0`

涓?Case A 鐨勫叧閿尯鍒細
- 娌℃湁 `block_table`
- `key/value` 鐨?shape 鐩存帴琛ㄨ揪鐪熷疄 KV 闀垮害
- 涓嶆槸 page attention

### Case C: MLA rope

杩欑被琛屽彧鍑虹幇灏戦噺鏍锋湰锛屼絾缁撴瀯鏈€鐗规畩銆?

鍏稿瀷鐗瑰緛锛?
- `query`: 4D锛屽舰濡?`(B, N, S, D)`锛屼緥濡?`4,16,1,512`
- `key/value`: 4D锛屽舰濡?`(block_num, KV_N, block_size, D)`锛屼緥濡?`892,1,128,512`
- `actual_seq_lengths`: 閫氬父涓虹┖
- `actual_seq_lengths_kv`: 瀛樺湪
- `block_table`: 瀛樺湪
- `query_rope`: 瀛樺湪锛屼緥濡?`4,16,1,64`
- `key_rope`: 瀛樺湪锛屼緥濡?`892,1,128,64`

瀵瑰簲妲戒綅锛?
- 蹇呭～锛歚0,1,2`
- 甯歌闈炵┖锛歚6,14,24,25`

鑴氭湰渚ф帹鏂細
- `input_layout = "BNSD_NBSD"`
- `num_heads = query.shape[1]`
- `num_key_value_heads = 1`
- `block_size = key.shape[2]`
- `scale` 闇€缁撳悎 `query` 涓?`query_rope` 鐨勬渶鍚庝竴缁寸悊瑙?

涓庡墠涓ょ被鐨勫叧閿尯鍒細
- `query` 鏄?4D 鑰屼笉鏄?3D
- 甯?`query_rope / key_rope`
- 鏄庢樉鏄?MLA 鐩稿叧璺緞锛岃€屼笉鏄櫘閫?TND attention

## 6. 涓轰粈涔堚€滃悓鏍?shape鈥濅笉绛変簬鈥滃悓鏍疯緭鍏モ€?

浠?replay 瑙掑害锛孋SV 鍙畬鏁磋褰曚簡锛?
- shape
- dtype
- format

浣嗘病鏈夎褰曠湡瀹炶繍琛屾椂鐨勶細
- `query/key/value` 瀹為檯鏁板€?
- `actual_seq_lengths` 瀹為檯鍒楄〃鍊?
- `actual_seq_lengths_kv` 瀹為檯鍒楄〃鍊?
- `block_table` 瀹為檯鏄犲皠鍐呭
- `atten_mask` 瀹為檯鍐呭
- 閮ㄥ垎鏍囬噺鍙傛暟鐨勭湡瀹炲彇鍊?

鍥犳锛屽嵆浣?shape/dtype/format 涓€鏍凤紝涔熶笉浠ｈ〃鏄€滃悓鏍疯緭鍏モ€濄€?

杩欎篃鏄负浠€涔?replay 鑴氭湰鍙兘鍋氬埌锛?
- 涓ユ牸澶嶅師杈撳叆缁撴瀯
- 杩戜技澶嶅師閮ㄥ垎鏍囬噺鍙傛暟
- 鍚堟硶鏋勯€犵己澶辩殑鍔ㄦ€佸唴瀹?

浣嗕笉鑳戒繚璇佷笌鍘?profiling 瀹屽叏涓€鑷淬€?

## 7. 褰撳墠鑴氭湰涓笌 CSV 瀵归綈鏃剁殑鐗瑰埆娉ㄦ剰鐐?

### 7.1 `softmax_lse_flag`

涓嶈兘鏍规嵁 `Output Data Types` 鏄惁鏈夌浜岄」鏉ュ垽鏂€?

鍘熷洜锛?
- 鏌愪簺 CSV 琛岄噷 `Output Data Types` 浼氬啓鎴?`DT_BF16;FLOAT`
- 浣?`Output Shapes` 瀹為檯鍙湁涓€涓緭鍑?shape

鍥犳鏇村悎鐞嗙殑鍒ゆ柇鏂瑰紡鏄細
- 鍙湁褰?`Output Shapes` 涓‘瀹炲瓨鍦ㄧ浜屼釜 shape 鏃讹紝鎵嶈涓?`softmax_lse_flag=True`

### 7.2 paged attention 鐨?`actual_seq_lengths_kv`

瀵?paged 鍦烘櫙锛?
- `key/value` 鐨勯缁存洿鍍?cache 姹犲ぇ灏?
- 涓嶈兘鐩存帴鎷挎潵褰撶湡瀹炰笂涓嬫枃 block 鏁?

鍚﹀垯寰堝鏄撴瀯閫犲嚭杩滃ぇ浜?`block_table` 瀹藉害鐨勯潪娉?`actual_seq_lengths_kv`銆?

### 7.3 `atten_mask` 涓?`sparse_mode`

鏍规嵁瀹樻柟绾︽潫锛?
- 浼犲叆 `atten_mask` 鏃讹紝`sparse_mode` 闇€瑕佷笌 mask 褰㈡€佸尮閰?
- 瀵?`2048x2048` 浼樺寲 mask锛岄€氬父闇€瑕佽蛋鐗瑰畾 sparse 璺緞

鍥犳 `atten_mask` 鏄惁瀛樺湪銆乻hape 鏄粈涔堬紝浼氱洿鎺ュ奖鍝?`sparse_mode` 鐨勬帹鏂€?

## 8. 鎺ㄨ崘浣跨敤鏂瑰紡

濡傛灉闇€瑕佺户缁垎鏋?`FusedInferAttentionScore.csv`锛屽缓璁寜浠ヤ笅椤哄簭鐪嬶細

1. 鍏堢湅鏌愪竴琛岄潪绌烘Ы浣嶅垎甯?
2. 鍐嶅垽鏂畠灞炰簬 Case A / B / C 鍝竴绫?
3. 鍐嶇粨鍚堣绫诲搴旂殑 `input_layout / block_table / rope` 璇箟鐞嗚В鍙傛暟
4. 鏈€鍚庡啀鐪?replay 鑴氭湰閲屽浣曡ˉ榻愭爣閲忓弬鏁板拰鍔ㄦ€佽緭鍏?

杩欐牱浼氭瘮鐩存帴浠?31 涓Ы浣嶇‖璇绘洿娓呮櫚銆?

