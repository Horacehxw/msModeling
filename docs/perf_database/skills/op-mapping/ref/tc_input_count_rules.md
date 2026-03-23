# tc_input_count 浣跨敤瑙勫垯

## 姒傝堪

`tc_input_count` 鎴柇 TC 鍜?CSV 鐨勮緭鍏ユ暟閲忓埌 N 杩涜鍖归厤銆傝繖鏄竴涓矖绮掑害宸ュ叿,浣跨敤涓嶅綋浼氬鑷磋鍖归厤銆?

## 瀹夊叏鍦烘櫙: 鎴柇 NPU 鍐呴儴鍙傛暟

褰?CSV 棰濆杈撳叆鏄?CANN 鍐呴儴鍙傛暟(axis, scale, offset 绛?鏃?`tc_input_count` 鏄畨鍏ㄧ殑:

| Op | tc_input_count | TC 杈撳叆 | CSV 杈撳叆 | 鎴柇鍘熷洜 |
|---|---|---|---|---|
| `aten.scatter.value` | 2 | (self, index) | (self, index, **axis**) | axis 鏄?CANN 鍐呴儴鍙傛暟 |
| `aten.gather.default` | 2 | (self, index) | (self, index, **axis**) | 鍚屼笂 |
| `aten.embedding.default` | 2 | (weight, indices) | (weight, indices, **axis**) | 鍚屼笂 |
| `tensor_cast.quantize.default` | 1 | (tensor, scale, zp) | (tensor) | scale/zp 鏄?kernel 鍙傛暟,涓嶅湪 CSV |
| `tensor_cast.static_quant_linear.default` | 2 | (x, weight, scale, ...) | (x, weight_NZ, scale, scale) | 鍙尮閰?x+weight |
| `tensor_cast.dispatch_ffn_combine.default` | 1 | (x, expert_indices) | (x, w1, w2, idx, s1, s2, probs) | 鏉冮噸鏄ā鍨嬪浐瀹氱殑 |

**鍒ゆ柇鏍囧噯**: CSV 澶氬嚭鐨勮緭鍏ユ槸 NPU kernel 鐨?*鍥哄畾鍙傛暟**(涓嶉殢 batch/seq 鍙樺寲)鈫?瀹夊叏鎴柇銆?

## 涓嶅畨鍏ㄥ満鏅? 鍙橀噺骞挎挱妯″紡鐨勯€愬厓绱犵畻瀛?

褰?CSV 琛岀殑杈撳叆鏁伴噺鍥犲箍鎾ā寮忎笉鍚岃€屽彉鍖栨椂,`tc_input_count` 涓嶅畨鍏?

```
# Add.csv 涓殑娣峰悎妯″紡:
(16, 7168;)              BF16   5.2 us   鈫?鏍囬噺骞挎挱 (1 tensor read)
(16, 7168; 7168)         BF16   6.8 us   鈫?鍚戦噺骞挎挱 (2 tensor reads)
(16, 7168; 16, 7168)     BF16   7.1 us   鈫?閫愬厓绱?(2 tensor reads, same shape)
```

璁剧疆 `tc_input_count=1` 鍚?
- TC 鐨?`add(x=(16,7168), y=(16,7168))` 浼氬尮閰嶇涓€琛?(5.2 us)
- 姝ｇ‘搴斿尮閰嶇涓夎 (7.1 us)
- **~30% 寤惰繜浣庝及**

**褰卞搷鐨勭畻瀛?*: `aten.add.Tensor`, `aten.mul.Tensor`, `aten.div.Tensor`, `aten.sub.Tensor`

## 鍐崇瓥娴佺▼

```
Q: CSV 澶氬嚭鐨勮緭鍏ユ槸浠€涔?
鈹?
鈹溾攢 NPU 鍐呴儴鍥哄畾鍙傛暟 (axis, scale, offset) 鈫?tc_input_count = N (瀹夊叏)
鈹溾攢 骞挎挱鎿嶄綔鏁?(鏈夋椂鏈?鏈夋椂娌℃湁) 鈫?涓嶈 tc_input_count (璁╁畠 MISS)
鈹斺攢 涓嶇‘瀹?鈫?涓嶈,淇濆畧璁?MISS,浜ょ粰 analytic fallback
```

## 宸茶В鍐? 閫愬厓绱犵畻瀛愪娇鐢?`query_mode: elementwise`

瀵逛簬閫愬厓绱犵畻瀛?(Add, Mul, Div), 涓嶉渶瑕?`tc_input_count`銆傝繖浜涚畻瀛愪娇鐢?`query_mode: elementwise`,
鎸?*杈撳嚭褰㈢姸**鍖归厤,瀹屽叏缁曡繃杈撳叆褰㈢姸姣旇緝銆?

**瑙勫垯**: `query_mode: elementwise` 涓?`tc_input_count` **浜掓枼**銆傝缃簡 `query_mode: elementwise`
鐨勬潯鐩笉寰楄缃?`tc_input_count`銆?

## 闀挎湡瑙ｅ喅鏂规

瀵归€愬厓绱犵畻瀛?鎺ㄨ崘浣跨敤 **杈撳嚭褰㈢姸鍖归厤** (`query_mode: elementwise`):
- 杈撳嚭褰㈢姸 = 骞挎挱鍚庣殑褰㈢姸,鏃犺杈撳叆鏄爣閲?鍚戦噺/鐩稿悓褰㈢姸,杈撳嚭纭畾
- 鑷劧鏀寔鎻掑€?娌胯緭鍑虹淮搴︽彃鍊兼湁鐗╃悊鎰忎箟)
- 闇€瑕佸湪 `profiling_data_source.py` 涓柊澧?`_lookup_elementwise()` 鏂规硶

## 鍙傝€?

- `profiling_data_source.py` `_inputs_match()` 绗?942-951 琛? tc_input_count 鎴柇閫昏緫
- `profiling_data_source.py` `_lookup_compute()` 绗?830-834 琛? TC 杈撳叆鎴柇
- 璁捐鏂囨。 S4.2: 鏌ヨ璋冨害閫昏緫

