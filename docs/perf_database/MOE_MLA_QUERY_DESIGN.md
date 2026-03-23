# MoE / MLA 鏌ヨ鎺ュ彛璁捐鏂囨。

**鐗堟湰**锛歷1.0锛堣崏绋匡級
**浣滆€?*锛歓H
**鏃ユ湡**锛?026-03-12
**鍏宠仈鏂囨。**锛歚OPERATOR_PERF_DATABASE_DESIGN_zh_v1.2.md` 搂4.2銆乣COMM_QUERY_DESIGN.md`
**鍏宠仈浠诲姟**锛欸1锛圡oE 绠楀瓙鍖归厤锛屾埅姝?3.18锛夈€丟2锛圡LA 鍒嗚В瀹屾暣瀹炵幇锛屾埅姝?3.20锛?

---

## 1. 姒傝堪

鏈枃妗ｆ弿杩?`ProfilingDataSource` 瀵?MoE 璺敱绠楀瓙鍜?MLA 娉ㄦ剰鍔涚畻瀛愮殑鏌ヨ閫昏緫璁捐銆?

### 1.1 绠楀瓙鍒嗙被

| 绫诲埆 | TC 绠楀瓙 | NPU Kernel | 鏌ヨ璺緞 | 浠诲姟 |
|------|---------|-----------|---------|------|
| MoE 璺敱 | `permute_tokens` | `MoeDistributeDispatchV2` | compute锛堟爣鍑?shape 鍖归厤锛?| G1 |
| MoE 璺敱 | `unpermute_tokens` | `MoeDistributeCombineV2` | compute锛堟爣鍑?shape 鍖归厤锛?| G1 |
| MoE 璺敱 | `moe_gating_topk` | `MoeGatingTopK` | compute锛堟爣鍑?shape 鍖归厤锛?| G1 |
| MLA attention | `multihead_latent_attention` | TransposeBatchMatMul + FusedInferAttentionScore | composite锛圔2 鍗犱綅锛孏2 瀹屾暣瀹炵幇锛?| G2 |
| MLA attention | `multihead_latent_attention_quant` | TransposeBatchMatMul + FusedInferAttentionScore | composite锛圔2 鍗犱綅锛孏2 瀹屾暣瀹炵幇锛?| G2 |
| MLA prolog | `mlapo` | MatMulV2 + KvRmsNormRopeCache | composite锛圙2锛?| G2 |
| MLA prolog | `mlapo_quant` | QuantBatchMatmulV3 + KvRmsNormRopeCache | composite锛圙2锛?| G2 |

---

## 2. MoE 绠楀瓙鏌ヨ璁捐锛圙1锛?

### 2.1 TC 绠楀瓙绛惧悕

**鏂囦欢**锛歚tensor_cast/ops/fused_moe.py`

```python
@register_tensor_cast_op("permute_tokens")
def _(
    x: torch.Tensor,           # args[0]: (num_tokens, hidden_size)
    topk_indices: torch.Tensor, # args[1]: (num_tokens, top_k)
) -> torch.Tensor:
    # 杩斿洖: (num_tokens * top_k, hidden_size)

@register_tensor_cast_op("unpermute_tokens")
def _(
    x: torch.Tensor,           # args[0]: (num_tokens * top_k, hidden_size)
    topk_indices: torch.Tensor, # args[1]: (num_tokens, top_k)
) -> torch.Tensor:
    # 杩斿洖: (num_tokens, top_k, hidden_size)
```

> `moe_gating_topk` 褰撳墠鏈湪 develop 鍒嗘敮娉ㄥ唽涓?TC 绠楀瓙锛圱C 鐢?`aten.topk` 瀹炵幇璺敱锛夈€?
> op_mapping.yaml 涓湁 `profiling.MoeGatingTopK` 鍗犱綅鏉＄洰锛孏1 鏆備笉瀹炵幇鍏舵煡璇㈤€昏緫銆?

### 2.2 CSV 鏍煎紡

鐜版湁 CSV锛坄vllm0.13.0_torch2.8.0_cann8.3/` 鐩綍锛変负**鍘熷 Profiling 鏍煎紡**锛屽寘鍚畬鏁寸殑 NPU 鎬ц兘璁℃暟鍣ㄥ垪锛?

```
OP State, Accelerator Core, Input Shapes, Input Data Types, Input Formats,
Output Shapes, Output Data Types, Output Formats, Average Duration(us), ...
```

杩欎笌鏍囧噯 compute 鏌ヨ璺緞锛坄_lookup_compute` 鈫?`_inputs_match`锛?*瀹屽叏鍏煎**锛屾棤闇€鐗规畩澶勭悊銆?

#### MoeDistributeDispatchV2 CSV 绀轰緥

```
Input Shapes: "3,7168;3,8;;3;3,8;"
Input Data Types: DT_BF16;INT32;DT_UNDEFINED;BOOL;FLOAT;DT_UNDEFINED
Average Duration(us): 1537.95
```

- `args[0]` = `x`锛歴hape `(num_tokens, hidden_size)` = `(3, 7168)`
- `args[1]` = `topk_indices`锛歴hape `(num_tokens, top_k)` = `(3, 8)`
- 鍏朵綑杈撳叆锛坄;;3;3,8;`锛変负 NPU 鍐呴儴鍙傛暟锛孴C 涓嶄紶閫?

#### MoeDistributeCombineV2 CSV 绀轰緥

```
Input Shapes: "384,7168;3,8;49152;1792;3,8;1;3;..."
Average Duration(us): 164.38
```

- `args[0]` = `x`锛歴hape `(num_tokens * top_k, hidden_size)` = `(384, 7168)`
- `args[1]` = `topk_indices`锛歴hape `(num_tokens, top_k)` = `(3, 8)`

#### MoeGatingTopK CSV 绀轰緥

```
Input Shapes: "3,256;256"
Input Data Types: DT_BF16;DT_BF16
Average Duration(us): 6.94
```

- `args[0]` = logits锛歴hape `(num_tokens, num_experts)` = `(3, 256)`
- `args[1]` = expert_bias锛歴hape `(num_experts,)` = `(256,)`

### 2.3 Shape 鍖归厤鎸戞垬

MoE CSV 鐨?Input Shapes 鍖呭惈 TC 涓嶄紶閫掔殑 NPU 鍐呴儴鍙傛暟锛堢┖瀛楁 `;;`锛夛紝瀵艰嚧 `_inputs_match` 涓?`len(tc_inputs) != len(csv_shapes)` 鐩存帴杩斿洖 False銆?

**瑙ｅ喅鏂规**锛氬湪 `_inputs_match` 涓鍔?MoE kernel 鐨勭壒娈婂鐞嗭紝鎴栧湪 `_lookup_compute` 涓 MoE kernel 鍋?shape 鍓嶇紑鍖归厤锛堝彧姣旇緝 TC 浼犻€掔殑鍓?N 涓緭鍏ワ級銆?

鍏蜂綋绛栫暐锛?

| 鏂规 | 鎻忚堪 | 浼樼己鐐?|
|------|------|--------|
| A锛氬墠缂€鍖归厤 | 鍙瘮杈?TC 杈撳叆鏁伴噺瀵瑰簲鐨勫墠 N 涓?CSV shape | 绠€鍗曪紝浣嗗彲鑳借鍖归厤 |
| B锛歰p_mapping 閰嶇疆 `tc_input_count` | 鍦?op_mapping.yaml 涓０鏄?TC 渚ц緭鍏ユ暟閲忥紝鍖归厤鏃舵埅鏂?CSV shapes | 鏄惧紡锛屽彲鎵╁睍 |
| C锛歮icrobenchmark CSV | 閲嶆柊閲囬泦鍙惈 TC 杈撳叆鐨?microbenchmark 鏍煎紡 CSV | 鏈€骞插噣锛屼絾闇€瑕佹暟鎹噰闆?|

**鎺ㄨ崘鏂规 B**锛氬湪 op_mapping.yaml 涓负 MoE kernel 娣诲姞 `tc_input_count` 瀛楁锛宍_inputs_match` 璇诲彇璇ュ瓧娈靛悗鎴柇 CSV shapes 鍐嶆瘮杈冦€?

### 2.4 op_mapping.yaml 閰嶇疆锛堝缓璁級

```yaml
"tensor_cast.permute_tokens.default":
  kernel_type: MoeDistributeDispatchV2
  tc_input_count: 2   # TC 鍙紶 x + topk_indices锛孋SV 鏈夋洿澶?NPU 鍐呴儴鍙傛暟

"tensor_cast.unpermute_tokens.default":
  kernel_type: MoeDistributeCombineV2
  tc_input_count: 2   # TC 鍙紶 x + topk_indices
```

### 2.5 鏌ヨ娴佺▼

```
_lookup_compute(op_invoke_info, mapping)
  鈹?
  鈹溾攢 璇诲彇 mapping.get("tc_input_count")
  鈹?   鑻ュ瓨鍦?鈫?tc_inputs = tc_inputs[:tc_input_count]
  鈹?
  鈹溾攢 _load_csv("MoeDistributeDispatchV2")
  鈹?
  鈹斺攢 _inputs_match(tc_inputs[:N], row, kernel_type)
       鈹溾攢 csv_shapes = _parse_shape_str(row["Input Shapes"])[:N]  鈫?鎴柇
       鈹斺攢 閫愮淮姣旇緝锛堝惈 block-padding 瀹瑰繊锛?
```

### 2.6 miss_reason

| miss_reason | 鍚箟 |
|-------------|------|
| `csv_not_found` | MoE CSV 鏂囦欢涓嶅瓨鍦?|
| `shape_mismatch` | num_tokens / hidden_size / top_k 涓嶅尮閰?|
| `input_count_mismatch` | tc_input_count 閰嶇疆閿欒 |

---

## 3. MLA 绠楀瓙鏌ヨ璁捐锛圙2锛?

### 3.1 TC 绠楀瓙绛惧悕

**鏂囦欢**锛歚tensor_cast/ops/mla.py`

#### multihead_latent_attention

```python
@register_tensor_cast_op("multihead_latent_attention")
def _(
    q: torch.Tensor,                    # args[0]: (num_tokens, num_heads, qk_head_dim)
    kv_cache: torch.Tensor,             # args[1]: (total_blocks, block_size, kv_lora_rank + qk_rope_head_dim)
    block_table: torch.Tensor,          # args[2]: (batch_size, max_blocks_per_seq)
    query_start_loc: torch.Tensor,      # args[3]: (batch_size + 1,)
    seq_lens: torch.Tensor,             # args[4]: (batch_size,) 鈥?KV 搴忓垪闀垮害
    query_lens: Optional[torch.Tensor], # args[5]: (batch_size,) 鈥?query 闀垮害锛孨one 鏃朵负 decode
    W_UK_T: Optional[torch.Tensor],     # args[6]: (num_heads, qk_nope_head_dim, kv_lora_rank)锛宒ecode 涓撶敤
    W_UV: Optional[torch.Tensor],       # args[7]: (num_heads, kv_lora_rank, v_head_dim)锛宒ecode 涓撶敤
    kv_b_proj: Optional[torch.Tensor],  # args[8]: (kv_lora_rank, num_heads*(qk_nope_head_dim+v_head_dim))锛宲refill 涓撶敤
    v_head_dim: int,                    # args[9]: scalar
) -> torch.Tensor:
    # 杩斿洖: (num_tokens, num_heads, v_head_dim)
```

#### mlapo锛圡LA Prolog锛?

```python
@register_tensor_cast_op("mlapo")
def _(
    hidden_states: torch.Tensor,        # args[0]: (num_tokens, hidden_size)
    cos: torch.Tensor,                  # args[1]: (1, seq_len, qk_rope_head_dim)
    sin: torch.Tensor,                  # args[2]: (1, seq_len, qk_rope_head_dim)
    q_a_proj_weight: Optional[...],     # args[3]: (hidden_size, q_lora_rank)
    q_a_layernorm_weight: Optional[...],# args[4]: (q_lora_rank,)
    q_b_proj_weight: Optional[...],     # args[5]: (q_lora_rank, num_heads * qk_head_dim)
    kv_a_proj_weight: Optional[...],    # args[6]: (hidden_size, kv_lora_rank + qk_rope_head_dim)
    kv_a_layernorm_weight: torch.Tensor,# args[7]: (kv_lora_rank,)
    num_heads: int,                     # args[8]
    qk_head_dim: int,                   # args[9]
    qk_nope_head_dim: int,              # args[10]
    qk_rope_head_dim: int,              # args[11]
    kv_lora_rank: int,                  # args[12]
    q_lora_rank: int,                   # args[13]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 杩斿洖: (q_states, kv_c_normed, k_rot)
```

### 3.2 Prefill vs Decode 璺緞鍒ゆ柇

`multihead_latent_attention` 閫氳繃 `query_lens`锛坅rgs[5]锛夊尯鍒嗚矾寰勶細

| 鏉′欢 | 璺緞 | NPU Kernel 鍒嗚В |
|------|------|----------------|
| `query_lens is None` 鎴栧叏涓?1 | **Decode** | TransposeBatchMatMul 脳 2 + FusedInferAttentionScore |
| `query_lens` 瀛樺湪涓?> 1 | **Prefill** | MatMulV2 + FusedInferAttentionScore |

### 3.3 Decode 璺緞鍒嗚В

Decode 璁＄畻锛歚softmax(q @ W_UK_T @ k_cache) @ v_cache @ W_UV`

NPU 涓婂垎瑙ｄ负锛?
1. **TransposeBatchMatMul**锛坬 @ W_UK_T锛夛細`(batch, num_heads, qk_nope_head_dim) @ (num_heads, qk_nope_head_dim, kv_lora_rank)`
2. **FusedInferAttentionScore**锛氭爣鍑?attention锛宬v_cache 涓瓨鍌ㄥ帇缂?KV
3. **TransposeBatchMatMul**锛坅ttn_out @ W_UV锛夛細`(batch, num_heads, kv_lora_rank) @ (num_heads, kv_lora_rank, v_head_dim)`

鏌ヨ缁村害锛圱ransposeBatchMatMul锛夛細
- Input Shapes锛歚(batch, num_heads, qk_nope_head_dim); (num_heads, qk_nope_head_dim, kv_lora_rank)`
- 浠?`args[0]`锛坬锛夊拰 `args[6]`锛圵_UK_T锛夋彁鍙?

鏌ヨ缁村害锛團usedInferAttentionScore锛夛細
- 璧?`attention_special` 璺緞锛坆atch_size, avg_seq_len, num_heads, head_dim锛?
- 浠?`args[4]`锛坰eq_lens锛夊拰 `args[0]`锛坬锛夋彁鍙?

### 3.4 Prefill 璺緞鍒嗚В

Prefill 璁＄畻锛歚kv_c_normed @ kv_b_proj 鈫?k_nope, v 鈫?softmax(q @ [k_nope, k_rot]) @ v`

NPU 涓婂垎瑙ｄ负锛?
1. **MatMulV2**锛坘v_c_normed @ kv_b_proj锛夛細`(num_tokens, kv_lora_rank) @ (kv_lora_rank, num_heads*(qk_nope_head_dim+v_head_dim))`
2. **FusedInferAttentionScore**锛氭爣鍑?attention

鏌ヨ缁村害锛圡atMulV2锛夛細
- 浠?kv_cache shape 鎺ㄥ kv_lora_rank锛屼粠 `args[8]`锛坘v_b_proj锛夋彁鍙栨潈閲?shape
- 娉ㄦ剰锛歱refill 鏃?kv_c_normed 涓嶅湪 args 涓紝闇€浠?kv_cache 鎺ㄥ

> **闃诲椤?*锛歱refill 璺緞鐨?MatMulV2 shape 鎺ㄥ渚濊禆 kv_c_normed锛岃€?TC 鐨?`multihead_latent_attention` 鎺ユ敹鐨勬槸宸插啓鍏?kv_cache 鐨勬暟鎹紝鏃犳硶鐩存帴鑾峰彇 kv_c_normed shape銆傞渶瑕佺‘璁?kv_cache shape 涓?kv_lora_rank 鐨勫搴斿叧绯汇€?

### 3.5 mlapo 鍒嗚В

`mlapo` 鍒嗚В涓猴細
1. **MatMulV2**锛歨idden_states @ q_a_proj_weight锛圦 projection锛?
2. **KvRmsNormRopeCache**锛欿V projection + norm + RoPE + cache write

鏌ヨ缁村害锛圡atMulV2锛夛細
- `args[0]`锛坔idden_states锛夛細`(num_tokens, hidden_size)`
- `args[3]`锛坬_a_proj_weight锛夛細`(hidden_size, q_lora_rank)`

鏌ヨ缁村害锛圞vRmsNormRopeCache锛夛細
- `args[0]`锛坔idden_states锛夛細`(num_tokens, hidden_size)`
- `args[6]`锛坘v_a_proj_weight锛夛細`(hidden_size, kv_lora_rank + qk_rope_head_dim)`

### 3.6 G2 瀹炵幇鍓嶇疆鏉′欢

| 鏉′欢 | 鐘舵€?| 璐熻矗浜?|
|------|------|--------|
| `FusedInferAttentionScore` microbenchmark CSV | 寰呬氦浠?| TCX |
| `TransposeBatchMatMul.csv`锛坢icrobenchmark 鏍煎紡锛?| 寰呬氦浠?| HDY |
| `KvRmsNormRopeCache.csv` | 宸叉湁锛堝師濮?profiling 鏍煎紡锛?| 鈥?|
| `MatMulV2.csv` | 宸叉湁 | 鈥?|

> 褰撳墠 `_lookup_composite` 瀵?MLA 杩斿洖 `None`锛坄miss_reason = "csv_not_found"`锛夛紝fallback 鍒?AnalyticPerformanceModel銆侴2 鍒颁綅鍚庡畬鏁村疄鐜般€?

---

## 4. 鏁版嵁閲囬泦闇€姹傦紙microbenchmark 鏍煎紡锛?

### 4.1 MoE CSV 鏍煎紡寤鸿

褰撳墠 MoE CSV 涓哄師濮?profiling 鏍煎紡锛堝惈澶ч噺 NPU 鍐呴儴鍙傛暟鍒楋級銆侴1 閫氳繃 `tc_input_count` 鎴柇鍖归厤锛屾棤闇€閲嶆柊閲囬泦銆?

鑻ュ悗缁渶瑕?microbenchmark 鏍煎紡锛屽缓璁垪瀹氫箟锛?

**MoeDistributeDispatchV2.csv**锛?

| 鍒楀悕 | 绫诲瀷 | 璇存槑 |
|------|------|------|
| `num_tokens` | int | token 鏁伴噺锛坆atch 脳 seq锛?|
| `hidden_size` | int | 闅愯棌灞傜淮搴?|
| `top_k` | int | 姣?token 閫夋嫨鐨?expert 鏁?|
| `num_experts` | int | 鎬?expert 鏁?|
| `dtype` | str | 鏁版嵁绫诲瀷 |
| `Duration(us)` | float | 骞冲潎鑰楁椂 |

**MoeGatingTopK.csv**锛?

| 鍒楀悕 | 绫诲瀷 | 璇存槑 |
|------|------|------|
| `num_tokens` | int | token 鏁伴噺 |
| `num_experts` | int | expert 鏁伴噺 |
| `top_k` | int | 閫夋嫨鐨?expert 鏁?|
| `dtype` | str | 鏁版嵁绫诲瀷 |
| `Duration(us)` | float | 骞冲潎鑰楁椂 |

### 4.2 MLA CSV 鏍煎紡寤鸿

**TransposeBatchMatMul.csv**锛坢icrobenchmark 鏍煎紡锛孏2 闇€瑕侊級锛?

| 鍒楀悕 | 绫诲瀷 | 璇存槑 |
|------|------|------|
| `batch_size` | int | batch 澶у皬 |
| `num_heads` | int | 娉ㄦ剰鍔涘ご鏁?|
| `m` | int | 鐭╅樀 M 缁村害 |
| `k` | int | 鐭╅樀 K 缁村害锛堟敹缂╃淮锛?|
| `n` | int | 鐭╅樀 N 缁村害 |
| `dtype` | str | 鏁版嵁绫诲瀷 |
| `Duration(us)` | float | 骞冲潎鑰楁椂 |

---

## 5. 寰呭姙涓庨樆濉為」

| 椤圭洰 | 鐘舵€?| 鎴 | 璐熻矗浜?|
|------|------|------|--------|
| G1锛歚tc_input_count` 鏈哄埗瀹炵幇 | 寰呭紑濮?| 3.18 | ZH |
| G1锛歁oE 绠楀瓙鏌ヨ閫昏緫 + 鍗曞厓娴嬭瘯 | 寰呭紑濮?| 3.18 | ZH |
| G1锛歚moe_gating_topk` TC 绠楀瓙娉ㄥ唽 | 寰呯‘璁ゆ槸鍚?G1 鑼冨洿 | 3.18 | ZH/TCX |
| G2锛歚multihead_latent_attention` prefill/decode 鍒嗚В | 闃诲锛堢瓑 CSV锛?| 3.20 | ZH |
| G2锛歚mlapo` / `mlapo_quant` 鍒嗚В | 闃诲锛堢瓑 CSV锛?| 3.20 | ZH |
| FIA microbenchmark CSV 閲囬泦 | 寰呬氦浠?| 3.18 | TCX |
| TransposeBatchMatMul.csv 閲囬泦 | 寰呬氦浠?| 3.18 | HDY |

---

## 6. 寮€鏀鹃棶棰?

1. **MoE CSV 鏍煎紡**锛氱幇鏈夊師濮?profiling CSV 鐨?Input Shapes 鍚?NPU 鍐呴儴鍙傛暟锛宍tc_input_count` 鎴柇鏂规鏄惁瓒冲锛岃繕鏄渶瑕侀噸鏂伴噰闆?microbenchmark 鏍煎紡锛?

2. **MLA prefill kv_c_normed shape**锛歱refill 璺緞鐨?MatMulV2 绗竴涓緭鍏?kv_c_normed 涓嶅湪 `multihead_latent_attention` 鐨?args 涓紝闇€瑕佷粠 kv_cache shape 鍙嶆帹 kv_lora_rank銆傜‘璁?kv_cache shape 绾﹀畾锛歚(total_blocks, block_size, kv_lora_rank + qk_rope_head_dim)`锛宬v_lora_rank 鍙粠 `kv_cache.shape[-1] - qk_rope_head_dim` 璁＄畻锛屼絾 qk_rope_head_dim 鏄?scalar 鍙傛暟锛屼笉鍦?args 涓€?*闇€瑕佺‘璁?prefill 璺緞鐨?shape 鎻愬彇鏂瑰紡銆?*

3. **moe_gating_topk 鏄惁绾冲叆 G1**锛歍C 褰撳墠鐢?`aten.topk` 瀹炵幇锛孨PU 鏈変笓鐢?`MoeGatingTopK` kernel銆侴1 鏄惁闇€瑕佹柊澧?TC 绠楀瓙娉ㄥ唽锛岃繕鏄粎鍋?op_mapping 鍗犱綅锛?

