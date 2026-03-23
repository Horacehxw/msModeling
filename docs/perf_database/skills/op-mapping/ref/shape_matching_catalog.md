# 褰㈢姸鍖归厤瑙勫垯鐩綍 (Shape Matching Catalog)

## 姒傝堪

TC 妯℃嫙鐨勫紶閲忓舰鐘朵笌 NPU profiling CSV 鐨勫舰鐘跺瓨鍦ㄧ郴缁熸€у樊寮傘€?
`profiling_data_source.py` 鐨?`_inputs_match()` 鎸夐『搴忓皾璇曚互涓嬪尮閰嶈鍒欍€?

## 10 绉嶅舰鐘跺樊寮傜被鍨?

### 1. 鎵规缁村害鍓ョ (Batch dim strip)
- **TC**: `(1, S, D)` 鈥?淇濇寔鏄惧紡 batch dim
- **NPU**: `(S, D)` 鈥?鍗?batch 鏃剁渷鐣?
- **澶勭悊**: `_strip_batch_dim()` 鍓ョ绗竴缁?=1 鐨勬儏鍐?
- **閫傜敤**: 鎵€鏈?3D鈫?D 绠楀瓙

### 2. 搴忓垪濉厖 (Seq padding)
- **TC**: `ceil(S/block) * block` 鈥?濉厖鍒?NPU tile 瀵归綈
- **NPU**: raw S 鈥?鍘熷搴忓垪闀垮害
- **澶勭悊**: `_shapes_match_with_padding()` 瀹瑰繊 block 鈭?{16, 32, 64}
- **閫傜敤**: matmul, norm 绛夎绠楃畻瀛?

### 3. FRACTAL_NZ 鏍煎紡
- **TC**: ND `(K, N)` 鈥?鏍囧噯浜岀淮
- **NPU**: `[H, W, bh, bw]` 鈥?FRACTAL_NZ 鍒嗗潡甯冨眬
- **澶勭悊**: `fractal_nz_to_nd()` 杩樺師: `(H*bw, W*bh)`
- **閫傜敤**: 鎵€鏈夋爣璁?`FRACTAL_NZ` 鏍煎紡鐨勬潈閲?

### 4. ND 鏉冮噸杞疆
- **TC**: `(K, N)` 鈥?鏉ヨ嚜 `weight.T`
- **NPU**: `(N, K)` 鈥?瀛樺偍椤哄簭涓嶅悓
- **澶勭悊**: 瀵?`_MATMUL_KERNELS` 鐨勭 2+ 涓緭鍏ユ鏌ヨ浆缃?
- **閫傜敤**: MatMulV2, MatMulV3, MatMulCommon, QuantBatchMatmulV3, BatchMatMulV2

### 5. SwiGlu 杈撳叆鎷兼帴
- **TC**: 2 涓緭鍏?`(S, D/2)` 鈥?gate 鍜?up 鍒嗗紑
- **NPU**: 1 涓緭鍏?`(S, D)` 鈥?鎷兼帴鍚庣殑
- **澶勭悊**: `_SWIGLU_KERNELS` 鐗规畩閫昏緫,鍚堝苟 last dim
- **閫傜敤**: SwiGlu kernel

### 6. RoPE 甯冨眬杞疆
- **TC**: `(B, H, S, D)` 椤哄簭 `[Q, K, cos, sin]`
- **NPU**: `(B, S, H, D)` 椤哄簭 `[K, Q, cos, sin]`
- **澶勭悊**: `_normalize_rope_inputs()` 浜ゆ崲 Q鈫擪 + 杞疆 H鈫擲
- **閫傜敤**: `_ROPE_KERNELS` (InterleaveRope, ApplyRotaryPosEmb, _triton_rope, etc.)

### 7. RoPE 澶?kernel 鍙樹綋
- **TC**: 鍗曚釜 `apply_rope` op
- **NPU**: InterleaveRope (interleave mode) / ApplyRotaryPosEmb (neox) / _triton_rope (CANN 8.5)
- **澶勭悊**: `alternate_kernel_types` 鍒楄〃
- **閫傜敤**: tensor_cast.apply_rope.default

### 8. 澶嶅悎绠楀瓙鍒嗚В
- **TC**: 铻嶅悎 op (濡?matmul_all_reduce)
- **NPU**: 鍙兘鎷嗕负鐙珛 kernel
- **澶勭悊**: `composite: true` + `sub_kernels` 鍒楄〃,`_lookup_composite()` 鍒嗗埆鏌ヨ姹傚拰
- **閫傜敤**: MC2 (matmul+allreduce), MLA (bmm+FIA+transpose_bmm), MLAPO (matmul+kvnormrope)

### 9. 鎵规灞曞钩 (Flatten batch)
- **TC**: `(B, M, D)` 鈥?3D (batch, seq, hidden)
- **NPU**: `(B*M, D)` 鈥?2D (tokens, hidden)
- **澶勭悊**: `_FLATTEN_BATCH_KERNELS` 闆嗗悎, 灏濊瘯 `(B*M, D)` 鍖归厤
- **閫傜敤**: AscendQuantV2, DynamicQuant, RmsNorm, AddRmsNormBias, AddRmsNorm, **DispatchFFNCombine**

### 10. 鏈淮鍚堝苟 (Merge last dims)
- **TC**: `(T, H, D)` 鈥?per-head 閲忓寲 (MLA)
- **NPU**: `(T, H*D)` 鈥?鍚堝苟涓?hidden_dim
- **澶勭悊**: `_MERGE_LAST_DIMS_KERNELS` 闆嗗悎
- **閫傜敤**: AscendQuantV2, DynamicQuant (MLA 杈撳嚭閲忓寲璺緞)

### 11. 杈撳嚭褰㈢姸鍖归厤 (Output-shape matching for elementwise ops)
- **TC**: 鍖归厤杈撳嚭褰㈢姸, 浠讳綍杈撳叆骞挎挱妯″紡
- **NPU**: 杈撳嚭褰㈢姸纭畾 (CSV `Output Shapes` 鍒?
- **Dtype**: 鏉惧紱 鈥?鎸夊瓧鑺傛瘮缂╂斁寤惰繜 (FP32/BF16 = 4/2 = 2脳)
- **澶勭悊**: `query_mode: elementwise` 鈫?`_lookup_elementwise()` / `_interpolate_elementwise()`
- **閫傜敤**: Add, Mul, Div (鍐呭瓨甯﹀鍙楅檺鐨勯€愬厓绱犵畻瀛?
- **涓?tc_input_count 浜掓枼**: 璁剧疆 `query_mode: elementwise` 鍚庝笉寰楄缃?`tc_input_count`

## 鏂板褰㈢姸瑙勫垯鐨勬祦绋?

1. 鍦?E2E 楠岃瘉涓彂鐜?`shape_mismatch` MISS
2. 瀵规瘮 TC 褰㈢姸 vs CSV 褰㈢姸,璇嗗埆绯荤粺鎬у樊寮?
3. 纭宸紓涓嶆槸鏁版嵁缂哄け(闇€妫€鏌?CSV 鏄惁鏈夎 batch size)
4. 鍦?`_inputs_match()` 涓坊鍔犳柊鐨勫尮閰嶈鍒?
5. 娣诲姞鍒板搴旂殑 kernel 闆嗗悎 (濡?`_FLATTEN_BATCH_KERNELS`)
6. 娣诲姞娴嬭瘯鐢ㄤ緥
7. 鏇存柊鏈枃妗?

## tc_input_count 涓庡舰鐘跺尮閰嶇殑浜や簰

`tc_input_count` 鍦ㄥ舰鐘跺尮閰嶄箣鍓嶆埅鏂緭鍏?
```python
# _lookup_compute() 涓?
tc_inputs = tc_inputs[:tc_input_count]  # 鎴柇 TC 杈撳叆

# _inputs_match() 涓?
csv_shapes = csv_shapes[:tc_input_count]  # 鎴柇 CSV 杈撳叆
```

鎴柇鍚?鍓╀綑杈撳叆鎸変笂杩?10 绉嶈鍒欏尮閰嶃€傝瑙?`ref/tc_input_count_rules.md`銆?

