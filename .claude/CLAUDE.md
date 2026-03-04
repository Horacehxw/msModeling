## Project-Specific References

### Documentation
- AI Configurator: local `/home/horacehxw/Projects/aiconfigurator`, GitHub: https://github.com/ai-dynamo/aiconfigurator
- op-plugin (torch_npu op mapping): `/home/horacehxw/Projects/op-plugin/`, master mapping: `op_plugin/config/op_plugin_functions.yaml` (7148 lines, 1200+ ops)
- Design doc: `docs/OPERATOR_PERF_DATABASE_DESIGN_zh_v1.2.md` (current version)
- Design doc examples: `docs/examples/op_mapping_example.yaml`, `docs/examples/comm_config_example.yaml`
- Wiki: https://deepwiki.com/Horacehxw/msModeling
- gitcode remote: `https://gitcode.com/Ascend/msmodeling.git` (develop branch has latest fused ops)
- HCCL Test tool doc: https://www.hiascend.com/document/detail/zh/mindstudio/70RC1/mscommandtoolug/mscommandug/auxiliarydevtool_0017.html

### Profiling Data (WSL mount)
- DeepSeekV3 Decode: `/mnt/d/Data/Profiling/prof-torchair-deepseekv3-decode/feb1abe5c743_785216_20251218195608873_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv`
- Qwen3-30B Prefill: `/mnt/d/Data/Profiling/comparison_qwen3_30b/kernel_details.csv`
- Qwen3-30B PD: `/mnt/d/Data/Profiling/profiling-qwen3-30b-pd_tegether/`

### kernel_details.csv Column Reference
- **Name**: Full hierarchical kernel name (e.g., `aclnnMatmulWeightNz_MatMulCommon_MatMulV2`), includes aclnn prefix + implementation path + base kernel
- **Type**: Base kernel type (e.g., `MatMulV2`, `Fill`, `FusedInferAttentionScore`). For custom/fused ops, Name == Type
- **建议用 Type 列做算子聚合**：Type 列对应硬件内核类型，Name 列含实现细节过于具体
- Required columns: `Name, Type, Input Shapes, Input Data Types, Input Formats, Output Shapes, Output Data Types, Output Formats, Duration(us)`
- Extended: `Accelerator Core, Block Dim, aicore_time(us), aic_mac_time(us), aic_mac_ratio, aic_scalar_time(us), aiv_time(us), aiv_vec_time(us), aiv_scalar_time(us), aiv_mte2_time(us), aiv_mte3_time(us), cube_utilization(%)`
- Data types seen: `DT_BF16, INT8, INT32, INT64, FLOAT, BOOL, DT_UNDEFINED`
- Formats seen: `ND` (normal dense), `FRACTAL_NZ` (tiled format for matmul weights), `NCL`, `NULL`

### Key Profiling Operator Counts (DSV3 Decode, 32 cards)
QuantBatchMatmulV3: 15006, AscendQuantV2: 10004, Add: 7545, TransposeBatchMatMul: 5002, InplaceAddRmsNorm: 5002, DequantSwigluQuant: 4879, GroupedMatmul: 4756, FusedInferAttentionScore: 2501, KvRmsNormRopeCache: 2501, InterleaveRope: 2501, DynamicQuant: 2501, MoeGatingTopK: 2378, MoeDistributeDispatchV2: 2378, MoeDistributeCombineV2: 2378, MatMul: 2378, MatMulV2: 41

### Key Profiling Operator Counts (Qwen3-30B Prefill, 16 cards)
TensorMove: 386, hcom_allReduce_: 276, MatMulV2: 275, AddRmsNorm: 131, FusedInferAttentionScore: 67, SwiGlu: 67, ReshapeAndCacheNdKernel: 67, split_qkv_rmsnorm_rope_kernel: 64

### vLLM Custom Ops References
- CustomOp replacement tracker: https://github.com/vllm-project/vllm/issues/32676
- torch_npu operator inventory: https://github.com/vllm-project/vllm-ascend/issues/1511
- vLLM torch_bindings.cpp: https://github.com/vllm-project/vllm/blob/main/csrc/torch_bindings.cpp
- MC2 (MatMul-AllReduce fusion): `torch_npu.npu_mm_all_reduce_base`, ~20% prefill improvement with TP
- MC2 graph patterns: https://github.com/vllm-project/vllm-ascend/issues/6092, https://github.com/vllm-project/vllm-ascend/issues/5743

### AI Configurator Key Findings
- Data stored as CSV (`.txt` suffix) in `src/aiconfigurator/systems/data/{device}/{framework}/{version}/`, managed via Git LFS (`.gitattributes`: `systems/**/*.txt filter=lfs`)
- Each row self-describing with repeated framework/version/device columns for cross-version aggregation and independent CSV loading
- 2D+1D hybrid interpolation: bilinear on two dims, then 1D on third
- sqrt transform for O(n^2) attention operators before interpolation
- DatabaseMode cascade: SILICON -> HYBRID -> EMPIRICAL -> SOL
- System metadata in YAML: `systems/{device}.yaml` with GPU specs, bandwidth, memory
- Total: 155 CSV files, ~186MB, ~118K rows across 6+ GPU types, 3+ frameworks

### OpInvokeInfo Key Fields
- `func`: torch op (e.g., `torch.ops.aten.mm.default`)
- `args/kwargs`: input tensors and scalars
- `cache_key`: SHA256 hash of func + shapes + dtypes + strides + device (captures dtype and layout)
- `PerformanceProperties`: compute_ops (per-dtype MMA/GP), memory read/write/readwrite bytes
- Properties registered via `@OpInvokeInfo.register_op_properties(torch.ops.aten.mm.default)`

### Develop Branch Fused Op Status (gitcode/develop)
- **Closed**: SwiGlu fusion (patterns/swiglu.py), GroupedMatmul+SwiGlu fusion (freezing_passes/grouped_matmul_swiglu_pass.py, 5 quant variants)
- **Open**: split_qkv_rmsnorm_rope, KvRmsNormRopeCache, MLA decomposition, aten.topk/MoeGatingTopK disambiguation, MC2

### Naming Convention for Documents
- 在进行项目内部分析时可以用本地路径，最终文档报告引用 GitHub 仓库链接

### User Workflow Preferences
- **不要自动 commit**：除非用户明确要求，否则不要执行 git commit
