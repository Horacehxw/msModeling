基于 `git diff feat/perf-database...HEAD` 的 13 个 commit，我的结论是：这批改动把 spec 的主框架基本落地了，但还**不算与 spec 完全一致**。有 2 个关键偏移，其中 1 个我认为是不合理且有明显行为风险的。

**Findings**
1. 高严重度：decomposed composite 在“全部 sub-kernel 都 miss”时仍返回 `PARTIAL`，会把一个完整 miss 当成 `0us` 经验值，而不是回退 analytic。这和 spec “some HIT and others MISS 才 PARTIAL” 不一致，也会绕过 interpolation/analytic fallback。[profiling_data_source.py#L1027](/Users/horacehxw/Projects/msmodeling/.claude/worktrees/fix-dsv3-mla-miss/tensor_cast/performance_model/profiling_database/profiling_data_source.py#L1027) [empirical.py#L276](/Users/horacehxw/Projects/msmodeling/.claude/worktrees/fix-dsv3-mla-miss/tensor_cast/performance_model/empirical.py#L276) [interpolating_data_source.py#L71](/Users/horacehxw/Projects/msmodeling/.claude/worktrees/fix-dsv3-mla-miss/tensor_cast/performance_model/profiling_database/interpolating_data_source.py#L71) 相关测试还把这个行为固化了：[test_mla_decomposition.py#L897](/Users/horacehxw/Projects/msmodeling/.claude/worktrees/fix-dsv3-mla-miss/tests/perf_database/test_mla_decomposition.py#L897)。
2. 高严重度：spec 为修复 quant MLA/MLAPO miss 引入的 `SubKernelSpec.tc_input_count` / `alternate_kernel_types`，目前只“加了字段”，但 decomposer 没真正填写，导致 root cause 没闭环。消费端已经支持这些字段：[profiling_data_source.py#L1000](/Users/horacehxw/Projects/msmodeling/.claude/worktrees/fix-dsv3-mla-miss/tensor_cast/performance_model/profiling_database/profiling_data_source.py#L1000)，字段也加了：[profiling_data_source.py#L341](/Users/horacehxw/Projects/msmodeling/.claude/worktrees/fix-dsv3-mla-miss/tensor_cast/performance_model/profiling_database/profiling_data_source.py#L341)，但 quant 子 kernel 构造时没传：[profiling_data_source.py#L609](/Users/horacehxw/Projects/msmodeling/.claude/worktrees/fix-dsv3-mla-miss/tensor_cast/performance_model/profiling_database/profiling_data_source.py#L609) [profiling_data_source.py#L615](/Users/horacehxw/Projects/msmodeling/.claude/worktrees/fix-dsv3-mla-miss/tensor_cast/performance_model/profiling_database/profiling_data_source.py#L615)。而 standalone quant matmul mapping 明确依赖 `tc_input_count: 2`：[op_mapping.yaml#L149](/Users/horacehxw/Projects/msmodeling/.claude/worktrees/fix-dsv3-mla-miss/tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5/op_mapping.yaml#L149)。这意味着 spec 里“解决 QuantBatchMatmulV3 2-vs-4 input miss”的核心目标其实没有完整实现，当前 coverage 改善更像是被 `PARTIAL` 掩盖。
3. 中严重度：`_query_by_attn_params` 没实现 spec 里的 `input_layout` optional tie-breaker；真实 CSV 也确实有 `Runtime input_layout` 列。[profiling_data_source.py#L1088](/Users/horacehxw/Projects/msmodeling/.claude/worktrees/fix-dsv3-mla-miss/tensor_cast/performance_model/profiling_database/profiling_data_source.py#L1088) [FusedInferAttentionScore.csv#L1](/Users/horacehxw/Projects/msmodeling/.claude/worktrees/fix-dsv3-mla-miss/tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5/FusedInferAttentionScore.csv#L1) 另外 `_lookup_attention` 会把 shared-core 的所有 miss 都压成 `shape_mismatch`，诊断比重构前弱。[profiling_data_source.py#L1378](/Users/horacehxw/Projects/msmodeling/.claude/worktrees/fix-dsv3-mla-miss/tensor_cast/performance_model/profiling_database/profiling_data_source.py#L1378)
4. 中严重度：PARTIAL 的数据结构和展示没有完全按 spec 落地。spec 要 `hit_kernels` + `missed_kernels`，现在只回了 `kernel_type` + `missed_kernels`。[profiling_data_source.py#L1035](/Users/horacehxw/Projects/msmodeling/.claude/worktrees/fix-dsv3-mla-miss/tensor_cast/performance_model/profiling_database/profiling_data_source.py#L1035) spec 要 `HIT / PARTIAL / MISS` 独立展示，`EmpiricalPerformanceModel.log_stats()` 仍只有 HIT/MISS 两类日志。[empirical.py#L326](/Users/horacehxw/Projects/msmodeling/.claude/worktrees/fix-dsv3-mla-miss/tensor_cast/performance_model/empirical.py#L326)

**Spec 对照**
1. `lookup` 总体架构和 dispatch：基本一致。`composite` / `comm` / `attention_special` / `elementwise` / `moe_fused` / default 都已接通。
2. `_query_by_shapes` 抽取：一致。旧 `_lookup_compute_by_shapes` 已删除，`_lookup_compute`、generic composite、decomposed composite 都改为复用。这里有一个 spec 外扩展：多了 `csv_file` 参数，我认为是合理自主设计。
3. `_query_by_attn_params` 抽取：部分一致。主干能力落地了，但缺 `input_layout` tie-breaker，miss 诊断也退化。
4. `SubKernelSpec` 扩展：表面一致，实质部分不一致。字段加了，但没有在 quant decomposer 里真正使用。
5. DFC EP Size matching：基本一致。`query_mode: moe_fused`、`_lookup_moe()`、`parallel_config` 透传都已实现。[model_runner.py#L75](/Users/horacehxw/Projects/msmodeling/.claude/worktrees/fix-dsv3-mla-miss/tensor_cast/core/model_runner.py#L75) 这里额外加了 `ep_size` kwarg，我认为是合理自主设计，主要方便直接构造/测试。[profiling_data_source.py#L653](/Users/horacehxw/Projects/msmodeling/.claude/worktrees/fix-dsv3-mla-miss/tensor_cast/performance_model/profiling_database/profiling_data_source.py#L653)
6. Composite partial match：部分一致。`some hit + some miss` 的累计逻辑有了，但“all miss 也 PARTIAL”是 spec 外行为，而且我认为不合理。
7. `QuerySource.PARTIAL`：一致。只是实现上用了 `Enum.auto()`，不是 spec 文中的字符串值，这属于实现细节。
8. `EmpiricalPerformanceModel` 对 PARTIAL 的计数语义：基本一致。PARTIAL 参与 E2E sum，也按 MISS 计数；但展示格式没补齐，且被 all-miss partial 的实现拖出了错误语义。
9. Generic composite 不使用 partial：一致。
10. 删除/重构列表：大体一致。`_lookup_compute_by_shapes`、`_lookup_attention_by_params` 已删，核心调用链已替换。

**自主设计**
- 合理的：
  - `ProfilingDataSource.__init__(..., *, ep_size=None)` 作为 `parallel_config` 的 fallback。
  - `_query_by_shapes(..., csv_file=None)`，保留旧 mapping 的 filename 解耦能力。
  - `InterpolatingDataSource` 也复用 shared core，减少重复逻辑。
- 不合理的：
  - “all miss => PARTIAL(0us)”。
  - 用 `kernel_type` 代替 spec 里的 `hit_kernels`。
  - 没把 `input_layout` tie-breaker 一起补齐。

**验证**
`PYTHONPATH=. pytest -q tests/perf_database/test_profiling_data_source.py tests/perf_database/test_fia_enriched_lookup.py tests/perf_database/test_mla_decomposition.py tests/perf_database/test_empirical_metrics.py` 通过，`225 passed in 1.48s`。但这些测试里已经把 `all miss => PARTIAL with 0 latency` 这个偏移写成了预期，所以“测试全绿”不能证明与 spec 完全一致。
