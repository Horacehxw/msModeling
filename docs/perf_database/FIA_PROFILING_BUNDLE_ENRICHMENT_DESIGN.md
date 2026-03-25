# FIA Profiling Bundle 增强设计

## 1. 背景

当前 FIA 数据库生成和 replay 流程如下：

1. 对模型 profiling 目录运行 `tools/perf_data_collection/parse_kernel_details.py`，例如：

```bash
python tools/perf_data_collection/parse_kernel_details.py \
  --kernel-details-path G:\仿真开发\profiling\最最新Profiling_0319\profiler-dsv3-input4096-output1536-concurrency8-rrate4 \
  --device ATLAS_800_A3_752T_128G_DIE \
  --vllm-ascend-version 0.20.0
```

这一步会生成初版算子数据库，其中包括 `FusedInferAttentionScore.csv`。

2. 运行 `tools/perf_data_collection/start_microbench.py`。

这一步会 replay 算子，并把 `Average Duration(us)` 回写到 CSV。对于 FIA，这个值来自 `tools/perf_data_collection/op_replay/FusedInferAttentionScore_run.py` 的 `msprof` replay 结果。

当前观察到的问题是：

- `Profiling Average Duration(us)` 表示 FIA 在整网执行中的平均耗时。
- `Average Duration(us)` 表示 FIA 在 replay microbench 中的平均耗时。
- 这两个值经常存在较大 gap，尤其是在 paged/dynamic FIA 场景下。

目前的结论是：

- gap 的主因并不在 `start_microbench.py` 的回写逻辑本身。
- 更核心的问题是 `parse_kernel_details.py` 生成的初版数据库没有为 FIA 保留足够的 runtime metadata。
- 因此 `FusedInferAttentionScore_run.py` 只能依赖启发式推断关键运行参数，而这些推断和真实模型执行状态并不一致。

## 2. 根因分析

### 2.1 现有 `parse_kernel_details.py` 输出以 shape 为中心

当前数据库主要保留：

- 输入 shape
- 输入 dtype
- 输入 format
- 输出 shape
- 输出 dtype
- profiling 耗时统计

对于很多常规算子，这些信息已经足够，但对 FIA 不够。

### 2.2 FIA replay 依赖当前未保留的 runtime metadata

对 FIA 而言，以下字段会直接影响 kernel 行为和时延：

- `actual_seq_lengths`
- `actual_seq_lengths_kv`
- `block_table`
- `num_key_value_heads`
- `sparse_mode`
- `input_layout`
- `block_size`
- K/V 是真实序列数据还是 KV cache pool view
- attention 所处阶段或状态，例如 prefill 与 decode-like paged path

如果这些字段缺失，replay 只能“猜”。

### 2.3 当前 FIA replay 含有非等价重建

当前的 `tools/perf_data_collection/op_replay/FusedInferAttentionScore_run.py` 存在若干近似重建：

- paged `actual_seq_lengths_kv` 被重建成每个 request 仅 1 个 block
- paged `num_key_value_heads` 被简化成 `1`
- `sparse_mode` 是依据 mask shape 推断，而不是真实模型运行值
- K/V tensor 被重建成独立随机 tensor，而不是 cache-pool-backed view
- `pre_tokens` 和 `next_tokens` 被显式传成 `65535`

这些做法能保证 replay 合法，但不等价于整网运行时状态。

## 3. 为什么要升级 `parse_kernel_details.py`

目标方向应该是：

- 在解析 FIA 对应 profiling 目录时，自动提取 replay 所需的 runtime metadata
- 将这些 metadata 作为新增列写入 `FusedInferAttentionScore.csv`
- 让 FIA replay 优先消费这些新列，而不是只根据 shape 推断

这件事是可行的，但前提是把 `parse_kernel_details.py` 从“单文件 kernel parser”升级成“profiling bundle parser”。

它不能再只依赖 `kernel_details.csv`。对于 FIA，应该把 profiling 目录整体当作 bundle 联合解析：

- `kernel_details.csv`
- `operator_details.csv`
- `trace_view.json`
- 可选：`ascend_pytorch_profiler_0.db`
- 可选：`analysis.db`

## 4. 数据可得性评估

基于当前可用的 profiling 目录：

- `G:\仿真开发\profiling\最最新Profiling_0319`

可以确认以下文件存在且与问题相关：

- `kernel_details.csv`
- `operator_details.csv`
- `trace_view.json`
- `api_statistic.csv`
- `op_statistic.csv`
- `ascend_pytorch_profiler_0.db`
- `analysis.db`

这意味着在不重新跑模型 profiling 的前提下，就可以先做一版离线增强方案。

但这里有一个关键限制：

- 如果 profiling bundle 只保存了 tensor 的 shape，而没有保存 `actual_seq_lengths` 或 `actual_seq_lengths_kv` 的实际值，那么 parser 最多只能恢复结构和 shape 级 metadata，未必能恢复逐 request 的真实列表值。

因此设计上必须同时支持：

- 完整 runtime metadata
- 部分 runtime metadata，并且显式标记完备度

## 5. 设计目标

升级 FIA 数据库生成链路，使其满足：

1. `parse_kernel_details.py` 能从 profiling bundle 中提取 FIA runtime metadata
2. `FusedInferAttentionScore.csv` 能以新增列形式保存这些 metadata
3. `FusedInferAttentionScore_run.py` 能基于这些 metadata 进行 replay
4. replay 出来的 FIA 状态尽可能接近整网真实执行状态
5. 耗时比较使用比当前 task-vs-kernel 更对齐的计时口径

## 6. 总体架构

### 6.1 解析阶段

输入：

- profiling 目录，而不仅仅是 `kernel_details.csv`

输出：

- 与现有逻辑一致的普通算子 CSV
- FIA 行额外补充 runtime metadata 列

### 6.2 Replay 阶段

输入：

- 增强后的 `FusedInferAttentionScore.csv`

行为：

- 优先使用保存下来的 runtime metadata
- 缺失时再回退到现有启发式推断

### 6.3 对比阶段

同时保存和比较：

- replay task duration
- replay kernel duration

这样 FIA 就可以用与整网 profiling 更一致的计时层级做比较。

## 7. 具体实现设计

## 7.1 `parse_kernel_details.py` 改造

### 7.1.1 修改输入语义

当前行为：

- `--kernel-details-path` 主要被当作 kernel details 数据路径

建议行为：

- 允许 `--kernel-details-path` 直接指向 profiling 目录
- 当该路径是目录时，自动加载整个 profiling bundle

建议新增辅助函数：

```python
def resolve_profiling_bundle(path: Path) -> ProfilingBundle:
    ...

def load_kernel_details_csv(bundle: ProfilingBundle) -> list[dict[str, str]]:
    ...

def load_operator_details_csv(bundle: ProfilingBundle) -> list[dict[str, str]]:
    ...

def load_trace_view_json(bundle: ProfilingBundle) -> list[dict[str, object]]:
    ...

def load_profiler_db_optional(bundle: ProfilingBundle) -> ProfilerDb | None:
    ...
```

建议新增 bundle 数据结构：

```python
@dataclass
class ProfilingBundle:
    root_dir: Path
    kernel_details_csv: Path | None
    operator_details_csv: Path | None
    trace_view_json: Path | None
    profiler_db: Path | None
    analysis_db: Path | None
```

### 7.1.2 增加 FIA 专用 enrich 分支

在导出阶段增加 FIA 专门逻辑：

```python
if op_type == "FusedInferAttentionScore":
    aggregated_row = enrich_fia_row_from_bundle(
        aggregated_row=aggregated_row,
        bundle=bundle,
        grouped_source_rows=source_rows,
    )
```

建议新增辅助函数：

```python
def enrich_fia_row_from_bundle(
    aggregated_row: dict[str, str],
    bundle: ProfilingBundle,
    grouped_source_rows: list[dict[str, str]],
) -> dict[str, str]:
    ...

def extract_fia_runtime_metadata(
    bundle: ProfilingBundle,
) -> list[FiaRuntimeMetadata]:
    ...

def match_fia_runtime_metadata(
    aggregated_row: dict[str, str],
    runtime_rows: list[FiaRuntimeMetadata],
) -> FiaRuntimeMetadata | None:
    ...
```

### 7.1.3 FIA runtime metadata 结构

建议数据结构：

```python
@dataclass
class FiaRuntimeMetadata:
    source_profile: str
    op_state: str
    input_shapes: str
    output_shapes: str
    actual_seq_lengths_shape: str | None
    actual_seq_lengths_values: str | None
    actual_seq_lengths_kv_shape: str | None
    actual_seq_lengths_kv_values: str | None
    block_table_shape: str | None
    block_table_valid_blocks: str | None
    num_heads: int | None
    num_key_value_heads: int | None
    sparse_mode: int | None
    input_layout: str | None
    block_size: int | None
    attn_state: str | None
    kv_cache_mode: str | None
    metadata_completeness: str
```

### 7.1.4 FIA CSV 新增列

建议在 `FusedInferAttentionScore.csv` 中新增以下列：

- `Runtime source_profile`
- `Runtime actual_seq_lengths_shape`
- `Runtime actual_seq_lengths_values`
- `Runtime actual_seq_lengths_kv_shape`
- `Runtime actual_seq_lengths_kv_values`
- `Runtime block_table_shape`
- `Runtime block_table_valid_blocks`
- `Runtime num_heads`
- `Runtime num_key_value_heads`
- `Runtime sparse_mode`
- `Runtime input_layout`
- `Runtime block_size`
- `Runtime attn_state`
- `Runtime kv_cache_mode`
- `Runtime metadata_completeness`

优先级建议：

- P0：
  - `Runtime actual_seq_lengths_values`
  - `Runtime actual_seq_lengths_kv_values`
  - `Runtime block_table_shape`
  - `Runtime num_key_value_heads`
  - `Runtime sparse_mode`
  - `Runtime input_layout`
  - `Runtime block_size`
- P1：
  - `Runtime block_table_valid_blocks`
  - `Runtime attn_state`
  - `Runtime kv_cache_mode`
  - `Runtime metadata_completeness`
- P2：
  - `Runtime actual_seq_lengths_shape`
  - `Runtime actual_seq_lengths_kv_shape`
  - `Runtime source_profile`

### 7.1.5 调整 FIA 分组键

对于 FIA，不应只依赖现有 shape signature 分组。

建议将以下 runtime 维度也纳入 FIA 分组或匹配键：

- `Input Shapes`
- `Input Data Types`
- `Input Formats`
- `Output Shapes`
- `Output Data Types`
- `Runtime actual_seq_lengths_shape`
- `Runtime actual_seq_lengths_kv_shape`
- `Runtime block_table_shape`
- `Runtime num_key_value_heads`
- `Runtime sparse_mode`
- `Runtime input_layout`

这样可以避免把不同真实执行状态错误合并到同一行 FIA 记录里。

## 7.2 `FusedInferAttentionScore_run.py` 改造

### 7.2.1 增加 runtime 列解析辅助函数

建议新增：

```python
def parse_runtime_list_field(value: str) -> list[int] | None:
    ...

def parse_runtime_int(value: str) -> int | None:
    ...

def get_runtime_override(row: dict[str, str], key: str):
    ...
```

### 7.2.2 从“启发式优先”改成“metadata 优先”

当前逻辑是启发式优先，建议改成：

1. 如果 `Runtime ...` 列存在，优先使用
2. 缺失时再退回现有推断逻辑

适用范围包括：

- `actual_seq_lengths`
- `actual_seq_lengths_kv`
- `num_heads`
- `num_key_value_heads`
- `sparse_mode`
- `input_layout`
- `block_size`

### 7.2.3 关键逻辑点

#### A. `infer_seq_lens_kv()`

当前问题：

- paged case 直接返回 `[block_size] * batch_size`

建议修改：

- 优先读取 `Runtime actual_seq_lengths_kv_values`
- 如果拿不到，再读取 `Runtime actual_seq_lengths_kv_shape`
- 仍然拿不到时，最后再回退到当前启发式

#### B. `infer_case_args()`

当前问题：

- paged case 把 `num_key_value_heads` 简化成 `1`

建议修改：

- 优先读取：
  - `Runtime num_heads`
  - `Runtime num_key_value_heads`
  - `Runtime input_layout`
  - `Runtime block_size`
- 缺失时才推断

#### C. `infer_sparse_mode()`

当前问题：

- 只在 mask 形状是 `(2048, 2048)` 时推断成 `3`

建议修改：

- 优先读取 `Runtime sparse_mode`
- 如果缺失且是标准 FIA causal path，默认使用 `3`
- mask-shape 推断只保留为最后兜底逻辑

#### D. `build_block_table_tensor()`

当前问题：

- 构造出来的是“合法但简化”的 block table

建议修改：

- 使用 `Runtime block_table_shape`
- 如果存在，进一步使用 `Runtime block_table_valid_blocks`
- 生成与 runtime metadata 更接近的 block_table，每行有效 block 数与真实分布一致

#### E. K/V tensor 布局

当前问题：

- K/V 被构造成独立随机 tensor

建议修改：

- paged 场景下先创建 cache pool tensor
- 再基于 block table 构造 K/V 访问视图

这仍然不是 bitwise 级别的整网状态还原，但会更接近真实访存模式。

#### F. `pre_tokens` 和 `next_tokens`

当前问题：

- 显式传成 `65535`

建议修改：

- 如果 runtime metadata 能提供真实值，就使用真实值
- 否则不显式传入，让 kernel 走默认行为

## 7.3 `start_microbench.py` 改造

### 7.3.1 保留现有 task duration

保留：

- `Average Duration(us)` 作为 replay 的对外查询列

### 7.3.2 增加 replay kernel duration

新增：

- `MicroBench Kernel Duration(us)`

这一列应优先从 replay 产物中的 kernel 级计时提取；如果 replay 目录没有 `kernel_details`，则应从 `task_time_*.csv` 中提取更接近设备执行的时长，并保持 `Average Duration(us)` 接口名不变。

### 7.3.3 对比规则

优先比较：

- `MicroBench Kernel Duration(us)` vs `Profiling Average Duration(us)`

其次比较：

- `Average Duration(us)` vs `Profiling Average Duration(us)`

这样可以消除一部分当前 task-vs-kernel 口径不一致带来的误差。

## 8. FIA runtime metadata 匹配策略

增强阶段必须把 profiling bundle 中抽取到的 FIA metadata 正确匹配到 CSV 行。

建议优先级如下：

1. 先做精确匹配：
   - `Input Shapes`
   - `Input Data Types`
   - `Input Formats`
   - `Output Shapes`
   - `Output Data Types`
2. 如果仍有多个候选，再用以下字段细化：
   - `actual_seq_lengths` shape
   - `actual_seq_lengths_kv` shape
   - `block_table` shape
3. 如果仍然无法唯一确定，则保留为不同记录，不要强行合并

这是 FIA 特别需要注意的点，因为它存在“顶层 tensor shape 相似，但真实 runtime state 不同”的情况。

## 9. 预期效果

完成这套重构后：

- FIA replay 不再把 paged KV length 压缩成“每个 request 仅 1 个 block”
- FIA replay 不再默认所有 paged case 都是 `num_key_value_heads = 1`
- FIA 的 sparse mode 和 layout 会更接近真实运行状态
- paged FIA 的 K/V 内存布局会更贴近 cache-pool 访问模式
- replay 和 profiling 的比较会使用更对齐的计时口径

预期结果：

- paged FIA 的大 gap 会明显收敛
- replay 值将从“合法但偏乐观的下界估计”，变成“可用于近似整网 runtime 的参考值”

## 10. 剩余限制

即便 parser 升级完成，如果 profiling bundle 本身没有保存精确 runtime 值，仍然会存在部分 FIA metadata 无法恢复的问题。

因此设计上必须显式支持：

- `Runtime metadata_completeness = complete`
- `Runtime metadata_completeness = partial`
- `Runtime metadata_completeness = inferred`

这样 replay 和后续分析才能区分一行数据到底是：

- 真正由 runtime metadata 支撑
- 部分重建
- 仍然主要依赖启发式推断

## 11. 推荐实施顺序

### Phase 1

1. 升级 `parse_kernel_details.py`，支持解析 profiling bundle
2. 给 `FusedInferAttentionScore.csv` 增加 FIA runtime metadata 列
3. 更新 `FusedInferAttentionScore_run.py`，优先消费 runtime 列

### Phase 2

4. 在 `start_microbench.py` 中增加 `MicroBench Kernel Duration(us)`
5. 改进 FIA paged K/V 重建方式，改为 pool + view 访问模式

### Phase 3

6. 如果 profiling bundle 仍然缺少精确 FIA metadata，则在模型 profiling 阶段增加 FIA runtime logging

## 12. 总结

正确方向不是继续孤立地打磨 FIA replay 的启发式逻辑。

正确方向是：

- 把 `parse_kernel_details.py` 升级成 profiling-bundle parser
- 在生成数据库时提取 FIA runtime metadata
- 将这些 metadata 保存到 `FusedInferAttentionScore.csv`
- 让 FIA replay 直接使用这些 metadata

这样才能把 FIA replay 从：

- 基于 shape 的合法重建

推进成：

- 基于 profiling 的 runtime-aware 重建

这也是让 FIA microbench 数据在工程上有意义地对齐整网 profiling 的唯一可行路径。
