# `tools/perf_data_collection` README

## 1. 这个目录是干什么的

`tools/perf_data_collection` 是这个仓库里“性能数据库数据生产工具链”的入口目录。它解决的是同一件事的几个阶段：

1. 从真实 profiling 结果里抽取算子样本，生成 perf database CSV。
2. 对难以仅靠 `kernel_details.csv` 还原的算子，补充运行时元数据，尤其是 `FusedInferAttentionScore`（下文简称 FIA）。
3. 基于数据库里的 shape / dtype / format，把单个算子重新 replay 出来，采集 microbench 数据并回写到 CSV。
4. 为缺样本或插值不稳定的算子批量扩充 shape grid。
5. 为通信算子生成、运行和清洗 HCCL microbench 数据。
6. 用离线指标检查“数据库预测”和“真实 profiling”是否足够接近。

一句话理解：

`docs/perf_database` 负责解释“为什么这样设计”，`tools/perf_data_collection` 负责把这些设计真正变成 CSV 数据、microbench 数据和校验结果。

---

## 2. 新手先建立一个整体心智模型

### 2.1 计算算子主线

最常见的链路是：

1. 跑整网 profiling，拿到 `kernel_details.csv` 或 profiling bundle。
2. 用 [`parse_kernel_details.py`](./parse_kernel_details.py) 把 profiling 聚合成按算子拆分的 CSV。
3. 如果是 FIA，再用 [`fill_fia_runtime_metadata.py`](./fill_fia_runtime_metadata.py) 把运行时 JSONL 回填到 `FusedInferAttentionScore.csv`。
4. 用 [`start_microbench.py`](./start_microbench.py) 驱动 [`op_replay`](./op_replay) 里的 replay 脚本，在 NPU 上重放这些算子。
5. 解析 `msprof` 输出，把 `Average Duration(us)` 和一批 profiling 平均列回写到对应 CSV。
6. 如果某些算子 shape 太稀疏，再用 [`generate_shape_grid.py`](./generate_shape_grid.py) 按约束扩样。

### 2.2 FIA 主线

FIA 是这个目录里最特殊的一类算子，因为它只靠 `Input Shapes` 往往不足以精确 replay。它还依赖：

- `actual_seq_lengths`
- `actual_seq_lengths_kv`
- `block_table`
- `num_heads`
- `num_key_value_heads`
- `input_layout`
- `sparse_mode`
- `block_size`

所以 FIA 的完整链路是：

1. `parse_kernel_details.py` 先从 profiling bundle 里生成初版 `FusedInferAttentionScore.csv`。
2. 如果 profiling 里没有真实运行时值，就在 `vllm-ascend` 的 FIA 调用点埋点，导出 `fia_runtime_metadata.jsonl`。
3. 用 `fill_fia_runtime_metadata.py` 把 JSONL 里的真实 runtime metadata 回填进 CSV。
4. `op_replay/FusedInferAttentionScore_run.py` 优先消费这些 runtime 列，而不是只靠 shape 猜。

### 2.3 shape grid 主线

`generate_shape_grid.py` 不是“凭空造随机 shape”，而是“基于现有 CSV 模板做约束感知扩样”：

- 保留原始真实数据。
- 维持算子内部约束，比如 matmul contract、norm hidden 维度、rope/head_dim 关系、FIA 若干槽位关系。
- 让新生成的样本尽量仍然能被 `op_replay/*_run.py` 读取并执行。

---

## 3. 相关设计文档怎么看

如果你只想知道这个目录在做什么，优先看下面这些文档：

- [`docs/perf_database/OPERATOR_PERF_DATABASE_DESIGN_zh_v1.5.md`](../../docs/perf_database/OPERATOR_PERF_DATABASE_DESIGN_zh_v1.5.md)
  - 总设计，解释 perf database 的目标、CSV 结构、microbench 的角色。
- [`docs/perf_database/GENERATE_SHAPE_GRID.md`](../../docs/perf_database/GENERATE_SHAPE_GRID.md)
  - 解释为什么要扩样、shape 生成的约束来源、和 replay 的关系。
- [`docs/perf_database/FIA_PROFILING_BUNDLE_ENRICHMENT_DESIGN.md`](../../docs/perf_database/FIA_PROFILING_BUNDLE_ENRICHMENT_DESIGN.md)
  - 解释为什么 FIA 只看 `kernel_details.csv` 不够，为什么要引入 profiling bundle 和 runtime metadata。
- [`docs/perf_database/FIA_RUNTIME_METADATA_BACKFILL_GUIDE.md`](../../docs/perf_database/FIA_RUNTIME_METADATA_BACKFILL_GUIDE.md)
  - 解释 JSONL 埋点、回填流程，以及 `fill_fia_runtime_metadata.py` 的使用。
- [`docs/perf_database/FusedInferAttentionScore_CSV_COLUMN_GUIDE.md`](../../docs/perf_database/FusedInferAttentionScore_CSV_COLUMN_GUIDE.md)
  - 解释 `FusedInferAttentionScore.csv` 各列的物理含义。
- [`docs/perf_database/FusedInferAttentionScore_CSV_MAPPING.md`](../../docs/perf_database/FusedInferAttentionScore_CSV_MAPPING.md)
  - FIA 输入输出槽位和 CSV 列的对应关系。
- [`docs/perf_database/reports/COMM_BENCH_GUIDE.md`](../../docs/perf_database/reports/COMM_BENCH_GUIDE.md)
  - 解释通信 microbench 的 bench mode、采集策略和后处理规则。
- [`docs/perf_database/COMM_DATA_SPEC.md`](../../docs/perf_database/COMM_DATA_SPEC.md)
  - 通信 CSV 的最终格式和数据语义。
- [`docs/perf_database/METRICS_GUIDE.md`](../../docs/perf_database/METRICS_GUIDE.md)
  - 指标口径，尤其是 M6 这类离线指标怎么解释。

建议阅读顺序：

1. 先看总设计文档。
2. 再看 `GENERATE_SHAPE_GRID.md` 和 `COMM_BENCH_GUIDE.md`。
3. 涉及 FIA 时，再连续看 FIA 三份文档。

---

## 4. 目录结构

```text
tools/perf_data_collection/
├─ parse_kernel_details.py
├─ fill_fia_runtime_metadata.py
├─ fia_common.py
├─ start_microbench.py
├─ generate_shape_grid.py
├─ generate_comm_microbench.py
├─ run_comm_bench.sh
├─ build_comm_csv.py
├─ validate_comm_alignment.py
├─ compute_m6.py
└─ op_replay/
   ├─ common.py
   ├─ run_all_op.py
   └─ *_run.py
```

可以把它理解成三个子系统：

- 计算算子数据库生成与 replay：`parse_kernel_details.py`、`fill_fia_runtime_metadata.py`、`start_microbench.py`、`op_replay/`
- shape 扩样：`generate_shape_grid.py`
- 通信数据采集：`generate_comm_microbench.py`、`run_comm_bench.sh`、`build_comm_csv.py`、`validate_comm_alignment.py`

`compute_m6.py` 是最终质量评估脚本。

---

## 5. 顶层脚本逐个解释

### 5.1 [`parse_kernel_details.py`](./parse_kernel_details.py)

作用：

- 读取 `kernel_details*.csv` 或整个 profiling 目录。
- 按算子类型、输入输出 shape / dtype / format 聚合样本。
- 生成 perf database 里的各个算子 CSV。
- 对 FIA 额外补 runtime metadata 列框架。

它是“从真实 profiling 生成数据库”的第一步。

典型命令：

```powershell
py -3 tools/perf_data_collection/parse_kernel_details.py `
  --device ATLAS_800_A3_752T_128G_DIE `
  --vllm-ascend-version 0.20.0 `
  --kernel-details-path G:\path\to\profiling_dir
```

输入：

- `kernel_details.csv` 文件，或包含 profiling CSV 的目录

输出：

- `tensor_cast/performance_model/perf_database/data/{device}/vllm_ascend/{version}/` 下的各算子 CSV

和 FIA / shape 的关系：

- FIA 的初版样本就是它生成的。
- `generate_shape_grid.py` 后续扩样时，会把这些 CSV 当模板。

### 5.2 [`fill_fia_runtime_metadata.py`](./fill_fia_runtime_metadata.py)

作用：

- 把 `fia_runtime_metadata.jsonl` 回填到 `FusedInferAttentionScore.csv`。
- 根据 FIA signature 匹配行，补真实的 `actual_seq_lengths`、`actual_seq_lengths_kv`、`block_table_valid_blocks`、`avg_seq_len` 等列。

它解决的问题是：FIA replay 不能只靠 shape 猜。

典型命令：

```powershell
py -3 tools/perf_data_collection/fill_fia_runtime_metadata.py `
  --csv-path tensor_cast/performance_model/perf_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/v0.20.0/FusedInferAttentionScore.csv `
  --jsonl-path G:\path\to\fia_runtime_metadata.jsonl
```

如果不想覆盖原文件：

```powershell
py -3 tools/perf_data_collection/fill_fia_runtime_metadata.py `
  --csv-path ...\FusedInferAttentionScore.csv `
  --jsonl-path ...\fia_runtime_metadata.jsonl `
  --output-path ...\FusedInferAttentionScore.enriched.csv
```

### 5.3 [`fia_common.py`](./fia_common.py)

作用：

- FIA 相关公共解析函数。
- 提供 shape / runtime int / runtime int list / metadata field 的通用解析工具。

它不是独立入口脚本，一般不直接运行。

### 5.4 [`start_microbench.py`](./start_microbench.py)

作用：

- 驱动 `op_replay/run_all_op.py` 或指定 replay 脚本执行。
- 用 `msprof` 采集 replay 结果。
- 解析 `op_summary_*.csv`。
- 把 `Average Duration(us)`、`Profiling Average Duration(us)` 和一批硬件统计列写回数据库 CSV。

这是“把静态 CSV 变成可用于建模的 microbench 数据”的主入口。

典型命令：

```powershell
py -3 tools/perf_data_collection/start_microbench.py `
  --device ATLAS_800_A3_752T_128G_DIE `
  --vllm-ascend-version 0.15.0
```

只跑部分算子：

```powershell
py -3 tools/perf_data_collection/start_microbench.py `
  --device ATLAS_800_A3_752T_128G_DIE `
  --vllm-ascend-version 0.15.0 `
  --op MatMulV2 PadV3 FusedInferAttentionScore
```

直接解析已有 `PROF_*`：

```powershell
py -3 tools/perf_data_collection/start_microbench.py `
  --device ATLAS_800_A3_752T_128G_DIE `
  --vllm-ascend-version 0.15.0 `
  --prof-path G:\path\to\PROF_xxx
```

### 5.5 [`generate_shape_grid.py`](./generate_shape_grid.py)

作用：

- 遍历 perf database CSV。
- 基于已有真实样本，追加新的 shape 组合。
- 保留关键列，清零 latency/time/cycles 之类的数值列，等待后续 microbench 回填。

典型命令：

```powershell
py -3 tools/perf_data_collection/generate_shape_grid.py `
  --device ATLAS_800_A3_752T_128G_DIE `
  --vllm-ascend-version 0.15.0 `
  --rows 1000 `
  --seed 123
```

直接指定目录：

```powershell
py -3 tools/perf_data_collection/generate_shape_grid.py `
  --data-dir tensor_cast/performance_model/perf_database/data `
  --rows 1000
```

它和 FIA / replay 的关系：

- 生成的 shape 不是随便拼的，而是尽量保证 `op_replay/*_run.py` 还能消费。
- 对 FIA、MatMul、Norm、Rope 类算子都会保留结构约束。

### 5.6 [`generate_comm_microbench.py`](./generate_comm_microbench.py)

作用：

- 生成或直接运行 HCCL 通信算子 microbench。
- 支持 `all_reduce`、`all_gather`、`reduce_scatter`、`all_to_all`。
- 支持 `profiler`、`event`、`pipeline`、`kernel`、`alternating` 多种 bench mode。

推荐用途：

- 日常批量采集时，优先通过 `run_comm_bench.sh` 调它。
- 想自定义 grid / tier / bytes 时，再直接用它。

生成脚本示例：

```powershell
py -3 tools/perf_data_collection/generate_comm_microbench.py `
  --output-dir .\comm_scripts `
  --ops all_reduce all_gather reduce_scatter `
  --grid-shape 48 8 2 `
  --num-devices 16 `
  --topology-tier 2
```

直接跑示例：

```powershell
torchrun --nproc_per_node=16 tools/perf_data_collection/generate_comm_microbench.py `
  --do-run `
  --bench-mode alternating `
  --ops all_gather reduce_scatter `
  --grid-shape 48 8 2 `
  --num-devices 16 `
  --output-dir .\hccl_data
```

### 5.7 [`run_comm_bench.sh`](./run_comm_bench.sh)

作用：

- 把通信 microbench 的推荐采集流程固定下来。
- 自动分两轮采集：
  - `alternating` 采集 allReduce 全量、allGather/reduceScatter 的大消息
  - `kernel` 采集 allGather/reduceScatter 的小消息
- 内置标准 grid 和一部分生产 `msg_bytes`

典型命令：

```bash
bash tools/perf_data_collection/run_comm_bench.sh ./hccl_bench_data
```

输出：

- `./hccl_bench_data/alternating/`
- `./hccl_bench_data/kernel/`

### 5.8 [`build_comm_csv.py`](./build_comm_csv.py)

作用：

- 把 `alternating` 和 `kernel` 两轮数据合并成最终 HCCL CSV。
- 对小消息做固定开销修正。
- 对 DSV3 某些特殊点用 profiler P50 替换。
- 重算 `bandwidth_gbps`。

典型命令：

```powershell
py -3 tools/perf_data_collection/build_comm_csv.py `
  --alternating-dir .\hccl_bench_data\alternating `
  --kernel-dir .\hccl_bench_data\kernel `
  --profiler-trace-dir G:\path\to\dsv3_profiler `
  --output-dir .\hccl_final
```

### 5.9 [`validate_comm_alignment.py`](./validate_comm_alignment.py)

作用：

- 用解析模型 `CommAnalyticModel` 的公式口径，对 HCCL CSV 做 sanity check。
- 比较 measured vs analytic 的 ratio，输出 PASS / WARN / FAIL。

典型命令：

```powershell
py -3 tools/perf_data_collection/validate_comm_alignment.py `
  --csv-dir .\hccl_final `
  --tolerance 1.5 `
  --verbose
```

### 5.10 [`compute_m6.py`](./compute_m6.py)

作用：

- 计算离线指标 M6。
- 用 perf database 的命中算子经验值总和，去对比真实 profiling 中每次 forward 的平均 kernel 时长。

什么时候用：

- 评估数据库预测是否过高或过低。
- 做 phase 级别效果检查。

典型命令：

```powershell
py -3 tools/perf_data_collection/compute_m6.py `
  --tc-report results/qwen3_prefill_metrics.json `
  --profiler-output G:\path\to\ASCEND_PROFILER_OUTPUT `
  --delimiter ArgMaxV2
```

---

## 6. `op_replay` 是什么，干什么用的

`op_replay` 可以理解成“数据库样本执行器”。

这里的每个 `*_run.py` 都做一件事：

1. 读取某个算子的 CSV。
2. 从 `Input Shapes`、`Input Data Types`、`Input Formats` 里重建张量。
3. 必要时再根据 runtime metadata 或算子约束补齐标量参数。
4. 调用对应的 PyTorch / torch_npu / 自定义算子接口执行一次。
5. 让 `msprof` 采集到这次重放的执行时间和硬件统计。

它的意义不是功能正确性测试，而是“把数据库里的样本变成真实可执行 workload”，以便采集 microbench 数据。

### 6.1 入口脚本

- [`op_replay/common.py`](./op_replay/common.py)
  - 公共工具：设备/version 解析、CSV 迭代、dtype 映射、张量构造、FRACTAL_NZ 处理、NPU 运行时初始化。
  - 不单独运行。
- [`op_replay/run_all_op.py`](./op_replay/run_all_op.py)
  - 批量发现并运行当前目录下所有 `*_run.py`。
  - `start_microbench.py` 默认就是通过它做统一 replay。

批量运行命令：

```powershell
py -3 tools/perf_data_collection/op_replay/run_all_op.py `
  --device ATLAS_800_A3_752T_128G_DIE `
  --vllm-ascend-version 0.15.0
```

只跑部分算子：

```powershell
py -3 tools/perf_data_collection/op_replay/run_all_op.py `
  --device ATLAS_800_A3_752T_128G_DIE `
  --vllm-ascend-version 0.15.0 `
  --op MatMulV2 RmsNorm FusedInferAttentionScore
```

### 6.2 每个 replay 脚本有什么用

下面这些脚本默认命令格式基本一致：

```powershell
py -3 tools/perf_data_collection/op_replay/<SCRIPT_NAME> `
  --device ATLAS_800_A3_752T_128G_DIE `
  --vllm-ascend-version 0.15.0
```

如果某个脚本没有额外参数，通常都可以直接套这个模板。

| 文件 | 作用 | 常用命令 |
|---|---|---|
| `Add_run.py` | 重放 `Add`。适合 elementwise binary 样本。 | `py -3 tools/perf_data_collection/op_replay/Add_run.py --device ... --vllm-ascend-version ...` |
| `AddRmsNormBias_run.py` | 重放融合 `Add + RmsNorm + Bias`。 | `py -3 tools/perf_data_collection/op_replay/AddRmsNormBias_run.py --device ... --vllm-ascend-version ...` |
| `ArgMaxV2_run.py` | 重放 `ArgMaxV2`。这个算子也常被 `compute_m6.py` 当作 forward delimiter。 | `py -3 tools/perf_data_collection/op_replay/ArgMaxV2_run.py --device ... --vllm-ascend-version ...` |
| `AscendQuantV2_run.py` | 重放 `AscendQuantV2` 量化路径。 | `py -3 tools/perf_data_collection/op_replay/AscendQuantV2_run.py --device ... --vllm-ascend-version ...` |
| `DispatchFFNCombine_run.py` | 重放 MoE/FFN 路径里的 `DispatchFFNCombine`。 | `py -3 tools/perf_data_collection/op_replay/DispatchFFNCombine_run.py --device ... --vllm-ascend-version ...` |
| `DynamicQuant_run.py` | 重放 `DynamicQuant`。 | `py -3 tools/perf_data_collection/op_replay/DynamicQuant_run.py --device ... --vllm-ascend-version ...` |
| `FusedInferAttentionScore_run.py` | 重放 FIA。会利用 CSV 里的 runtime metadata 恢复 paged / MLA / non-paged 场景，是 FIA replay 核心脚本。 | `py -3 tools/perf_data_collection/op_replay/FusedInferAttentionScore_run.py --device ... --vllm-ascend-version ...` |
| `GatherV2_run.py` | 重放 `GatherV2`。 | `py -3 tools/perf_data_collection/op_replay/GatherV2_run.py --device ... --vllm-ascend-version ...` |
| `InterleaveRope_run.py` | 重放 `InterleaveRope`。常见于 rope 相关链路。 | `py -3 tools/perf_data_collection/op_replay/InterleaveRope_run.py --device ... --vllm-ascend-version ...` |
| `KvRmsNormRopeCache_run.py` | 重放 KV/RMSNorm/Rope/Cache 融合算子。 | `py -3 tools/perf_data_collection/op_replay/KvRmsNormRopeCache_run.py --device ... --vllm-ascend-version ...` |
| `MaskedFill_run.py` | 重放 `MaskedFill`。 | `py -3 tools/perf_data_collection/op_replay/MaskedFill_run.py --device ... --vllm-ascend-version ...` |
| `MatMulCommon_run.py` | 重放 `MatMulCommon`。 | `py -3 tools/perf_data_collection/op_replay/MatMulCommon_run.py --device ... --vllm-ascend-version ...` |
| `MatMulV2_run.py` | 重放 `MatMulV2`，内部按 CSV 重建输入并执行 `torch.mm()`。 | `py -3 tools/perf_data_collection/op_replay/MatMulV2_run.py --device ... --vllm-ascend-version ...` |
| `MatMulV3_run.py` | 重放 `MatMulV3`。 | `py -3 tools/perf_data_collection/op_replay/MatMulV3_run.py --device ... --vllm-ascend-version ...` |
| `PadV3_run.py` | 重放 `PadV3`。 | `py -3 tools/perf_data_collection/op_replay/PadV3_run.py --device ... --vllm-ascend-version ...` |
| `QuantBatchMatmulV3_run.py` | 重放量化 batch matmul。 | `py -3 tools/perf_data_collection/op_replay/QuantBatchMatmulV3_run.py --device ... --vllm-ascend-version ...` |
| `ReshapeAndCacheNdKernel_run.py` | 重放 reshape + cache 类 kernel。 | `py -3 tools/perf_data_collection/op_replay/ReshapeAndCacheNdKernel_run.py --device ... --vllm-ascend-version ...` |
| `RINGMLAPrefillBF16Kernel_run.py` | 重放 MLA prefill BF16 ring kernel。 | `py -3 tools/perf_data_collection/op_replay/RINGMLAPrefillBF16Kernel_run.py --device ... --vllm-ascend-version ...` |
| `RmsNorm_run.py` | 重放 `RmsNorm`，内部调用 `torch_npu.npu_rms_norm()`。 | `py -3 tools/perf_data_collection/op_replay/RmsNorm_run.py --device ... --vllm-ascend-version ...` |
| `Slice_run.py` | 重放 `Slice`。 | `py -3 tools/perf_data_collection/op_replay/Slice_run.py --device ... --vllm-ascend-version ...` |
| `SoftmaxV2_run.py` | 重放 `SoftmaxV2`。 | `py -3 tools/perf_data_collection/op_replay/SoftmaxV2_run.py --device ... --vllm-ascend-version ...` |
| `Sort_run.py` | 重放 `Sort`。 | `py -3 tools/perf_data_collection/op_replay/Sort_run.py --device ... --vllm-ascend-version ...` |
| `split_qkv_rmsnorm_rope_kernel_run.py` | 重放 `split_qkv_rmsnorm_rope_kernel` 融合路径。 | `py -3 tools/perf_data_collection/op_replay/split_qkv_rmsnorm_rope_kernel_run.py --device ... --vllm-ascend-version ...` |
| `SwiGlu_run.py` | 重放 `SwiGlu`。 | `py -3 tools/perf_data_collection/op_replay/SwiGlu_run.py --device ... --vllm-ascend-version ...` |
| `TensorMove_run.py` | 重放 `TensorMove`。 | `py -3 tools/perf_data_collection/op_replay/TensorMove_run.py --device ... --vllm-ascend-version ...` |
| `Transpose_run.py` | 重放 `Transpose`。 | `py -3 tools/perf_data_collection/op_replay/Transpose_run.py --device ... --vllm-ascend-version ...` |

重点理解两类：

- FIA 类：`FusedInferAttentionScore_run.py`
- shape 约束强的类：`MatMulV2_run.py`、`MatMulV3_run.py`、`RmsNorm_run.py`、`QuantBatchMatmulV3_run.py`

---

## 7. FIA、shape 生成、replay 三者是怎么串起来的

### 7.1 为什么 FIA 是单独一条线

FIA 的性能高度依赖真实运行态，而不仅是静态 shape。比如：

- 同样的 `query/key/value` shape，不同 `actual_seq_lengths_kv` 会走不同 cache 使用状态。
- 同样是 attention，`input_layout=TND` 和 `BNSD_NBSD` 的重放方式不一样。
- paged 路径是否有 `block_table`，会决定 replay 走哪条分支。

所以 README 里最重要的结论是：

`FusedInferAttentionScore.csv` 不是普通算子 CSV，它是“shape 签名 + 运行时补充列”的混合体。

### 7.2 shape 生成为什么要参考 `op_replay`

`generate_shape_grid.py` 的目标不是把数据库变大，而是把数据库变成“可 replay、可插值、对模型真实场景更友好”的数据库。

因此它会尽量保持：

- elementwise 同形关系
- matmul 的 `M/K/N` contract
- norm 的 hidden-size 约束
- rope/head_dim 约束
- FIA 31 个输入槽位里关键位置的结构关系

这也是为什么它会把 `op_replay` 目录加入 `sys.path` 并复用部分公共逻辑。shape 生成和 replay 在这里不是两套独立系统，而是一前一后的上下游。

### 7.3 推荐 FIA 工作流

```text
profiling bundle
  -> parse_kernel_details.py
  -> FusedInferAttentionScore.csv（初版）
  -> fill_fia_runtime_metadata.py（如有 JSONL）
  -> FusedInferAttentionScore_run.py
  -> start_microbench.py 回写 Average Duration(us)
```

如果没有 runtime metadata：

- FIA 仍然可以 replay。
- 但它更依赖推断逻辑，和真实整网状态的偏差通常更大。

---

## 8. 对新手最实用的几条命令

### 8.1 从 profiling 生成数据库

```powershell
py -3 tools/perf_data_collection/parse_kernel_details.py `
  --device ATLAS_800_A3_752T_128G_DIE `
  --vllm-ascend-version 0.20.0 `
  --kernel-details-path G:\path\to\profiling_dir
```

### 8.2 补 FIA 运行时元数据

```powershell
py -3 tools/perf_data_collection/fill_fia_runtime_metadata.py `
  --csv-path tensor_cast/performance_model/perf_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/v0.20.0/FusedInferAttentionScore.csv `
  --jsonl-path G:\path\to\fia_runtime_metadata.jsonl
```

### 8.3 跑全部 replay 并回写 microbench

```powershell
py -3 tools/perf_data_collection/start_microbench.py `
  --device ATLAS_800_A3_752T_128G_DIE `
  --vllm-ascend-version 0.20.0
```

### 8.4 只验证某个算子 replay 是否能跑

```powershell
py -3 tools/perf_data_collection/op_replay/FusedInferAttentionScore_run.py `
  --device ATLAS_800_A3_752T_128G_DIE `
  --vllm-ascend-version 0.20.0
```

### 8.5 给数据库扩 shape

```powershell
py -3 tools/perf_data_collection/generate_shape_grid.py `
  --device ATLAS_800_A3_752T_128G_DIE `
  --vllm-ascend-version 0.20.0 `
  --rows 1000 `
  --seed 123
```

### 8.6 采通信数据

```bash
bash tools/perf_data_collection/run_comm_bench.sh ./hccl_bench_data
```

然后：

```powershell
py -3 tools/perf_data_collection/build_comm_csv.py `
  --alternating-dir .\hccl_bench_data\alternating `
  --kernel-dir .\hccl_bench_data\kernel `
  --profiler-trace-dir G:\path\to\dsv3_profiler `
  --output-dir .\hccl_final
```

### 8.7 算 M6

```powershell
py -3 tools/perf_data_collection/compute_m6.py `
  --tc-report results/qwen3_prefill_metrics.json `
  --profiler-output G:\path\to\ASCEND_PROFILER_OUTPUT
```

---

## 9. 新手最容易混淆的几点

### 9.1 `parse_kernel_details.py` 和 `start_microbench.py` 不是一回事

- `parse_kernel_details.py` 负责“从整网 profiling 生成静态数据库样本”。
- `start_microbench.py` 负责“把数据库样本 replay 成单算子 workload，再采 microbench 数据回写”。

前者更像“建库”，后者更像“补齐经验时延”。

### 9.2 `generate_shape_grid.py` 不会直接得到真实耗时

它只会：

- 追加新行
- 保留结构列
- 把时间类列清零

真正的耗时需要后面通过 `start_microbench.py` 回填。

### 9.3 FIA 的 `Average Duration(us)` 不一定天然接近 `Profiling Average Duration(us)`

如果 FIA 缺 runtime metadata，replay 会更依赖推断，偏差通常更大。这也是为什么仓库里专门有 FIA enrichment 和 backfill 设计文档。

### 9.4 `op_replay` 的目标不是功能测试，而是“可被 profiler 采样的 workload 重放”

因此很多脚本关注的是：

- 张量能不能构出来
- API 参数能不能匹配
- kernel 路径能不能尽量贴近真实场景

而不是单元测试式的数值校验。

---

## 10. 一个最小可执行上手流程

如果你第一次接触这个目录，建议只做下面 4 步：

1. 用 `parse_kernel_details.py` 对一个 profiling 目录建库。
2. 单独运行一次 `op_replay/FusedInferAttentionScore_run.py` 或 `MatMulV2_run.py`，确认 replay 能通。
3. 用 `start_microbench.py --op ...` 只跑少量算子，确认能回写 `Average Duration(us)`。
4. 再考虑是否需要 `generate_shape_grid.py` 扩样，或者通信链路的 `run_comm_bench.sh`。

这样你会先理解“数据库样本从哪里来”，再理解“为什么还要 replay”，最后再理解“为什么还要扩 shape 和做离线指标校验”。

---

## 11. 总结

对新人来说，`tools/perf_data_collection` 最重要的理解不是每个脚本怎么跑，而是这三件事：

1. 它是 perf database 的数据生产与校验工具链，不是单个脚本集合。
2. `op_replay` 是数据库样本和 microbench 之间的桥。
3. FIA 和 shape grid 是这里最关键的两条增强主线：前者解决 runtime 还原问题，后者解决样本稀疏问题。

如果只记住一句话：

这个目录的职责就是把“整网 profiling 里看到的算子行为”沉淀成“可查询、可 replay、可扩样、可校验”的性能数据库。
