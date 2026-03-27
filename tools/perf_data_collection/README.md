# `tools/perf_data_collection` README

## 1. 这个目录在做什么

`tools/perf_data_collection` 是 perf database 的数据生产、补全、回放和校验工具集。

它主要负责 4 件事：

1. 从整网 profiling 结果里抽取算子样本，生成数据库 CSV。
2. 为 FIA 这类仅靠 `kernel_details.csv` 无法完全还原的算子补 runtime metadata。
3. 基于数据库里的 shape / dtype / format 回放单算子 workload，采集 microbench 数据并回写 CSV。
4. 在样本稀疏时按约束扩充 shape grid，供后续 replay 和建模使用。

一句话理解：

`docs/perf_database` 解释为什么这样设计，`tools/perf_data_collection` 负责把这些设计真正落成可用的数据、脚本和工作流。

## 2. 新手先建立整体心智模型

### 2.1 计算算子主线

最常见流程是：

1. 跑整网 profiling，拿到 `kernel_details*.csv` 或完整 profiling 目录。
2. 用 [`parse_kernel_details.py`](./parse_kernel_details.py) 把 profiling 聚合成按算子拆分的数据库 CSV。
3. 如果是 FIA，再用 [`fill_fia_runtime_metadata.py`](./fill_fia_runtime_metadata.py) 把运行时 JSONL 回填进 `FusedInferAttentionScore.csv`。
4. 用 [`start_microbench.py`](./start_microbench.py) 驱动 [`op_replay`](./op_replay) 里的 replay 脚本，在 NPU 上重放这些算子。
5. 解析 `msprof` 产物，把 `Average Duration(us)` 和各类 profiling 统计列回写到数据库 CSV。
6. 如果样本太稀疏，再用 [`generate_shape_grid.py`](./generate_shape_grid.py) 扩样。

### 2.2 FIA 主线

FIA 是这个目录里最特殊的一类算子，因为它不只依赖 `Input Shapes`，还依赖：

- `actual_seq_lengths`
- `actual_seq_lengths_kv`
- `block_table`
- `num_heads`
- `num_key_value_heads`
- `input_layout`
- `sparse_mode`
- `block_size`

所以 FIA 的完整流程通常是：

1. `parse_kernel_details.py` 先生成初版 `FusedInferAttentionScore.csv`。
2. 如果 profiling bundle 里 runtime 信息不够，就在 `vllm-ascend` 的 FIA 调用点埋点，导出 `fia_runtime_metadata.jsonl`。
3. 用 `fill_fia_runtime_metadata.py` 把 JSONL 里的真实 runtime metadata 回填到 CSV。
4. `op_replay/FusedInferAttentionScore_run.py` 重放时优先消费这些 runtime 列，而不是只按 shape 猜。

### 2.3 shape grid 主线

`generate_shape_grid.py` 不是“凭空造随机 shape”，而是“基于已有 CSV 模板做约束感知扩样”：

- 保留原始真实数据。
- 维持算子内部约束，比如 matmul contract、norm hidden size、rope/head_dim 关系、FIA 若干槽位关系。
- 让新生成的样本尽量仍然能被 `op_replay/*_run.py` 正常读取和执行。

## 3. 数据库路径规则

当前工具支持两种数据库定位方式。

### 3.1 直接传数据库目录

最稳妥的方式是传 `--database-path`，工具会直接读写这个目录，不再自动拼版本目录名。

例如：

```powershell
py -3 tools/perf_data_collection/start_microbench.py `
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5
```

适合：

- 已经有现成数据库目录。
- 目录名不想交给工具推导。
- 需要精确读写指定版本库。

### 3.2 按约定格式自动推导数据库目录

不传 `--database-path` 时，工具会使用：

`tensor_cast/performance_model/profiling_database/data/{device}/vllm_ascend/{version_dir}`

其中：

- `device` 默认是 `ATLAS_800_A3_752T_128G_DIE`
- `version_dir` 约定格式是 `vllm{vllm_version}_torch{torch_version}_cann{cann_version}`

例如：

- `vllm0.13.0_torch2.8.0_cann8.3`
- `vllm0.15.0_torch2.9.0_cann8.5`
- `vllm0.18.0_torch2.9.0_cann8.5`

这时有两种传法：

- `--vllm-version` 直接传完整目录名
- 或分别传 `--vllm-version`、`--torch-version`、`--cann-version`

如果没有把版本参数传全，工具会尝试自动扫描当前环境里的：

- `vllm-ascend`
- `torch`
- `CANN`

如果扫不全，会提示你：

- 改传 `--database-path`
- 或手动补齐版本参数

### 3.3 CANN 自动探测规则

当前 CANN 版本自动探测优先查这些位置：

1. `${HOME}/Ascend/cann/`
2. `/usr/local/Ascend/cann/`

并会尝试读取：

- `ascend_toolkit_install.info`
- `arm64-linux/ascend_toolkit_install.info`
- `x86_64-linux/ascend_toolkit_install.info`

例如：

```text
package_name=Ascend-cann-toolkit
version=8.5.0
path=/usr/local/Ascend/cann-8.5.0
```

工具会从中解析出 `8.5.0`。

## 4. 相关设计文档怎么读

建议优先看这些文档：

- [`docs/perf_database/OPERATOR_PERF_DATABASE_DESIGN_zh_v1.5.md`](../../docs/perf_database/OPERATOR_PERF_DATABASE_DESIGN_zh_v1.5.md)
- [`docs/perf_database/GENERATE_SHAPE_GRID.md`](../../docs/perf_database/GENERATE_SHAPE_GRID.md)
- [`docs/perf_database/FIA_PROFILING_BUNDLE_ENRICHMENT_DESIGN.md`](../../docs/perf_database/FIA_PROFILING_BUNDLE_ENRICHMENT_DESIGN.md)
- [`docs/perf_database/FIA_RUNTIME_METADATA_BACKFILL_GUIDE.md`](../../docs/perf_database/FIA_RUNTIME_METADATA_BACKFILL_GUIDE.md)
- [`docs/perf_database/FusedInferAttentionScore_CSV_COLUMN_GUIDE.md`](../../docs/perf_database/FusedInferAttentionScore_CSV_COLUMN_GUIDE.md)
- [`docs/perf_database/FusedInferAttentionScore_CSV_MAPPING.md`](../../docs/perf_database/FusedInferAttentionScore_CSV_MAPPING.md)
- [`docs/perf_database/reports/COMM_BENCH_GUIDE.md`](../../docs/perf_database/reports/COMM_BENCH_GUIDE.md)
- [`docs/perf_database/COMM_DATA_SPEC.md`](../../docs/perf_database/COMM_DATA_SPEC.md)
- [`docs/perf_database/METRICS_GUIDE.md`](../../docs/perf_database/METRICS_GUIDE.md)

推荐顺序：

1. 先看总设计。
2. 再看 shape grid 和 comm bench。
3. 涉及 FIA 时，再看 FIA 相关文档。

## 5. 目录结构

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
   ├─ replay_framework.py
   ├─ run_all_op.py
   └─ *_run.py
```

可以把它理解成 3 个子系统：

- 计算算子数据库生成与 replay：
  `parse_kernel_details.py`、`fill_fia_runtime_metadata.py`、`start_microbench.py`、`op_replay/`
- shape 扩样：
  `generate_shape_grid.py`
- 通信数据采集：
  `generate_comm_microbench.py`、`run_comm_bench.sh`、`build_comm_csv.py`、`validate_comm_alignment.py`

`compute_m6.py` 用于离线质量评估。

## 6. 顶层脚本逐个解释

### 6.1 [`parse_kernel_details.py`](./parse_kernel_details.py)

作用：

- 读取 `kernel_details*.csv` 或完整 profiling 目录。
- 按算子类型、输入输出 shape / dtype / format 聚合样本。
- 生成数据库里的各个算子 CSV。
- 为 FIA 额外补一批 runtime metadata 推断列。

关键参数：

- 必选：
  - `--profiling-path`
- 二选一：
  - `--database-path`
  - 或 `--device` + `--vllm-version`，并可选补 `--torch-version` / `--cann-version`

典型命令：

```powershell
py -3 tools/perf_data_collection/parse_kernel_details.py `
  --profiling-path G:\path\to\profiling_dir `
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.20.0_torch2.9.0_cann8.5
```

或：

```powershell
py -3 tools/perf_data_collection/parse_kernel_details.py `
  --profiling-path G:\path\to\profiling_dir `
  --device ATLAS_800_A3_752T_128G_DIE `
  --vllm-version 0.20.0 `
  --torch-version 2.9.0 `
  --cann-version 8.5
```

### 6.2 [`fill_fia_runtime_metadata.py`](./fill_fia_runtime_metadata.py)

作用：

- 把 `fia_runtime_metadata.jsonl` 回填到 `FusedInferAttentionScore.csv`
- 根据 FIA signature 补真实的 `actual_seq_lengths`、`actual_seq_lengths_kv`、`block_table_valid_blocks` 等列

典型命令：

```powershell
py -3 tools/perf_data_collection/fill_fia_runtime_metadata.py `
  --csv-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.20.0_torch2.9.0_cann8.5/FusedInferAttentionScore.csv `
  --jsonl-path G:\path\to\fia_runtime_metadata.jsonl
```

如果不想覆盖原 CSV，可以额外传：

```powershell
  --output-path G:\path\to\FusedInferAttentionScore.out.csv
```

### 6.3 [`start_microbench.py`](./start_microbench.py)

作用：

- 驱动 `op_replay/run_all_op.py` 跑 replay
- 用 `msprof` 采集 replay 结果
- 解析 `op_summary_*.csv`
- 回写 `Average Duration(us)`、`Profiling Average Duration(us)` 以及各类 profiling 统计列

当前时延策略：

- replay 默认会重复执行
- `Average Duration(us)` 来自 replay 的 `op_summary_*.csv`
- 对同一签名，取多次 replay 中最小的 `Task Duration(us)` 作为最终 `Average Duration(us)`

典型命令：

```powershell
py -3 tools/perf_data_collection/start_microbench.py `
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5
```

只跑部分算子：

```powershell
py -3 tools/perf_data_collection/start_microbench.py `
  --device ATLAS_800_A3_752T_128G_DIE `
  --vllm-version 0.15.0 `
  --torch-version 2.9.0 `
  --cann-version 8.5 `
  --op MatMulV2 PadV3 FusedInferAttentionScore
```

如果包含 `DispatchFFNCombine`，建议把 EP 大小显式传给统一回填链路：

```powershell
py -3 tools/perf_data_collection/start_microbench.py `
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.20.0_torch2.9.0_cann8.5 `
  --op DispatchFFNCombine `
  --dispatch-ffn-combine-ep-size 16
```

只解析已有 profiling 目录，不重新跑 replay：

```powershell
py -3 tools/perf_data_collection/start_microbench.py `
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5 `
  --prof-path G:\path\to\PROF_xxx
```

### 6.4 [`generate_shape_grid.py`](./generate_shape_grid.py)

作用：

- 遍历数据库 CSV
- 基于已有真实样本追加新的 shape 组合
- 保留关键列，清空时间相关列，等待后续 microbench 回填

注意：

- 这个脚本当前用的是 `--data-dir`
- 不是 `--database-path`

典型命令：

```powershell
py -3 tools/perf_data_collection/generate_shape_grid.py `
  --data-dir tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5 `
  --rows 1000 `
  --seed 123
```

或者让它按版本推导目录：

```powershell
py -3 tools/perf_data_collection/generate_shape_grid.py `
  --device ATLAS_800_A3_752T_128G_DIE `
  --vllm-version 0.15.0 `
  --torch-version 2.9.0 `
  --cann-version 8.5 `
  --rows 1000 `
  --seed 123
```

### 6.5 通信相关脚本

- [`generate_comm_microbench.py`](./generate_comm_microbench.py)
  - 生成或直接运行 HCCL microbench
- [`run_comm_bench.sh`](./run_comm_bench.sh)
  - 固定通信 microbench 推荐采集流程
- [`build_comm_csv.py`](./build_comm_csv.py)
  - 把多轮通信数据合并成最终 CSV
- [`validate_comm_alignment.py`](./validate_comm_alignment.py)
  - 用解析模型做 sanity check
- [`compute_m6.py`](./compute_m6.py)
  - 计算离线指标 M6

## 7. `op_replay` 是什么

`op_replay` 可以理解成“数据库样本执行器”。

每个 `*_run.py` 都做几件事：

1. 读取某个算子的 CSV。
2. 从 `Input Shapes`、`Input Data Types`、`Input Formats` 里重建张量。
3. 必要时根据 runtime metadata 或算子约束补齐标量参数。
4. 调用对应的 PyTorch / `torch_npu` / 自定义算子接口执行。
5. 让 `msprof` 采到这次重放的执行时间和硬件统计。

它的目标不是功能测试，而是把数据库样本变成真实可执行 workload，用于 microbench 采样。

### 7.1 入口脚本

- [`op_replay/common.py`](./op_replay/common.py)
  - 公共工具：数据库路径解析、设备和版本探测、CSV 迭代、dtype 映射、张量构造、NPU runtime 初始化
- [`op_replay/replay_framework.py`](./op_replay/replay_framework.py)
  - 共享运行时框架：统一 argparse、database path、repeat 执行、CSV 遍历、tensor 构造、同步和 success log
- [`op_replay/run_all_op.py`](./op_replay/run_all_op.py)
  - 批量发现并运行当前目录下所有 `*_run.py`

批量运行命令：

```powershell
py -3 tools/perf_data_collection/op_replay/run_all_op.py `
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5
```

只跑部分算子：

```powershell
py -3 tools/perf_data_collection/op_replay/run_all_op.py `
  --device ATLAS_800_A3_752T_128G_DIE `
  --vllm-version 0.15.0 `
  --op MatMulV2 PadV3 FusedInferAttentionScore
```

如果批量运行里包含 `DispatchFFNCombine`，可以额外传：

```powershell
  --dispatch-ffn-combine-ep-size 16
```

### 7.2 replay repeat 策略

当前默认 repeat 策略是：

- 通用 `*_run.py` 默认重复次数：`30`
- 公共环境变量：`MSMODELING_OP_REPLAY_REPEAT_COUNT`
- 通用 CLI 参数：`--repeat-count`

FIA 仍保留一层专用兼容逻辑：

- 专用环境变量：`MSMODELING_FIA_REPLAY_REPEAT_COUNT`
- 如果没设，会回退到公共 repeat 策略

### 7.3 FIA replay 的特殊点

[`FusedInferAttentionScore_run.py`](./op_replay/FusedInferAttentionScore_run.py) 是手写重脚本，不走通用框架，原因是它需要：

- 处理 31 个输入槽位
- 重建 `actual_seq_lengths` / `actual_seq_lengths_kv`
- 根据 `block_table_valid_blocks` 重建 paged block table
- 区分 `TND`、`BNSD_NBSD`、paged / non-paged、MLA / 非 MLA 路径

典型命令：

```powershell
py -3 tools/perf_data_collection/op_replay/FusedInferAttentionScore_run.py `
  --device ATLAS_800_A3_752T_128G_DIE `
  --vllm-version 0.20.0 `
  --torch-version 2.9.0 `
  --cann-version 8.5 `
  --repeat-count 30
```

更稳妥的方式仍然是直接传库路径：

```powershell
py -3 tools/perf_data_collection/op_replay/FusedInferAttentionScore_run.py `
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.20.0_torch2.9.0_cann8.5 `
  --repeat-count 30
```

### 7.4 replay framework 的定位

`replay_framework.py` 适合这类脚本：

- 主体逻辑简单
- 只需要少量 `build_case` / `run_case` / `format_success` / `prepare` hook
- 不需要复杂 runtime 推断或分布式初始化

当前已接入框架的脚本包括：

- `Add_run.py`
- `AddRmsNormBias_run.py`
- `ArgMaxV2_run.py`
- `AscendQuantV2_run.py`
- `DynamicQuant_run.py`
- `GatherV2_run.py`
- `InterleaveRope_run.py`
- `KvRmsNormRopeCache_run.py`
- `MaskedFill_run.py`
- `MatMulCommon_run.py`
- `MatMulV2_run.py`
- `MatMulV3_run.py`
- `PadV3_run.py`
- `ReshapeAndCacheNdKernel_run.py`
- `RmsNorm_run.py`
- `Slice_run.py`
- `SoftmaxV2_run.py`
- `Sort_run.py`
- `split_qkv_rmsnorm_rope_kernel_run.py`
- `SwiGlu_run.py`
- `TensorMove_run.py`
- `Transpose_run.py`

仍保留手写实现的重型脚本主要是：

- `FusedInferAttentionScore_run.py`
- `QuantBatchMatmulV3_run.py`
- `RINGMLAPrefillBF16Kernel_run.py`
- `DispatchFFNCombine_run.py`

其中 `DispatchFFNCombine_run.py` 现在已经切到统一 replay / `start_microbench` 回填链路，不再自带 `--output-csv` 和内部 profiler 落盘逻辑；它仍然保留 EP / torchrun 特化启动。

## 8. FIA、shape 生成、replay 三者的关系

### 8.1 为什么 FIA 是单独一条线

FIA 的性能高度依赖真实运行态，而不只是静态 shape。例如：

- 同样的 `query/key/value` shape，不同 `actual_seq_lengths_kv` 会走不同 cache 使用状态
- 同样是 attention，`input_layout=TND` 和 `BNSD_NBSD` 的重放方式不同
- paged 路径有没有 `block_table` 会决定 replay 走哪条分支

所以 `FusedInferAttentionScore.csv` 不是普通算子 CSV，而是“shape 签名 + runtime metadata”的混合体。

### 8.2 为什么 shape 生成要参考 `op_replay`

`generate_shape_grid.py` 的目标不是单纯把数据库变大，而是把数据库变成“可 replay、可插值、对模型真实场景更友好”的数据库，所以它要尽量保证新样本仍能被 replay 脚本消费。

## 9. 对新手最实用的几条命令

### 9.1 从 profiling 建库

```powershell
py -3 tools/perf_data_collection/parse_kernel_details.py `
  --profiling-path G:\path\to\profiling_dir `
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.20.0_torch2.9.0_cann8.5
```

### 9.2 补 FIA runtime metadata

```powershell
py -3 tools/perf_data_collection/fill_fia_runtime_metadata.py `
  --csv-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.20.0_torch2.9.0_cann8.5/FusedInferAttentionScore.csv `
  --jsonl-path G:\path\to\fia_runtime_metadata.jsonl
```

### 9.3 跑整套 replay 并回填 microbench

```powershell
py -3 tools/perf_data_collection/start_microbench.py `
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.20.0_torch2.9.0_cann8.5
```

### 9.4 只验证某个算子 replay 是否能跑

```powershell
py -3 tools/perf_data_collection/op_replay/FusedInferAttentionScore_run.py `
  --database-path tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.20.0_torch2.9.0_cann8.5 `
  --repeat-count 30
```

## 10. 新手最容易混淆的几点

- `parse_kernel_details.py` 负责“建库”，`start_microbench.py` 负责“重放并补时延”，两者不是一回事。
- `generate_shape_grid.py` 只会追加样本并清零时间列，不会直接生成真实耗时。
- FIA 的 `Average Duration(us)` 不一定天然接近 `Profiling Average Duration(us)`，缺 runtime metadata 时偏差通常会更大。
- `op_replay` 的目标不是功能测试，而是“可被 profiler 采样的 workload 重放”。

## 11. 一个最小可执行上手流程

如果你第一次接触这个目录，建议只做 4 步：

1. 用 `parse_kernel_details.py` 对一个 profiling 目录建库。
2. 单独运行一次 `op_replay/FusedInferAttentionScore_run.py` 或 `MatMulV2_run.py`，确认 replay 能跑通。
3. 用 `start_microbench.py --op ...` 只跑少量算子，确认能回写 `Average Duration(us)`。
4. 再决定是否需要 `generate_shape_grid.py` 扩样，或者跑通信链路的 `run_comm_bench.sh`。

## 12. 总结

对新人来说，最重要的不是把每个脚本参数背下来，而是记住三件事：

1. 这里是 perf database 的数据生产与校验工具链。
2. `op_replay` 是数据库样本和 microbench 之间的桥。
3. FIA 和 shape grid 是这里最关键的两条增强主线。

如果只记住一句话：

这个目录的职责，就是把“整网 profiling 里看到的算子行为”沉淀成“可查询、可 replay、可扩样、可校验”的性能数据库。
