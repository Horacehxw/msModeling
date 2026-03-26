# Op Mapping 教程 v2

> 如何创建和维护 `op_mapping.yaml` —— TensorCast 仿真与 NPU profiling 数据之间的桥梁。

## 1. op_mapping.yaml 是什么？

`op_mapping.yaml` 将每个 TensorCast (TC) 虚拟算子映射到真实设备 profiling 数据中的 NPU 内核类型（kernel type）。这个映射使得 `EmpiricalPerformanceModel` 能够查找真实 profiling 延迟，而不是使用解析估算。

**文件位置：** `tensor_cast/performance_model/profiling_database/data/{device}/vllm_ascend/{version}/op_mapping.yaml`

**版本命名规范：** `{version}` 编码了用于 profiling 的软件栈版本（例如 `v0.13.0` 或 `vllm0.13.0_torch2.8.0_cann8.3`）

### 文件结构

```yaml
version: "<vllm_ascend_version>"
device: <DEVICE>
cann_version: "<cann_version>"

interpolation_policy:
  default_method: linear
  kernel_overrides:
    FusedInferAttentionScore:
      shape_transform: sqrt    # O(seq²) 算子在 sqrt 空间中插值

operator_mappings:
  "aten.mm.default":
    kernel_type: MatMulV2              # Profiling Type 列 = CSV 文件名
    notes: "[HIGH] Path A. op-plugin: ..."

  "tensor_cast.matmul_all_reduce.default":
    composite: true                     # 分解为多个内核
    sub_kernels: [MatMulV2, hcom_allReduce_]

  "aten.view.default":
    zero_cost: true                     # 纯元数据操作，无 NPU 执行

torch_npu_reference:
  MatMulV2:
    apis: [torch.mm, torch.matmul]
    aclnn: [aclnnMatmul, aclnnMatmulWeightNz]
```

### 条目类型（互斥）

| 类型 | 字段 | 含义 | 示例 |
|------|------|------|------|
| **计算** | `kernel_type: X` | 通过 Type=X 直接查询 CSV | MatMulV2, SwiGlu |
| **复合** | `composite: true` | 分解为 `sub_kernels`，分别查询 | matmul_all_reduce → [MatMulV2, hcom_allReduce_] |
| **零开销** | `zero_cost: true` | 返回 latency=0（纯元数据算子） | view, permute, split |

每个条目必须恰好具有以上三种类型之一。

### 可选字段

- `alternate_kernel_types: [Type1, Type2]` —— 主类型未命中时的备选 CSV 类型
- `category: communication` —— 触发 message_bytes+num_devices 查询而非 shape 匹配
- `query_mode: attention_special` —— 触发 (batch, seq, heads, head_dim) 匹配
- `query_mode: elementwise` —— 触发输出形状匹配 + dtype 松弛缩放（逐元素算子）
- `notes: "..."` —— 包含置信度级别的证据链

#### 逐元素算子 (Elementwise Ops)

对于内存带宽受限的逐元素算子 (`aten.add.Tensor`, `aten.mul.Tensor`, `aten.div.Tensor`),
使用 `query_mode: elementwise` 代替默认的输入形状匹配:

```yaml
"aten.add.Tensor":
  kernel_type: Add
  query_mode: elementwise
```

此模式按**输出形状**匹配 CSV,并支持 dtype 松弛匹配 (FP32 → BF16 × 2.0 字节比缩放)。
不需要设置 `tc_input_count`。

## 2. 数据流：PyTorch → NPU 内核

理解此流水线是创建正确映射的基础：

```
PyTorch aten 算子 (如 aten.mm)
  → op-plugin 分发 (op_plugin_functions.yaml)
    → C++ 实现 (opapi/*.cpp)
      → EXEC_NPU_CMD(aclnn*) 调用
        → CANN aclnn Host API
          → L0 OpType 注册 (CMakeLists.txt)
            → NPU 内核执行
              → Profiling kernel_details.csv
```

**关键标识符：** `kernel_details.csv` 中的 `Type` 列 = CANN OPTYPE = op_mapping.yaml 中的 `kernel_type`。

**Name 列：** 具有3段结构：`aclnnAPI_DispatchFunc_L0OpType`。第3段 = Type 列的值。

## 3. 三种映射路径

### 路径 A：aten → op-plugin → aclnn（最常见）

适用于通过 Ascend op-plugin 分发的标准 PyTorch 算子：

1. **在 op-plugin 中查找算子：** `grep "aten::mm" op_plugin/config/op_plugin_functions.yaml`
2. **找到 C++ 实现：** `op_plugin/ops/opapi/MmKernelNpuOpApi.cpp`
3. **找到 aclnn 调用：** `EXEC_NPU_CMD(aclnnMm, ...)` 或 `EXEC_NPU_CMD(aclnnMatmulWeightNz, ...)`
4. **找到 OPTYPE：** 在 CANN 仓库中搜索 aclnn 名称 → 找到 `OP_TYPE_REGISTER(MatMulV2)`
5. **对照 profiling 验证：** 检查 `MatMulV2` 是否出现在 kernel_details.csv 的 Type 列中

**示例：aten.mm.default → MatMulV2**
```
op-plugin YAML → MmKernelNpuOpApi.cpp → EXEC_NPU_CMD(aclnnMm)
  → cann-ops-nn/matmul/mat_mul_v2/ → OP_TYPE_REGISTER(MatMulV2)
  → Profiling: Type=MatMulV2
```

### 路径 B：torch_npu.npu_* → op-plugin → aclnn

适用于使用 torch_npu API 的 vLLM-ascend 专用算子：

1. **找到 vllm-ascend 调用：** `torch_npu.npu_grouped_matmul_swiglu_quant(...)`
2. **在 op-plugin 中查找：** `grep "npu_grouped_matmul_swiglu_quant" op_plugin/`
3. **沿相同的 aclnn → OPTYPE 链路追踪**

**示例：grouped_matmul_quant_swiglu → GroupedMatmulSwigluQuant**
```
vllm-ascend moe_mlp.py → torch_npu.npu_grouped_matmul_swiglu_quant
  → op-plugin → aclnnGroupedMatmulSwigluQuantWeightNZ
  → cann-ops-transformer/gmm/ → OP_TYPE_REGISTER(GroupedMatmulSwigluQuant)
  → Profiling: Type=GroupedMatmulSwigluQuant
```

### 路径 C：vLLM-ascend 自定义算子 / Triton 内核

适用于不在 op-plugin 中的算子（自定义内核、Triton、ATB）：

1. **找到 vllm-ascend 自定义算子：** 如 `vllm_ascend/ops/attention.py`
2. **判断是 Triton、csrc 还是 ATB：** 函数名通常 = profiling Type
3. **在 profiling 数据中验证**

**示例：ATB 内核**
```
vllm-ascend mla_v1.py → torch_npu.atb.npu_ring_mla()
  → ATB 内核: RINGMLAPrefillBF16Kernel
  → Profiling: Type=RINGMLAPrefillBF16Kernel
```

### 通信算子 (HCCL)

通信算子完全绕过 op-plugin：
```
TC all_reduce → torch.distributed.all_reduce → HCCL → hcom_allReduce_
```
这些算子使用 `message_bytes + num_devices` 进行查询，而非 shape 匹配。

## 4. 如何追踪单个算子（分步指南）

**目标：** 将 `tensor_cast.swiglu.default` 映射到其 NPU 内核类型。

**步骤 1：理解 TC 算子**
```bash
grep -r "def swiglu" tensor_cast/ops/
# → tensor_cast/ops/activation.py: SwiGlu 激活函数 (gate * sigmoid(gate) * up)
```

**步骤 2：找到 aten/torch_npu 路径**
SwiGlu 是一个自定义 TC 算子，因此检查 vLLM-ascend：
```bash
grep -r "swiglu\|silu_and_mul" /path/to/vllm-ascend/
# → vllm_ascend/ops/activation.py → torch_npu.npu_swiglu(...)
```

**步骤 3：查找 op-plugin 条目**
```bash
grep "npu_swiglu" /path/to/op-plugin/op_plugin/config/op_plugin_functions.yaml
# → 第 5742 行: npu_swiglu
```

**步骤 4：查找 EXEC_NPU_CMD**
```bash
grep -r "npu_swiglu" /path/to/op-plugin/op_plugin/ops/
# → SwigluKernelNpuOpApi.cpp: EXEC_NPU_CMD(aclnnSwiglu, ...)
```

**步骤 5：查找 OPTYPE**
```bash
grep -r "SwiGlu\|SWIGLU" /path/to/cann-ops-transformer/ --include="CMakeLists.txt"
# → set(OPTYPE "SwiGlu")
```

**步骤 6：在 profiling 中验证**
```bash
grep "SwiGlu" kernel_details.csv | head -3
# → Type=SwiGlu，DSv3 中 390 次，Qwen3 中 670 次
```

**结果：**
```yaml
"tensor_cast.swiglu.default":
  kernel_type: SwiGlu
  notes: "[HIGH] Path B. op-plugin: SwigluKernelNpuOpApi.cpp → aclnnSwiglu → SwiGlu."
```

## 5. 8 种 Shape 差异

TC tensor shape 与 NPU profiling shape 存在差异。`profiling_data_source.py` 自动处理这些差异，但调试时需要理解它们：

| # | 差异类型 | TC Shape | NPU Profiling Shape | 处理方式 |
|---|---------|----------|---------------------|----------|
| 1 | 批次维度 | `(1,S,D)` | `(S,D)` | 移除两边的前导 batch=1 |
| 2 | 序列填充 | `S=144` | `S=136` | block-padding 容差（向上取整到 16/32） |
| 3 | FRACTAL_NZ | `(K,N)` ND 格式 | `[H,W,bh,bw]` 分块格式 | `fractal_nz_to_nd()` 还原 |
| 4 | ND 转置 | `(K,N)` | `(N,K)` | MatMul 权重转置检查 |
| 5 | SwiGlu 拼接 | 2×`(S,D/2)` | 1×`(S,D)` | 在最后维度上拼接输入 |
| 6 | RoPE 布局 | `(B,H,S,D)` Q,K | `(B,S,H,D)` K,Q | 转置维度 + 重排输入 |
| 7 | RoPE 内核 | 单个 TC 算子 | 多个 NPU 内核 | `alternate_kernel_types` |
| 8 | 复合算子 | 融合的 TC 算子 | 分离的 NPU 内核 | `sub_kernels` 分解 |

## 6. 使用 Profiling 数据

### kernel_details.csv 列说明

| 列名 | 含义 | 用途 |
|------|------|------|
| **Type** | CANN OPTYPE = 我们的 `kernel_type` | 聚合的主键 |
| **Name** | `aclnn_Dispatch_L0OpType` 三段式 | 追溯到 aclnn API |
| **Input Shapes** | tensor shape 字符串 | CSV 中的 shape 匹配 |
| **Duration(us)** | 内核执行时间 | 性能数据 |
| **Accelerator Core** | AI Core 或 AI Vector Core | 硬件利用率 |

### 解析 Profiling 数据

```bash
# 从 kernel_details.csv 生成按内核拆分的 CSV
python3.10 -m tools.perf_data_collection.parse_kernel_details \
  --device ATLAS_800_A3_752T_128G_DIE \
  --vllm-ascend-version <version_string> \
  --kernel-details-path /path/to/kernel_details.csv

# 验证生成的数据库
python3.10 -m tools.perf_data_collection.validate \
  --database tensor_cast/performance_model/profiling_database/data/{device}/vllm_ascend/{version}/
```

### 获取唯一内核类型

```python
import csv
from collections import Counter
with open('kernel_details.csv') as f:
    types = Counter(row['Type'] for row in csv.DictReader(f))
for t, c in types.most_common():
    print(f"{c:6d}  {t}")
```

## 7. 查询分发类别

`ProfilingDataSource` 根据 op_mapping 配置通过 5 条路径路由查询：

```
是复合算子？ → _lookup_composite(sub_kernels)
是通信算子？ → _lookup_comm(message_bytes, num_devices)
是特殊注意力？ → _lookup_attention(batch, seq, heads, head_dim)
是零开销？ → QueryResult(latency=0)
默认 → _lookup_compute(kernel_type, alternate_kernel_types)
```

| 类别 | 查询方式 | 匹配依据 |
|------|---------|---------|
| `compute` | CSV shape 查找 | 输入/输出 tensor shape |
| `communication` | 消息字节数 | `tensor_nbytes * dtype_size` |
| `attention_special` | 注意力维度 | `(batch, seq_len, num_heads, head_dim)` |
| `composite` | 分解 + 求和 | 每个 sub_kernel 独立查询 |
| `zero_cost` | 返回 0 | 无需查找 |

## 8. 处理 CANN 版本差异

内核类型会在不同 CANN 版本之间发生变化。常见模式：

| 变更类型 | 示例 | 处理方式 |
|---------|------|---------|
| **重命名** | `ScatterElements` → `ScatterElementsV2` | 更新 `kernel_type`，用 `alternate_kernel_types` 兼容 |
| **融合** | 独立的 matmul+activation → 单个融合内核 | 更新 `kernel_type`（不仅仅是 `alternate_kernel_types`） |
| **拆分** | 一个内核 → 两个独立内核 | 可能需要 `composite: true` + `sub_kernels` |
| **移除** | Triton 内核被 CANN 原生融合替代 | 删除条目或更新为新内核类型 |
| **新增内核** | 新的 ATB/CANN 融合内核 | 添加新条目，通过 5 层流水线追踪 |

### 如何发现版本差异

1. **对比两个 CANN 版本的 profiling type：**
   ```bash
   # 从每次 profiling 提取唯一类型
   awk -F',' 'NR>1 {print $2}' old_kernel_details.csv | sort -u > old_types.txt
   awk -F',' 'NR>1 {print $2}' new_kernel_details.csv | sort -u > new_types.txt
   diff old_types.txt new_types.txt
   ```
2. **通过 5 层流水线（路径 A/B/C）追踪每个差异**，确定正确映射
3. **使用 `alternate_kernel_types`**：当新旧名称可能出现在不同 profiling 数据集中时

**核心经验：** 更换 CANN 版本时务必重新生成 op_mapping。profiling 的 `Type` 列是唯一真相。

### aclgraph 一致性

vllm-ascend 的 aclgraph 确保 **eager 模式和 graph 模式产生完全相同的算子**（包括融合 pass）。两种模式的 profiling 数据对 op_mapping 同样有效。

## 9. 端到端验证

验证要求**从 profiling 数据本身推导**正确的 TC 仿真参数。使用错误参数（如 profiling 捕获的是 decode 但用了 prefill 参数）会导致大量 shape 不匹配，这**不是** op_mapping 的问题。

### 步骤 1：分析 profiling 数据推导参数

**判断负载类型（prefill vs decode）：**
```bash
# 检查计算内核的批次维度 —— 小值 (1-50) = decode，大值 (100+) = prefill
for f in MatMulV2.csv AddRmsNorm.csv SwiGlu.csv; do
  echo "=== $f ===" && awk -F',' 'NR>1 {print $3}' $DATA_DIR/$f | sort | uniq -c | sort -rn | head -5
done
```

**判断量化方式：**
```bash
# QuantBatchMatmulV3.csv 存在且含 INT8 → W8A8_STATIC；仅 BF16 MatMulV2 → DISABLED
ls $DATA_DIR/*.csv | grep -i quant
```

**判断并行度 (TP/DP/EP)：**
- 比较 CSV 中间维度与模型配置：`intermediate_per_card = model.intermediate_size / TP`
- 检查 FIA 头数：`q_heads_per_card = model.num_attention_heads / TP`
- 存在 MoE 算子 (GroupedMatmul*) → 启用 EP
- 推导：`world_size = TP × DP × EP_size`

**判断批次大小：**
- 计算内核中最高频的批次维度 = 目标 `--num-queries`
- Decode：`--num-queries=<batch> --query-length=1 --context-length=4500`
- Prefill：`--num-queries=1 --query-length=<batch>`（或 nq=2 ql=batch/2）
- block-padding 匹配：TC 向上填充到 16 的倍数，因此 `ceil(nq*ql/16)*16` 必须与 CSV seq 维度匹配

### 步骤 2：使用推导的参数运行 TC 仿真

```bash
python3.10 -m tensor_cast.scripts.text_generate $MODEL \
  --num-queries $NQ --query-length $QL [--context-length $CL] \
  --device $DEVICE --world-size $WS --tp-size $TP [--dp-size $DP] [--ep-size $EP] \
  --quantize-linear-action $QUANT \
  --performance-model profiling --compile \
  --profiling-database $DATA_DIR
```

如果你在验证 FlashCommV1 对齐，可额外添加 `--enable-flashcomm-v1`。
该开关只在 `--compile` 打开时生效；同时它与 `matmul_allreduce` 这类
MC2 融合路径会竞争同一部分通信子图，因此通常应作为单独配置显式开启，
不要默认与其他 compile pass 一起混用。当前该开关仅用于 prefill 对齐；
原始 decode profiling 不启用 FlashCommV1，因此 decode 场景下应保持关闭。

### 步骤 3：对每个 MISS 进行分类

输出会显示 `EmpiricalPerformanceModel: X/Y ops matched`。对每个 MISS 进行分类：

| 差距类别 | 示例 | 处理方式 |
|---------|------|---------|
| **算子映射错误** | kernel_type 错误或缺少条目 | 修复 op_mapping.yaml |
| **Shape 数据缺口** | 内核正确但 CSV 中没有对应 shape | 补充 profiling 数据或微基准测试 |
| **TC 分解不匹配** | TC 中间 shape ≠ 真实 vLLM-ascend | 已知限制，非映射问题 |
| **结构性缺失** | Embedding、KV cache、通信算子 | 预期之中 —— TC 与 NPU 接口不同 |
| **参数不匹配** | 错误的 batch/TP 导致缺失 | 从步骤 1 重新推导参数 |

**核心原则：** shape MISS + 正确的 kernel_type = 数据覆盖缺口。shape MISS + 错误的 kernel_type = 算子映射错误。只有后者需要修复。

### 步骤 4：迭代

1. 修复所有算子映射错误
2. 如发现参数不匹配，用修正后的参数重新运行
3. 重复直到无新的算子映射错误
4. 按类别记录剩余差距

### 步骤 5：运行自动化测试

```bash
python3.10 -m pytest tests/perf_database/test_reference_data_e2e.py -v
```

### 各模型类型预期结果

| 模型类型 | 典型匹配率 | 说明 |
|---------|----------|------|
| Dense BF16（如 Qwen3-32B） | 80-90% | 剩余缺失：attention、KV cache、embedding、通信 |
| Dense W8A8 | 70-85% | 量化算子可能有不同的中间 shape |
| MoE W8A8（如 DSv3） | 35-50% | MoE 路由产生可变批次大小；MLA 分解复杂 |

MoE 模型匹配率较低是预期的，因为 TC 的 compile pass 产生的中间 shape 与真实 vLLM-ascend 不同（尤其是 MLA 投影和 MoE 分发部分）。

## 10. 常见陷阱

1. **缺少 `--compile`**：不加此参数时，融合算子（SwiGlu、AddRmsNorm、MC2、FlashCommV1）会分解为 70+ 个 aten 原始算子，无法匹配 profiling 内核。profiling 模式下务必使用 `--compile`。

2. **混用 `--enable-flashcomm-v1` 与 MC2 配置**：FlashCommV1 与 `matmul_allreduce` 会改写部分重叠的通信模式，通常应视为二选一的 compile 配置。做 profiling 对齐时，先明确当前要验证哪条路径，再决定是否添加 `--enable-flashcomm-v1`。另外，当前 FlashCommV1 只用于 prefill，对齐 decode profiling 时不要开启。

3. **验证参数错误**：用 prefill 参数（`--query-length 3500`）去验证 decode 的 profiling 数据（`batch=4, query-length=1`）会导致大量 shape 不匹配。务必先从 CSV shape 推导参数（见第 9 节步骤 1）。

4. **重命名内核类型错误**：CANN 版本可能重命名内核。务必核实 profiling 数据中的 `Type` 列，并用 `alternate_kernel_types` 实现跨版本兼容。

5. **混淆 Name 和 Type 列**：`Type` 列是干净的 OPTYPE（我们的查询键），`Name` 列是完整的层级路径。始终按 Type 聚合。

6. **复合 vs 单一**：某些 TC 算子映射到多个 NPU 内核（MLA decode = BatchMatMulV2 + FIA + batch_matmul_transpose）。使用 `composite: true` + `sub_kernels`。

7. **Shape 不匹配 ≠ 映射错误**：shape MISS（CSV 中无匹配 shape）与映射错误（kernel_type 错误）不同。shape 缺失是数据覆盖缺口，不是 op_mapping 的问题。

8. **MoE 模型低匹配率**：MoE 模型（DSv3）天然匹配率较低（35-50%），因为 TC 的 MoE 分发和 MLA 分解产生的中间 shape 与真实 vLLM-ascend 不同。这是已知的 TC 仿真限制，不是算子映射错误。

9. **通信算子不使用 shape**：HCCL 算子（allreduce、allgather 等）使用 message_bytes，不使用 tensor shape。不要尝试 shape 匹配。

## 11. 快速参考：常用映射

### 标准 aten 算子
| TC 算子 | NPU 内核 | 说明 |
|---------|---------|------|
| aten.mm | MatMulV2 | 标准矩阵乘法 |
| aten.bmm | BatchMatMulV2 | 批次矩阵乘法（备选：batch_matmul_transpose） |
| aten.addmm | MatMulV2 | Bias 融合到 MatMulV2 |
| aten.add.Tensor | Add | 逐元素加法 |
| aten.mul.Tensor | Mul | 逐元素乘法 |
| aten.div.Tensor | Div | 除法（备选：RealDiv） |
| aten.embedding | GatherV2 | Embedding 查找（备选：GatherV3） |
| aten.to.dtype | Cast | 类型转换（备选：TensorMove） |
| aten.clone | TensorMove | 内存拷贝 |
| aten.scatter.value | ScatterElementsV2 | Scatter 写入 |
| aten.sum.dim_IntList | ReduceSum | 求和归约 |
| aten.topk | TopKV2 | Top-K 选择 |

### TensorCast 融合算子
| TC 算子 | NPU 内核 | 说明 |
|---------|---------|------|
| tc.swiglu | SwiGlu | SwiGlu 激活 |
| tc.rms_norm | RmsNorm | RMS 归一化 |
| tc.add_rms_norm/2 | AddRmsNorm | 残差 + RmsNorm |
| tc.apply_rope | InterleaveRope | RoPE（备选：ApplyRotaryPosEmb） |
| tc.attention | FusedInferAttentionScore | 融合注意力 |
| tc.reshape_and_cache | ReshapeAndCacheNdKernel | KV cache 写入 |
| tc.kv_rmsnorm_rope_cache | KvRmsNormRopeCache | 融合 KV norm+RoPE+cache |

### 量化算子
| TC 算子 | NPU 内核 | 说明 |
|---------|---------|------|
| tc.static_quant_linear | QuantBatchMatmulV3 | INT8 矩阵乘法 |
| tc.static_quant_linear_int4 | QuantBatchMatmulV3 | INT4 矩阵乘法 |
| tc.fp8_linear | QuantBatchMatmulV3 | FP8 矩阵乘法 |
| tc.quantize | AscendQuantV2 | 静态量化 |
| tc.dynamic_quantize_symmetric | DynamicQuant | 动态量化 |
| tc.grouped_matmul_quant_swiglu | GroupedMatmulSwigluQuant | MoE 融合 gate-up |
| tc.grouped_matmul_quant | GroupedMatmul | MoE 矩阵乘法 |

### 通信算子
| TC 算子 | NPU 内核 | 说明 |
|---------|---------|------|
| tc.all_reduce | hcom_allReduce_ | HCCL all-reduce |
| tc.all_gather | hcom_allGather_ | HCCL all-gather |
| tc.all_to_all | hcom_alltoallv_ | HCCL all-to-all（MoE） |
| tc.reduce_scatter | HcomReduceScatter | HCCL reduce-scatter |

### 复合算子
| TC 算子 | 子内核 | 说明 |
|---------|-------|------|
| tc.matmul_all_reduce | MatMulV2 + hcom_allReduce_ | MC2 融合 |
| tc.static_quant_linear_all_reduce | QuantBatchMatmulV3 + hcom_allReduce_ | 量化 MC2 |
| tc.multihead_latent_attention | BatchMatMulV2 + FIA + batch_matmul_transpose | MLA decode |
| tc.mlapo | MatMulV2 + KvRmsNormRopeCache | MLA 预处理 |

### 零开销算子
view, permute, split, split_with_sizes, select, slice, transpose, unsqueeze, expand, full, detach, alias, arange, t, convert_element_type

## 12. 工具参考

| 工具 | 用途 | 关键参数 |
|------|------|---------|
| `parse_kernel_details.py` | 将 kernel_details.csv 拆分为逐内核 CSV | `--device`, `--vllm-ascend-version`, `--kernel-details-path` |
| `extract_tc_ops.py` | 从 chrome trace 提取 TC 算子 | `--chrome-trace`, `--output`, `--op-mapping` |
| `validate.py` | 验证 CSV 数据库质量 | `--database` |
| `discover_operators.py` | 对比 profiling 与 op_mapping 覆盖率 | — |
| `generate_shape_grid.py` | 生成微基准测试 shape 网格 | — |
| `generate_microbench.py` | 生成 torch_npu 基准测试脚本 | — |
| `build_database.py` | 合并多个 CSV 数据源 | `--sources`, `--target` |

## 13. 相关文档

- [设计文档](../OPERATOR_PERF_DATABASE_DESIGN_zh_v1.2.md) —— 完整架构和设计原理
- [Op Mapping 技能](../skills/op-mapping/SKILL.md) —— 使用并行子代理的自动化 op_mapping 生成
- [Spike 报告](../reports/spike_executive_summary_zh.md) —— 初始 spike 调研结果
