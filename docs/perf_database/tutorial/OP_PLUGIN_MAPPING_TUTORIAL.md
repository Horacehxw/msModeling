# Op-Plugin 算子映射教程

> 从 PyTorch 算子追溯到 NPU Kernel，从 Profiling 数据反向映射回 PyTorch 算子，
> 以及如何编写和验证 `op_mapping.yaml` 条目。

---

## 目录

- [1. 环境准备](#1-环境准备)
- [2. 整体数据流](#2-整体数据流)
- [3. 三条映射路径](#3-三条映射路径)
- [4. 路径 A: op-plugin ATen 映射](#4-路径-a-op-plugin-aten-映射)
- [5. 路径 B: vLLM-ascend pybind 映射](#5-路径-b-vllm-ascend-pybind-映射)
- [6. 路径 C: vLLM-ascend Triton / Graph Fusion](#6-路径-c-vllm-ascend-triton--graph-fusion)
- [7. 反向映射: Profiling Kernel → PyTorch 算子](#7-反向映射-profiling-kernel--pytorch-算子)
- [8. op_mapping.yaml 结构与字段](#8-op_mappingyaml-结构与字段)
- [9. TC vs NPU 的 8 种 Shape 差异](#9-tc-vs-npu-的-8-种-shape-差异)
- [10. 常见算子映射速查表](#10-常见算子映射速查表)
- [11. 完整实战: Qwen3-32B Prefill 映射](#11-完整实战-qwen3-32b-prefill-映射)
- [12. 验证: 端到端 Profiling 模式仿真](#12-验证-端到端-profiling-模式仿真)
- [13. 附录: op-plugin 搜索技巧](#13-附录-op-plugin-搜索技巧)
- [14. 自动化验证: extract_tc_ops.py](#14-自动化验证-extract_tc_opspy)
- [15. AI Agent 工作流](#15-ai-agent-工作流)
- [16. 已知限制与版本特定说明](#16-已知限制与版本特定说明)
- [17. 端到端 Shape Matching 验证](#17-端到端-shape-matching-验证)
- [18. 相关文档](#18-相关文档)

---

## 1. 环境准备

### 1.1 设置环境变量

本教程全程使用环境变量指向外部项目，**不使用硬编码路径**。请根据实际安装位置设置：

```bash
# 必需: op-plugin 源码 (branch: 7.2.0, CANN 8.3)
export OP_PLUGIN_DIR="/path/to/op-plugin"

# 可选: vLLM-ascend 源码 (branch: releases/v0.13.0)
export VLLM_ASCEND_DIR="/path/to/vllm-ascend"

# 可选: pytorch-npu (PTA) 源码 (branch: v2.8.0-7.2.0)
export PYTORCH_NPU_DIR="/path/to/pytorch-npu"

# msmodeling 项目根目录
export MSMODELING_DIR="/path/to/msmodeling"
```

### 1.2 验证环境

```bash
# Checklist: 确认关键文件存在
[ -f "$OP_PLUGIN_DIR/op_plugin/config/op_plugin_functions.yaml" ] && echo "OK: op-plugin YAML" || echo "MISSING: op-plugin YAML"
[ -d "$OP_PLUGIN_DIR/op_plugin/ops/opapi/" ] && echo "OK: opapi dir" || echo "MISSING: opapi dir"
[ -f "$MSMODELING_DIR/tensor_cast/performance_model/perf_database/profiling_data_source.py" ] && echo "OK: profiling_data_source.py" || echo "MISSING: profiling_data_source.py"
```

期望输出：三行均为 `OK`。

### 1.3 op_mapping.yaml 位置

```
$MSMODELING_DIR/tensor_cast/performance_model/perf_database/data/{device}/vllm_ascend/{version}/op_mapping.yaml
```

当前生产版本路径：

```
tensor_cast/performance_model/perf_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/v0.13.0/op_mapping.yaml
```

版本 0.15.0，包含 102 个条目（71 kernel_type、14 composite、17 zero_cost）。

---

## 2. 整体数据流

```
PyTorch 用户代码 (torch.mm / F.linear / ...)
    │ ATen dispatcher
    ▼
aten::mm.default (PyTorch ATen IR)
    │ torch_npu dispatch key
    ▼
op_plugin::mm() (OpInterface.cpp, codegen 生成)
    │ JIT 检查
    ▼
op_api::mm() (ops/opapi/MmKernelNpuOpApi.cpp)
    │ EXEC_NPU_CMD
    ▼
aclnnMm / aclnnMatmulWeightNz (CANN Runtime)
    │ NPU 硬件执行
    ▼
kernel_details.csv:
  Name = aclnnMatmulWeightNz_MatMulCommon_MatMulV2
  Type = MatMulV2
```

**关键概念**：

| 概念 | 说明 |
|------|------|
| **Name 列** | 完整调用链 `aclnn前缀_实现路径_基础内核` |
| **Type 列** | 基础内核类型，**用 Type 列做算子聚合和数据库查询** |
| **EXEC_NPU_CMD** | C++ 宏，第一个参数就是 aclnn 内核名 |
| **一对多** | 一个 PyTorch 算子可能映射到多个 aclnn kernel（取决于 format、dtype） |

---

## 3. 三条映射路径

```
路径 A (op-plugin ATen):
  PyTorch aten → op-plugin YAML → C++ EXEC_NPU_CMD(aclnn*) → Profiling Type

路径 B (vLLM-ascend pybind):
  vLLM-ascend Python → torch_npu.npu_* → op-plugin → aclnn* → Profiling Type

路径 C (vLLM-ascend Triton / Graph Fusion):
  vLLM-ascend fusion pass → custom kernel → Profiling Type
```

**选择路径的决策树**：

1. 在 op-plugin `EXEC_NPU_CMD` 中搜到 → **路径 A**
2. 在 vLLM-ascend 中搜到 `torch_npu.npu_*` 调用 → **路径 B**
3. 算子名带 `_kernel` 后缀 / 来自 graph fusion pass → **路径 C**
4. HCCL 通信算子（`hcom_*`）→ 直接映射，不经过 op-plugin

---

## 4. 路径 A: op-plugin ATen 映射

### 正向映射 Checklist (PyTorch → NPU Kernel)

给定一个 PyTorch 算子（如 `aten::mm`），执行以下步骤：

#### Step 1: 在 YAML 中查找算子签名

```bash
grep -n "func: mm\b" \
  "$OP_PLUGIN_DIR/op_plugin/config/op_plugin_functions.yaml" | head -5
```

期望输出（行号仅供参考，以 grep 实际结果为准）：

```
3476:  - func: mm(Tensor self, Tensor mat2) -> Tensor
3481:  - func: mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
```

确认有 `op_api` 字段 → 存在 ACLNN 实现。

#### Step 2: 找到实现文件

```bash
find "$OP_PLUGIN_DIR/op_plugin/ops/opapi/" -iname "*mm*" -o -iname "*matmul*" | head -10
```

#### Step 3: 提取 CANN 内核名

```bash
grep "EXEC_NPU_CMD" \
  "$OP_PLUGIN_DIR/op_plugin/ops/opapi/MmKernelNpuOpApi.cpp"
```

期望输出：

```
EXEC_NPU_CMD(aclnnMatmulWeightNz, self, mat2, result, cube_math_type);
EXEC_NPU_CMD(aclnnMm, self, mat2, result, cube_math_type);
```

#### Step 4: 确定 Profiling 中的 Type

在 `kernel_details.csv` 中搜索 Name 列包含 `aclnnMm` 的行，确认 Type 列值：

- `aclnnMm` → Type = `MatMulV2`
- `aclnnMatmulWeightNz` → Type = `MatMulV2`

#### Step 5: 写入 op_mapping.yaml

```yaml
"aten.mm.default":
  kernel_type: MatMulV2
  notes: >
    op-plugin: MmKernelNpuOpApi.cpp.
    aclnn: aclnnMm (ND), aclnnMatmulWeightNz (NZ权重).
```

### 快速查询脚本

```bash
#!/bin/bash
# find_kernel.sh — 查找 PyTorch 算子对应的 NPU 内核
# 用法: ./find_kernel.sh <算子名>  例: ./find_kernel.sh mm
# 前提: 已设置 $OP_PLUGIN_DIR 环境变量

OP_NAME="$1"
: "${OP_PLUGIN_DIR:?ERROR: OP_PLUGIN_DIR not set}"

echo "=== Step 1: YAML 配置 ==="
grep -n "func: ${OP_NAME}\b" \
  "$OP_PLUGIN_DIR/op_plugin/config/op_plugin_functions.yaml" | head -5

echo ""
echo "=== Step 2: 实现文件 ==="
find "$OP_PLUGIN_DIR/op_plugin/ops/opapi/" -iname "*${OP_NAME}*" 2>/dev/null

echo ""
echo "=== Step 3: CANN 内核 ==="
grep -r "EXEC_NPU_CMD" "$OP_PLUGIN_DIR/op_plugin/ops/opapi/" 2>/dev/null \
  | grep -i "${OP_NAME}" | grep -o "EXEC_NPU_CMD([^,]*" | sort -u
```

### op_plugin_functions.yaml 关键字段

| 字段 | 含义 | 示例 |
|------|------|------|
| `func` | PyTorch 函数签名 (ATen IR) | `mm(Tensor self, Tensor mat2) -> Tensor` |
| `acl_op` | ACLOP 版本范围 (legacy) | `all_version`, `[v2.1, v2.5]` |
| `op_api` | ACLNN 版本范围 (优先) | `all_version`, `[v2.7, newest]` |
| `gen_opapi.exec` | 指定 CANN 内核名 | `aclnnAcos` |
| `internal_format_opapi` | 支持内部 tensor 格式 | `all_version` |

### aclnn 命名规则

```
aclnn<操作名>[<变体>]

aclnnMm                               # 标准矩阵乘
aclnnMatmulWeightNz                   # NZ 格式权重矩阵乘
aclnnAddRmsNorm                       # Add + RMSNorm 融合
aclnnFusedInferAttentionScoreV2/V3    # 融合 Attention（版本后缀常见）
aclnnGroupedMatmulSwigluQuantWeightNZ # 融合 GroupedMatmul+SwiGlu+Quant
```

---

## 5. 路径 B: vLLM-ascend pybind 映射

适用于 Profiling 算子不来自 ATen dispatcher，而来自 vLLM-ascend 直接调用 `torch_npu.npu_*` API 的情况。

### Checklist

#### Step 1: 在 vLLM-ascend 中搜索 torch_npu 调用

```bash
: "${VLLM_ASCEND_DIR:?ERROR: VLLM_ASCEND_DIR not set}"
grep -rn "torch_npu\." "$VLLM_ASCEND_DIR/vllm_ascend/" --include="*.py" \
  | grep "<目标关键字>"
```

#### Step 2: 确认 op-plugin 入口

在 op-plugin YAML 中搜索对应的 `npu_*` 自定义算子：

```bash
grep -n "func: npu_<目标算子名>" \
  "$OP_PLUGIN_DIR/op_plugin/config/op_plugin_functions.yaml"
```

#### Step 3: 在 C++ 实现中提取 aclnn 名

```bash
grep "EXEC_NPU_CMD" "$OP_PLUGIN_DIR/op_plugin/ops/opapi/<实现文件>.cpp"
```

#### Step 4: 在 Profiling 中确认 Type

### 示例: MoeGatingTopK

**映射链**：

1. vLLM-ascend: `vllm_ascend/ops/experts_selector.py` → `torch_npu.npu_moe_gating_top_k()`
2. op-plugin: `npu_moe_gating_top_k` → `MoeGatingTopKKernelNpuOpApi.cpp` → `EXEC_NPU_CMD(aclnnMoeGatingTopK)`
3. Profiling: Type = `MoeGatingTopK`

**背景**: TensorCast 用 `aten.topk` 实现 MoE 路由，NPU 有专用融合 kernel `MoeGatingTopK`。

### 示例: KvRmsNormRopeCache

**映射链**：

1. vLLM-ascend: `vllm_ascend/ops/mla_v1.py` → `torch_npu.npu_kv_rmsnorm_rope_cache()`
2. op-plugin: `npu_kv_rmsnorm_rope_cache` → `EXEC_NPU_CMD(aclnnKvRmsNormRopeCache)`
3. Profiling: Type = `KvRmsNormRopeCache`

**背景**: TensorCast 分解为 `rms_norm` + `apply_rope` + `reshape_and_cache` 三个独立算子。

---

## 6. 路径 C: vLLM-ascend Triton / Graph Fusion

适用于 vLLM-ascend 自定义 Triton kernel 或 graph fusion pass 生成的算子，**不在 op-plugin 中**。

### Checklist

#### Step 1: 搜索 fusion pass

```bash
: "${VLLM_ASCEND_DIR:?ERROR: VLLM_ASCEND_DIR not set}"
grep -rn "FusionPass\|fusion_pass" "$VLLM_ASCEND_DIR/vllm_ascend/" --include="*.py"
```

#### Step 2: 搜索 Triton kernel

```bash
find "$VLLM_ASCEND_DIR/vllm_ascend/" -name "*.py" \
  -exec grep -l "triton\|tl\." {} \;
```

#### Step 3: 确认 Profiling Type

Triton kernel 的 Profiling Type 通常是 **函数名本身**（如 `split_qkv_rmsnorm_rope_kernel`）。

### 示例: split_qkv_rmsnorm_rope_kernel

1. vLLM-ascend: `vllm_ascend/ops/attention.py` 中 `QKNormRopeFusionPass`
2. 这是 Triton 自定义 kernel，不在 op-plugin 中
3. Profiling: Type = `split_qkv_rmsnorm_rope_kernel`（Qwen3-30B Prefill 64x）

**背景**: Qwen3 等有 `qk_norm` 的模型，vLLM-ascend 将 split QKV + RmsNorm + RoPE 融合为单个 kernel。

### 何时使用路径 C

- Profiling Type 在 op-plugin `EXEC_NPU_CMD` 中搜不到
- 算子名带 `_kernel` 后缀
- vLLM-ascend 特有的 graph fusion pass（MC2, QKNormRopeFusionPass 等）

---

## 7. 反向映射: Profiling Kernel → PyTorch 算子

给定 Profiling `kernel_details.csv` 中的一个内核，反向查找对应的 PyTorch 算子。

### Checklist

#### Step 1: 从 CSV 提取内核信息

```csv
Name,Type,Input Shapes,Input Data Types,Duration(us)
aclnnMatmulWeightNz_MatMulCommon_MatMulV2,MatMulV2,"1,4096,4096","DT_BF16",123.45
```

- **Name** 的 `aclnn` 前缀 = `aclnnMatmulWeightNz`
- **Type** = `MatMulV2`（数据库查询 key）

#### Step 2: 在 op-plugin 中搜索 aclnn 名称

```bash
grep -r "aclnnMatmulWeightNz" \
  "$OP_PLUGIN_DIR/op_plugin/ops/" --include="*.cpp" -l
```

期望输出：

```
ops/opapi/MmKernelNpuOpApi.cpp
ops/opapi/MatmulKernelNpuOpApi.cpp
```

#### Step 3: 从实现文件确定函数名

```bash
grep -B 10 "aclnnMatmulWeightNz" \
  "$OP_PLUGIN_DIR/op_plugin/ops/opapi/MmKernelNpuOpApi.cpp" \
  | grep "at::Tensor.*("
```

期望输出：

```
at::Tensor mm(const at::Tensor &self, const at::Tensor &mat2) {
```

#### Step 4: 在 YAML 中确认 ATen 签名

```bash
grep -A 3 "func: mm(" \
  "$OP_PLUGIN_DIR/op_plugin/config/op_plugin_functions.yaml"
```

**结论**: Profiling `MatMulV2` ← `aclnnMatmulWeightNz` ← `op_api::mm()` ← `aten::mm.default`

### 反向查询脚本

```bash
#!/bin/bash
# find_pytorch_op.sh — 从 Profiling 内核名反向查找 PyTorch 算子
# 用法: ./find_pytorch_op.sh <aclnn名称>  例: ./find_pytorch_op.sh aclnnMatmulWeightNz
# 前提: 已设置 $OP_PLUGIN_DIR 环境变量

KERNEL_NAME="$1"
: "${OP_PLUGIN_DIR:?ERROR: OP_PLUGIN_DIR not set}"

echo "=== Step 1: 搜索实现文件 ==="
grep -r "${KERNEL_NAME}" "$OP_PLUGIN_DIR/op_plugin/ops/" --include="*.cpp" -l

echo ""
echo "=== Step 2: 提取函数名 ==="
for f in $(grep -r "${KERNEL_NAME}" "$OP_PLUGIN_DIR/op_plugin/ops/opapi/" --include="*.cpp" -l); do
    echo "--- ${f} ---"
    grep -B 20 "${KERNEL_NAME}" "$f" | grep -E "^(at::Tensor|void|std::tuple)" | tail -1
done

echo ""
echo "=== Step 3: 推断算子名 ==="
for f in $(grep -r "${KERNEL_NAME}" "$OP_PLUGIN_DIR/op_plugin/ops/opapi/" --include="*.cpp" -l); do
    basename "$f" | sed 's/KernelNpuOpApi.cpp//' | sed 's/NpuOpapi.cpp//'
done
```

---

## 8. op_mapping.yaml 结构与字段

### 文件结构

```yaml
version: "0.15.0"
device: ATLAS_800_A3_752T_128G_DIE
cann_version: "8.1.RC1"
op_plugin_version: "7.2.0"

communication_data_ref: "../../hccl/v8.1.RC1/"
communication_fallback: analytic

interpolation_policy:
  default_method: linear
  kernel_overrides:
    FusedInferAttentionScore:
      shape_transform: sqrt    # O(seq^2) 的算子在 sqrt(seq) 空间插值

operator_mappings:
  "aten.mm.default":
    kernel_type: MatMulV2
    notes: >
      证据链...
```

### operator_mappings 条目字段

| 字段 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `kernel_type` | string | * | Profiling Type 列的值，CSV 查询 key |
| `category` | string | | 算子类别，驱动查询分派（如 `communication`） |
| `query_mode` | string | | 特殊查询模式（如 `attention_special`） |
| `composite` | bool | | `true` 表示 1:N 映射，需分解后逐 kernel 查询 |
| `sub_kernels` | list | | 复合映射的子内核列表 |
| `zero_cost` | bool | | `true` 表示纯 shape 操作，不产生 kernel 执行 |
| `alternate_kernel_types` | list | | 备选 kernel_type 列表，主 kernel 未命中时依次尝试 |
| `interpolation_policy` | object | | 条目级插值策略覆盖 |
| `notes` | string | | 证据链 + 补充说明 |

`*` = `kernel_type`、`composite`、`zero_cost` 三者必须有且仅有一个。

### 五种查询分派类别

| 类别 | 触发条件 | 查询方法 | 数据源 |
|------|----------|----------|--------|
| `compute` | 有 `kernel_type`，无特殊 category | op_mapping → kernel_type → CSV shape match | `{kernel_type}.csv` |
| `communication` | `category: communication` | message_bytes + topology_tier | comm CSV |
| `composite` | `composite: true` | 分解 sub_kernels → 逐个查询 → 求和 | 各 sub-kernel CSV |
| `attention_special` | `query_mode: attention_special` | (batch, seq, heads, head_dim) | `FusedInferAttentionScore.csv` |
| `zero_cost` | `zero_cost: true` | return 0 | 无 |

### notes 字段规范

每条映射的 `notes` 应包含完整证据链，格式：

```
[置信度] op-plugin: <YAML 位置 / 实现文件>.
aclnn: <aclnn 内核名>.
Profiling: <Type(出现次数)>.
```

置信度: `[HIGH]` 有 profiling 验证，`[MEDIUM]` 有 op-plugin 证据但无 profiling，`[LOW]` placeholder。

**重要**: YAML 行号是脆弱的（文件更新会变化），搜索时始终使用 grep 而非行号。

---

## 9. TC vs NPU 的 8 种 Shape 差异

TensorCast (TC) 虚拟运行时的 tensor shape 与 NPU Profiling CSV 中记录的 shape 存在系统性差异。
`profiling_data_source.py` 中的 `_inputs_match()` 方法依次处理这些差异。

### 差异总览

| # | 类型 | TC Shape | NPU Profiling Shape | 处理方法 |
|---|------|----------|---------------------|----------|
| 1 | Batch dim | `(1, S, D)` | `(S, D)` | `_strip_batch_dim()` 去除前导 dim=1 |
| 2 | Seq padding | `ceil(S/16)*16` | 原始 S | `_shapes_match_with_padding()` block-padding 容差 |
| 3 | FRACTAL_NZ | ND `(K, N)` | `[H, W, bh, bw]` | `fractal_nz_to_nd()` 还原 |
| 4 | ND transpose | `(K, N)` | `(N, K)` | MatMul 专用检查（仅限 2D ND 权重） |
| 5 | SwiGlu inputs | 2x `(S, D/2)` | 1x `(S, D)` | 沿 last dim 拼接 |
| 6 | RoPE layout | `(B,H,S,D)` Q,K | `(B,S,H,D)` K,Q | `_normalize_rope_inputs()` 维度转置 + 输入重排 |
| 7 | RoPE kernel | 单个 TC op | 多个 NPU kernels | `alternate_kernel_types` 回退匹配 |
| 8 | Composite ops | 融合 op (matmul+allreduce) | 可能独立 | `_lookup_composite()` 分解查询 |

### 详细说明

#### 差异 1: Batch Dim 剥离

TC 始终保留 batch 维度 `(1, seq, dim)`，NPU Profiling 中通常去掉 batch=1 得到 `(seq, dim)`。

```python
# profiling_data_source.py
def _strip_batch_dim(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    if len(shape) > 1 and shape[0] == 1:
        return shape[1:]
    return shape
```

匹配顺序：先尝试原始 shape，再尝试 strip 后的 shape。

#### 差异 2: Seq Padding (Block 对齐)

TC 对 sequence 长度做 block 对齐（Da Vinci Cube unit: BF16 16x16, INT8 16x32），
NPU Profiling 记录的是原始未 padding 的 sequence 长度。

```python
# 容差: tc_dim == ceil(csv_dim / block) * block
_BLOCK_SIZES = (16, 32, 64)
```

例如: TC shape `(3520, 4096)` 匹配 CSV shape `(3500, 4096)`，因为 `ceil(3500/16)*16 = 3504` 或使用 `ceil(3500/32)*32 = 3520`。

#### 差异 3: FRACTAL_NZ Format

NPU 权重可能以 FRACTAL_NZ 格式存储（tiled layout for matmul 加速）。
当 CSV 的 `Input Formats` 列为 `FRACTAL_NZ` 时，调用 `fractal_nz_to_nd()` 还原。

```python
def fractal_nz_to_nd(nz_shape):
    # [..., H, W, block_h, block_w] -> [..., H*block_w, W*block_h]
    *batch, H, W, block_h, block_w = nz_shape
    return (*batch, H * block_w, W * block_h)
```

示例:
- BF16: `[256, 256, 16, 16]` → `(4096, 4096)` (K=256*16, N=256*16)
- INT8: `[128, 256, 16, 32]` → `(4096, 4096)` (K=128*32, N=256*16)

#### 差异 4: ND Transpose (MatMul 权重)

TC 中 `F.linear` 内部做 `aten.mm(input, weight.T)`，权重已转置为 `(K, N)`。
NPU Profiling 中 ND 格式权重可能存储为 `(N, K)`。

仅对 `_MATMUL_KERNELS = {MatMulV2, MatMul, TransposeBatchMatMul}` 中的第 2 个输入（`i >= 1`）、
且为 2D ND 格式时检查转置匹配。

#### 差异 5: SwiGlu 输入拼接

TC dispatches 两个独立输入 `gate(S, D/2)` 和 `up(S, D/2)`。
NPU SwiGlu kernel 接收一个拼接后的输入 `(S, D)`。

```python
if kernel_type in _SWIGLU_KERNELS and len(tc_inputs) == 2 and len(csv_shapes) == 1:
    merged_shape = s1[:-1] + (s1[-1] + s2[-1],)  # 沿 last dim 拼接
```

#### 差异 6: RoPE Layout 转置

TC dispatches: `[Q(B,H,S,D), K(B,H,S,D), cos(1,S,D), sin(1,S,D)]`
NPU CSV 期望: `[K(B,S,H,D), Q(B,S,H,D), cos(B,S,1,D), sin(B,S,1,D)]`

转换包括:
- `(B,H,S,D)` → `(B,S,H,D)` 维度转置
- Q, K 输入顺序交换
- cos/sin 增加 head dim=1

#### 差异 7: RoPE Alternate Kernels

TC 的 `tensor_cast.apply_rope` 是单个 op，但 NPU 有两种 RoPE kernel:
- `InterleaveRope` (DeepSeek interleave 模式)
- `ApplyRotaryPosEmb` (neox 模式, 如 Qwen3)

通过 `alternate_kernel_types` 实现自动回退:

```yaml
"tensor_cast.apply_rope.default":
  kernel_type: InterleaveRope
  alternate_kernel_types: [ApplyRotaryPosEmb]
```

查询时先尝试 `InterleaveRope.csv`，未命中则尝试 `ApplyRotaryPosEmb.csv`。

#### 差异 8: Composite Ops 分解

TC 的融合 op（如 `matmul_all_reduce`）在 NPU 上可能是独立的 kernels。
`composite: true` 的条目由 `_lookup_composite()` 处理：将 sub_kernels 逐个查询后求和。

```yaml
"tensor_cast.matmul_all_reduce.default":
  composite: true
  sub_kernels: [MatMulV2, hcom_allReduce_]
```

---

## 10. 常见算子映射速查表

### 路径 A: 标准 ATen 算子

| PyTorch / TensorCast 算子 | op-plugin 函数 | aclnn 内核 | Profiling Type |
|---|---|---|---|
| `aten::mm.default` | `op_api::mm` | `aclnnMm`, `aclnnMatmulWeightNz` | `MatMulV2` |
| `aten::bmm.default` | `op_api::bmm` | `aclnnBatchMatMul` | `TransposeBatchMatMul` |
| `aten::addmm.default` | `op_api::addmm` | `aclnnAddmm` | `MatMulV2` |
| `aten::embedding.default` | `op_api::embedding` | `aclnnEmbedding` | `GatherV2` |
| `aten::cat.default` | `op_api::cat` | `aclnnCat` | `ConcatD` |
| `aten::add.Tensor` | `op_api::add` | `aclnnAdd` | `Add` |
| `aten::to.dtype` | `_to_copy` | `aclnnInplaceCopy` | `Cast` |

### 路径 A: torch_npu 自定义算子

| TensorCast 算子 | torch_npu API | aclnn 内核 | Profiling Type |
|---|---|---|---|
| `tensor_cast.static_quant_linear` | `npu_weight_quant_batchmatmul` | `aclnnWeightQuantBatchMatmulV3` | `QuantBatchMatmulV3` |
| `tensor_cast.grouped_matmul` | `npu_grouped_matmul` | `aclnnGroupedMatmulV4/V5` | `GroupedMatmul` |
| `tensor_cast.attention` | `npu_fused_infer_attention_score` | `aclnnFusedInferAttentionScoreV3` | `FusedInferAttentionScore` |
| `tensor_cast.add_rms_norm` | `npu_add_rms_norm` | `aclnnAddRmsNorm` | `AddRmsNorm` |
| `tensor_cast.swiglu` | `npu_swiglu` | `aclnnSwiGlu` | `SwiGlu` |
| `tensor_cast.quantize` | `npu_quantize` | `aclnnAscendQuantV3` | `AscendQuantV2` |
| `tensor_cast.apply_rope` | `npu_interleave_rope` | `aclnnInterleaveRope` | `InterleaveRope` |

### 路径 B: vLLM-ascend 专有

| PyTorch / TensorCast 算子 | 来源 | aclnn 内核 | Profiling Type |
|---|---|---|---|
| `torch_npu.npu_moe_gating_top_k` | vLLM-ascend + op-plugin | `aclnnMoeGatingTopK` | `MoeGatingTopK` |
| `torch_npu.npu_kv_rmsnorm_rope_cache` | vLLM-ascend + op-plugin | `aclnnKvRmsNormRopeCache` | `KvRmsNormRopeCache` |

### 路径 C: vLLM-ascend Triton

| 算子 | 来源 | Profiling Type |
|---|---|---|
| `split_qkv_rmsnorm_rope_kernel` | vLLM-ascend QKNormRopeFusionPass | `split_qkv_rmsnorm_rope_kernel` |

### 通信 (HCCL, 不经过 op-plugin)

| TensorCast 算子 | HCCL API | Profiling Type |
|---|---|---|
| `tensor_cast.all_reduce` | `torch.distributed.all_reduce` | `hcom_allReduce_` |
| `tensor_cast.all_gather` | `torch.distributed.all_gather` | `hcom_allGather_` |
| `tensor_cast.reduce_scatter` | `torch.distributed.reduce_scatter` | `HcomReduceScatter` |
| `tensor_cast.all_to_all` | `torch.distributed.all_to_all` | `hcom_alltoallv_` |

---

## 11. 完整实战: Qwen3-32B Prefill 映射

本节以 Qwen3-32B BF16 Prefill (16 卡 TP=16) 为例，从零建立完整映射。

### 11.1 获取 TC 算子列表

运行 TC profiling 模式仿真，收集所有 dispatched ops:

```bash
cd "$MSMODELING_DIR"

python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 2 --query-length 3500 \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --performance-model profiling --compile \
  --chrome-trace /tmp/qwen3_32b_prefill_trace.json 2>&1 | tee /tmp/tc_run.log
```

**注意**: `--compile` 是必须的 — 没有它，融合算子 (RmsNorm, SwiGlu, RoPE) 会分解为 72+ 个 aten 原语，无法匹配 profiling kernels。

### 11.2 检查 op_mapping 覆盖率

从日志中提取 MISS (未命中) 的算子:

```bash
grep "MISS" /tmp/tc_run.log | sort | uniq -c | sort -rn | head -20
```

期望: 大部分算子应命中（有 `kernel_type` 映射）。未命中的算子 fallback 到 AnalyticPerformanceModel。

### 11.3 分析 Profiling 数据中的 Top 算子

假设已有 Qwen3-32B 的 `kernel_details.csv`，提取 Top 算子:

```bash
# 统计 Type 列出现频次
PROFILING_CSV="/path/to/kernel_details.csv"
awk -F',' 'NR>1 {print $2}' "$PROFILING_CSV" | sort | uniq -c | sort -rn | head -15
```

Qwen3-30B Prefill 典型结果（16 卡）:

```
386 TensorMove
276 hcom_allReduce_
275 MatMulV2
131 AddRmsNorm
 67 FusedInferAttentionScore
 67 SwiGlu
 67 ReshapeAndCacheNdKernel
 64 split_qkv_rmsnorm_rope_kernel
```

### 11.4 逐个建立映射

按出现频次从高到低:

| # | Profiling Type | 查找路径 | TC 算子 | op_mapping 条目 |
|---|---|---|---|---|
| 1 | `MatMulV2` | A: `aclnnMm` → `aten::mm` | `aten.mm.default` | `kernel_type: MatMulV2` |
| 2 | `hcom_allReduce_` | HCCL direct | `tensor_cast.all_reduce.default` | `kernel_type: hcom_allReduce_`, `category: communication` |
| 3 | `AddRmsNorm` | A: `aclnnAddRmsNorm` → `npu_add_rms_norm` | `tensor_cast.add_rms_norm.default` | `kernel_type: AddRmsNorm` |
| 4 | `FusedInferAttentionScore` | A: `aclnnFusedInferAttentionScoreV3` | `tensor_cast.attention.default` | `kernel_type: FusedInferAttentionScore`, `query_mode: attention_special` |
| 5 | `SwiGlu` | A: `aclnnSwiGlu` → `npu_swiglu` | `tensor_cast.swiglu.default` | `kernel_type: SwiGlu` |
| 6 | `ReshapeAndCacheNdKernel` | B: ATB cache op | `tensor_cast.reshape_and_cache.default` | `kernel_type: ReshapeAndCacheNdKernel` |
| 7 | `split_qkv_rmsnorm_rope_kernel` | C: vLLM Triton | 无直接对应 (TC 分解) | profiling-only placeholder |
| 8 | `TensorMove` | 数据搬移 | `aten.clone.default` 等 | `kernel_type: TensorMove` |

### 11.5 验证搜索命令

对 Step 4 中每个映射执行验证:

```bash
# 验证 MatMulV2
grep -r "EXEC_NPU_CMD(aclnnMm\b" "$OP_PLUGIN_DIR/op_plugin/ops/opapi/" --include="*.cpp" -l
# 期望: MmKernelNpuOpApi.cpp

# 验证 AddRmsNorm
grep -r "EXEC_NPU_CMD(aclnnAddRmsNorm\b" "$OP_PLUGIN_DIR/op_plugin/ops/opapi/" --include="*.cpp" -l
# 期望: 至少 1 个文件

# 验证 SwiGlu
grep -r "EXEC_NPU_CMD(aclnnSwiGlu\b" "$OP_PLUGIN_DIR/op_plugin/ops/opapi/" --include="*.cpp" -l
# 期望: 至少 1 个文件

# 验证 FusedInferAttentionScore
grep -r "EXEC_NPU_CMD(aclnnFusedInferAttentionScore" "$OP_PLUGIN_DIR/op_plugin/ops/opapi/" --include="*.cpp" -l
# 期望: FusedInferAttentionScoreKernelNpuOpApi.cpp
```

### 11.6 写入 op_mapping.yaml

将验证通过的映射写入 YAML 文件。参考现有生产版本:

```bash
cat "$MSMODELING_DIR/tensor_cast/performance_model/perf_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/v0.13.0/op_mapping.yaml" \
  | grep -c "kernel_type:\|composite: true\|zero_cost: true"
```

期望输出: `102` (当前条目总数)

### 11.7 处理 Attention 的特殊插值

Qwen3-32B Attention 的 complexity 是 O(seq^2)，需要特殊插值策略:

```yaml
interpolation_policy:
  default_method: linear
  kernel_overrides:
    FusedInferAttentionScore:
      shape_transform: sqrt    # 在 sqrt(seq) 空间做插值
```

这意味着 `InterpolatingDataSource` 对 `FusedInferAttentionScore` 查询时，
会先对 seq 维度取 sqrt 后再做最近邻 / 线性插值。

---

## 12. 验证: 端到端 Profiling 模式仿真

### 12.1 运行仿真

```bash
cd "$MSMODELING_DIR"

# Qwen3-32B BF16 Prefill, 16 卡 TP
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 2 --query-length 3500 \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --performance-model profiling --compile \
  --chrome-trace /tmp/qwen3_32b_trace.json
```

### 12.2 检查命中率

```bash
# 统计 HIT vs MISS
grep -c "HIT\|PROFILING" /tmp/tc_run.log   # 命中数
grep -c "MISS\|ANALYTIC_FALLBACK" /tmp/tc_run.log  # 未命中数
```

目标: 命中率 > 80% 表示 op_mapping 覆盖良好。

### 12.3 检查 Chrome Trace

在 Chrome 浏览器中打开 `chrome://tracing`，加载 `/tmp/qwen3_32b_trace.json`。

检查要点:
- 每个 op 标注了 `profiling` 或 `analytic` 来源
- 主要计算算子 (MatMulV2, FusedInferAttentionScore) 应为 `profiling`
- 通信算子 (hcom_allReduce_) 应为 `profiling` 或 `communication`

### 12.4 对比验证

如果有实测数据，对比 TC 仿真结果:

```bash
# 提取 TC 总延迟
grep "total_time_us\|Total latency" /tmp/tc_run.log

# 提取 profiling 实测总延迟
awk -F',' 'NR>1 {sum += $NF} END {print sum " us"}' "$PROFILING_CSV"
```

期望: TC 仿真延迟与实测在 +/-20% 以内。

---

## 13. 附录: op-plugin 搜索技巧

### 按 gen_opapi.exec 搜索

对于 YAML 中指定了 aclnn 名的算子:

```bash
grep -B 3 "exec: aclnn" \
  "$OP_PLUGIN_DIR/op_plugin/config/op_plugin_functions.yaml" \
  | grep -E "func:|exec:"
```

### 统计所有 aclnn 内核

```bash
grep -roh "EXEC_NPU_CMD(aclnn[^,]*" \
  "$OP_PLUGIN_DIR/op_plugin/ops/opapi/" \
  | sed 's/EXEC_NPU_CMD(//' | sort -u | wc -l
```

### 查看 custom 算子 (torch_npu 专有)

```bash
grep "func: npu_" \
  "$OP_PLUGIN_DIR/op_plugin/config/op_plugin_functions.yaml" | wc -l
```

### 搜索特定融合算子

```bash
grep -r "grouped_matmul_swiglu" \
  "$OP_PLUGIN_DIR/op_plugin/ops/opapi/" --include="*.cpp" -l
```

### op-plugin 项目结构概览

```
$OP_PLUGIN_DIR/
  op_plugin/
    config/
      op_plugin_functions.yaml   # 核心: 1200+ 算子的映射配置 (~7148 行)
      derivatives.yaml           # 反向传播绑定
    ops/
      opapi/                     # ACLNN 实现 (当前主用)
        MmKernelNpuOpApi.cpp
        MatmulKernelNpuOpApi.cpp
        ...
      aclops/                    # ACLOP 实现 (legacy)
      atb/                       # Ascend Tensor Backend
    python/meta/                 # Meta shape inference
  codegen/
    gen.py                       # 代码生成入口
```

### vLLM-ascend 搜索方法

```bash
# 搜索 torch_npu API 调用
grep -rn "torch_npu\." "$VLLM_ASCEND_DIR/vllm_ascend/" --include="*.py" | grep "<关键字>"

# 搜索 graph fusion passes
grep -rn "FusionPass\|fusion_pass" "$VLLM_ASCEND_DIR/vllm_ascend/" --include="*.py"

# 搜索 Triton kernels
find "$VLLM_ASCEND_DIR/vllm_ascend/" -name "*.py" -exec grep -l "triton\|tl\." {} \;
```

---

## 14. 自动化验证: extract_tc_ops.py

### 14.1 生成 TC Trace

运行 TensorCast 仿真生成 chrome trace（`--chrome-trace` 输出 JSON）：

```bash
cd "$MSMODELING_DIR"

# Qwen3-32B BF16 Prefill
python3.10 -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 2 --query-length 3500 \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --performance-model profiling --compile \
  --chrome-trace /tmp/qwen3_32b_prefill_trace.json

# DSv3 W8A8 Decode
python3.10 -m tensor_cast.scripts.text_generate deepseek-ai/DeepSeek-V3 \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 32 --tp-size 4 --dp-size 8 --ep \
  --quantize-linear-action W8A8_STATIC \
  --performance-model profiling --compile \
  --chrome-trace /tmp/dsv3_decode_trace.json
```

### 14.2 提取 Op 列表

使用 `extract_tc_ops.py` 从 trace 中提取所有 dispatched ops：

```bash
python3.10 tools/perf_data_collection/extract_tc_ops.py \
  --chrome-trace /tmp/qwen3_32b_prefill_trace.json \
  --output docs/perf_database/reports/op_mapping/traces/qwen3_32b_prefill.json \
  --model "Qwen/Qwen3-32B" --scenario prefill
```

输出 JSON 包含：
- `total_ops`: 总算子调用数
- `unique_ops`: 唯一算子数
- `mapped_count / unmapped_count`: 映射覆盖率
- `ops[]`: 每个算子的 count、shape_variants、kernel_type 等
- `unmapped_ops[]`: 未在 op_mapping.yaml 中找到的算子列表

### 14.3 检查未映射算子

```bash
python3.10 -c "
import json
with open('docs/perf_database/reports/op_mapping/traces/qwen3_32b_prefill.json') as f:
    data = json.load(f)
print(f'Mapped: {data[\"mapped_count\"]}, Unmapped: {data[\"unmapped_count\"]}')
if data['unmapped_ops']:
    print(f'Unmapped: {data[\"unmapped_ops\"]}')
else:
    print('All ops mapped!')
"
```

### 14.4 验证覆盖率

运行所有 4 个配置并检查全覆盖：

```bash
for trace in /tmp/{qwen3_32b,dsv3}_{prefill,decode}_trace.json; do
    name=$(basename "$trace" .json)
    python3.10 tools/perf_data_collection/extract_tc_ops.py \
      --chrome-trace "$trace" \
      --output "docs/perf_database/reports/op_mapping/traces/${name}.json" \
      --model "auto" --scenario "auto"
done
```

---

## 15. AI Agent 工作流

本节提供 AI Agent（如 Claude）自动验证或生成 op_mapping 的步骤。

### 15.1 自动化验证流程

```
1. 读取 op_mapping.yaml → 获取所有 operator_mappings 条目
2. 对每条 entry:
   a. 提取 TC op name
   b. 如果有 kernel_type:
      - 在 op-plugin YAML 中搜索对应的 npu_* API
      - grep EXEC_NPU_CMD → 确认 aclnn 内核名
      - 确认 Profiling Type 列值 == kernel_type
   c. 如果 composite: true:
      - 逐个验证 sub_kernels
   d. 如果 zero_cost: true:
      - 确认 NPU 上无 kernel 执行（view/permute/slice 等）
3. 运行 TC 仿真 → 检查 unmapped_ops 列表为空
4. 生成 verification_report.md
```

### 15.2 自动化新模型 op_mapping 生成

```
1. 运行 TC 仿真: text_generate <model> --compile --chrome-trace
2. 提取 ops: extract_tc_ops.py → JSON
3. 对每个 unmapped op:
   a. 在 op-plugin 中搜索 (Path A/B/C)
   b. 确定 kernel_type
   c. 添加到 op_mapping.yaml
4. 重新运行仿真验证 100% 覆盖
```

### 15.3 关键 grep 命令参考

```bash
# 正向: TC op → op-plugin
grep -n "func: <op_name>" "$OP_PLUGIN_DIR/op_plugin/config/op_plugin_functions.yaml"
grep -r "EXEC_NPU_CMD" "$OP_PLUGIN_DIR/op_plugin/ops/opapi/<Impl>.cpp"

# 反向: Profiling Type → op-plugin
grep -r "<aclnn_name>" "$OP_PLUGIN_DIR/op_plugin/ops/" --include="*.cpp" -l

# vLLM-ascend API 搜索
grep -rn "torch_npu\." "$VLLM_ASCEND_DIR/vllm_ascend/" --include="*.py"
```

### 15.4 版本注意事项

- **YAML 行号是脆弱的**: 始终用 grep 搜索，不要硬编码行号
- **op-plugin 版本**: 当前验证基于 7.2.0，新版本可能新增/移除 API
- **CANN 版本**: profiling 数据的 CANN 版本可能与 op-plugin build 版本不同
- **npu_mla_prolog_v3**: op-plugin 7.2.0 中不存在，仅有 V1/V2
- **npu_grouped_matmul_swiglu_quant_v2**: op-plugin 7.2.0 中不存在，仅有 V1

---

## 16. 已知限制与版本特定说明

### op-plugin 7.2.0 限制

| 限制 | 说明 | 影响 |
|------|------|------|
| 无 MLA Prolog V3 | 仅 V1/V2 可用 | MLAPO 映射为 composite fallback |
| 无 GMM-SwiGlu V2 | V1 仅支持 W8A8 | W4A8/FP8/MXFP4 融合路径不可用 |
| 无 MoeDistributeDispatch V4 | V2/V3 可用 | V4 新增参数在未来 CANN 中 |
| FP8 matmul 路径不确定 | 可能走 npu_weight_quant_batchmatmul | 待 FP8 profiling 验证 |
| MXFP4 支持有限 | npu_dynamic_block_quant 可用 | 但 MXFP4 scale dtype 支持待验证 |
| InterleaveRope shape normalization | `_ROPE_KERNELS` 仅含 ApplyRotaryPosEmb | InterleaveRope 为 3-input `(x,cos,sin)` 格式，需独立 normalization path；当前通过 `alternate_kernel_types` fallback 到 ApplyRotaryPosEmb 工作 |

### 仿真 vs 实际部署差异

| 差异 | TC 仿真 | 实际 NPU 部署 |
|------|---------|---------------|
| MoE routing | aten.topk + 分解 ops | MoeGatingTopK 融合 kernel |
| KV cache | rms_norm + rope + cache 分解 | KvRmsNormRopeCache 融合 |
| Qwen3 QK norm | linear + rms_norm + rope | split_qkv_rmsnorm_rope_kernel Triton |
| MC2 | matmul + all_reduce 分解 | 单个 MC2 pipeline kernel |
| InplaceAddRmsNorm | AddRmsNorm (非 in-place) | InplaceAddRmsNorm (CANN 优化) |

---

## 17. 端到端 Shape Matching 验证

本节描述如何使用 stub CSV 和 ProfilingDataSource 进行端到端验证，确认 op_mapping + shape matching 全流程正确。

### 17.1 生成 Stub CSV

```bash
# Step 1: 运行 TC 仿真生成 chrome trace (4 个配置)
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 2 --query-length 3500 \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --performance-model profiling --compile \
  --chrome-trace ./qwen3_prefill_trace.json

# Step 2: 提取 ops (使用 extract_tc_ops.py)
python tools/perf_data_collection/extract_tc_ops.py \
  --trace ./qwen3_prefill_trace.json \
  --op-mapping tensor_cast/performance_model/perf_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/v0.13.0/op_mapping.yaml \
  --output docs/perf_database/reports/op_mapping/traces/qwen3_32b_prefill.json

# Step 3: 从 trace 数据生成 stub CSV
python tools/perf_data_collection/generate_stub_csvs.py \
  --traces docs/perf_database/reports/op_mapping/traces/ \
  --op-mapping tensor_cast/performance_model/perf_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/v0.13.0/op_mapping.yaml \
  --output-dir /tmp/stub_csvs
```

### 17.2 运行端到端测试

```bash
pytest tests/perf_database/test_op_mapping_e2e.py -v
```

测试覆盖:
- **test_stub_csvs_generated**: 验证 stub CSV 文件生成 (≥25 个 kernel type)
- **test_all_kernel_types_have_csvs**: 每个非 skip op 的 kernel_type 都有对应 CSV
- **test_zero_cost_ops_return_zero**: 17 个 zero_cost ops 返回 latency=0
- **test_communication_ops_return_none**: 4 个通信 ops 返回 None (fallback to analytic)
- **test_compute_ops_shape_match**: **所有 33 个计算 ops 的全部 shape variants 100% 命中**
- **test_config_full_coverage[{config}]**: 4 个配置分别验证 100% 覆盖率

### 17.3 Shape Transform 对照表

Stub CSV 生成器应用以下 TC→NPU 转换 (与 `profiling_data_source._inputs_match` 互逆):

| 转换 | TC 侧 | NPU CSV 侧 | 生成器逻辑 |
|------|--------|-------------|-----------|
| Batch dim strip | `(1, S, D)` | `(S, D)` | `strip_batch_dim()` |
| SwiGlu concat | 2×`(S, D/2)` | 1×`(S, D)` | 合并最后维度 |
| RoPE swap+transpose | `[Q(B,H,S,D), K, cos, sin]` | `[K(B,S,H,D), Q, cos(B,S,1,D), sin]` | 交换+转置+插入 head dim |
| MatMul weight | `(K, N)` | `(N, K)` | 转置 weight 矩阵 |
| Scalar filter | `()` scalar tensors | 不出现在 CSV | 过滤空形状 |
| Composite decomp | 融合 op | 子 kernel CSV | 按 sub_kernels 分别查找 |

### 17.4 运行真实 TC Profiling 仿真验证

除了 mock 测试，还可以运行真实 TC 仿真确认全流程：

```bash
# Qwen3-32B Prefill — 预期 95.9% (47/49, 仅 attention_special + communication miss)
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 2 --query-length 3500 \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --performance-model profiling --compile --perf-database /tmp/stub_csvs

# DSv3 Prefill (W8A8) — 预期 90.7% (78/86, MLA composite + communication + MoE shapes)
python -m tensor_cast.scripts.text_generate deepseek-ai/DeepSeek-V3 \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 32 --tp-size 4 --dp-size 8 --ep-size 8 \
  --quantize-linear-action W8A8_STATIC --num-queries 2 --query-length 3500 \
  --performance-model profiling --compile --perf-database /tmp/stub_csvs
```

查看输出中 `EmpiricalPerformanceModel: X/Y ops matched` 行。预期 miss 分类：

| Miss 类别 | 原因 | 处理方式 |
|-----------|------|----------|
| `attention.default` | attention_special query mode | Phase 2 实现 |
| `all_reduce/all_gather/all_to_all` | communication fallback | analytic model |
| `multihead_latent_attention` | 9 tensor args, composite 无法分解 | 需要 sub-input 提取逻辑 |
| MoE combine shapes | 与 trace 不同的 shape | 需要插值或更多数据 |

### 17.5 已知 Shape Matching 细节

- **Scalar tensor (ndim=0)**: `_extract_tensor_inputs()` 过滤标量张量。如 `static_quant_linear` 的 scale 参数 shape=`()` 不参与 shape matching。
- **MoE float32**: MoE routing ops (sigmoid, topk, where 等) 使用 float32 保证数值稳定性。Stub CSV 需同时包含 DT_BF16 和 FLOAT 变体。
- **Gather/Scatter 混合 dtype**: gather 的第一个输入是数据 (FLOAT/DT_BF16), 第二个是索引 (INT64)。

### 17.6 新增 op_mapping 时的验证流程

1. 在 `op_mapping.yaml` 中添加新映射条目
2. 运行 TC 仿真生成 trace → 提取 ops → 确认新 op 出现在 trace JSON 中
3. 运行 `generate_stub_csvs.py` → 确认生成对应 kernel_type CSV
4. 运行 `pytest tests/perf_database/test_op_mapping_e2e.py -v` → 确认 100% hit rate
5. 运行真实 TC 仿真 → 确认 match rate 在预期范围内
6. 如果测试失败，检查:
   - shape 转换逻辑是否需要新的 transform (如新的 kernel 类型)
   - dtype 映射是否完整 (是否需要 FLOAT/INT64 变体)
   - scalar tensor 是否被过滤
   - op_mapping.yaml 中的 kernel_type 是否与 profiling CSV 文件名匹配

---

## 18. 相关文档

- 设计文档: [`OPERATOR_PERF_DATABASE_DESIGN_zh_v1.2.md`](../OPERATOR_PERF_DATABASE_DESIGN_zh_v1.2.md)
- op_mapping 示例: [`examples/op_mapping_example.yaml`](../examples/op_mapping_example.yaml)
- comm_config 示例: [`examples/comm_config_example.yaml`](../examples/comm_config_example.yaml)
- Spike 报告: [`reports/spike_executive_summary_zh.md`](../reports/spike_executive_summary_zh.md)
- 工作计划: [`WORK_PLAN_Q1.md`](../WORK_PLAN_Q1.md)
- Shape matching 实现: `tensor_cast/performance_model/perf_database/profiling_data_source.py`
- 插值包装器: `tensor_cast/performance_model/perf_database/interpolating_data_source.py`
- op-plugin 源码: `$OP_PLUGIN_DIR` (branch: 7.2.0, CANN 8.3.RC1)
- vLLM-ascend 源码: `$VLLM_ASCEND_DIR` (branch: releases/v0.13.0)
- 验证报告: [`reports/op_mapping/verification_report.md`](../reports/op_mapping/verification_report.md)
- 变更日志: [`reports/op_mapping/CHANGELOG.md`](../reports/op_mapping/CHANGELOG.md)
- 覆盖矩阵: [`reports/op_mapping/coverage_matrix.md`](../reports/op_mapping/coverage_matrix.md)
- 版本兼容性: [`reports/op_mapping/version_compatibility.md`](../reports/op_mapping/version_compatibility.md)
- Op 提取工具: `tools/perf_data_collection/extract_tc_ops.py`
- Stub CSV 生成器: `tools/perf_data_collection/generate_stub_csvs.py`
- 端到端测试: `tests/perf_database/test_op_mapping_e2e.py`
