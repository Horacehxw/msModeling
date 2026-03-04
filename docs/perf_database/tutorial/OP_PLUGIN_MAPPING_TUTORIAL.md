# Op-Plugin 算子映射教程

> 如何从 PyTorch 算子追溯到 NPU Kernel 名称，以及如何从 Profiling 数据反向映射回 PyTorch 算子。

## 1. 背景与目标

在 Ascend NPU 上运行 PyTorch 模型时，PyTorch 的 ATen 算子会被 **op-plugin** 映射到 NPU 硬件内核（aclnn kernel）。理解这个映射关系对以下场景至关重要：

- 将 Profiling 数据（`kernel_details.csv`）中的 NPU 内核反向映射到 PyTorch 算子
- 编写 `op_mapping.yaml` 将 TensorCast 虚拟算子对齐到实测 Profiling 内核
- 为新算子补充 microbenchmark 脚本

**前置条件**：
- 本地已有 op-plugin 代码：`/home/horacehxw/Projects/op-plugin/`
- 已有 Profiling 数据（`kernel_details.csv`）

## 2. 整体数据流

```
PyTorch 用户代码 (torch.mm / F.linear / ...)
    ↓ ATen dispatcher
aten::mm.default (PyTorch ATen IR)
    ↓ torch_npu dispatch key
op_plugin::mm() (OpInterface.cpp, 由 codegen 生成)
    ↓ JIT 检查
op_api::mm() (ops/opapi/MmKernelNpuOpApi.cpp)
    ↓ EXEC_NPU_CMD
aclnnMm / aclnnMatmulWeightNz (CANN Runtime)
    ↓ NPU 硬件执行
kernel_details.csv:
  Name = aclnnMatmulWeightNz_MatMulCommon_MatMulV2
  Type = MatMulV2
```

关键洞察：
- **Name 列**：包含完整调用链（`aclnn前缀_实现路径_基础内核`）
- **Type 列**：基础内核类型，**用 Type 列做算子聚合**
- 一个 PyTorch 算子可能映射到多个 aclnn kernel（取决于 tensor format、dtype 等）

## 3. Op-Plugin 项目结构

```
/home/horacehxw/Projects/op-plugin/
├── op_plugin/
│   ├── config/
│   │   ├── op_plugin_functions.yaml   # ★ 核心：1200+ 算子的映射配置
│   │   ├── derivatives.yaml           # 反向传播绑定
│   │   └── deprecated.yaml            # 已废弃算子
│   ├── ops/
│   │   ├── opapi/                     # ★ ACLNN 实现（当前主用）
│   │   │   ├── MmKernelNpuOpApi.cpp
│   │   │   ├── MatmulKernelNpuOpApi.cpp
│   │   │   └── ...
│   │   ├── aclops/                    # ACLOP 实现（legacy）
│   │   └── atb/                       # Ascend Tensor Backend
│   └── python/meta/                   # Meta shape inference
├── codegen/
│   ├── gen.py                         # 代码生成入口
│   └── gen_backend_stubs.py           # 后端 stub 生成
└── build/gen_code/                      # 运行 gencode.sh 后生成，源码中不存在
    └── OpInterface.cpp                # ★ 自动生成的 dispatcher
```

## 4. YAML 配置文件详解

### 4.1 `op_plugin_functions.yaml` 格式

文件位置：`op_plugin/config/op_plugin_functions.yaml`（~7148 行）

```yaml
all_version: [v2.1, v2.2, v2.3, v2.4, v2.5, v2.6, v2.7, v2.8, v2.9, v2.10]

official:    # PyTorch 官方 aten 算子
  - func: mm(Tensor self, Tensor mat2) -> Tensor
    acl_op: all_version                    # ACLOP 实现（legacy）
    op_api: all_version                    # ACLNN 实现（当前）
    internal_format_opapi: all_version     # 支持内部 tensor 格式

  - func: acos(Tensor self) -> Tensor
    acl_op: all_version
    op_api: all_version
    gen_opapi:                             # 自动生成配置
      out:
        size: self
        dtype: '...'
      exec: aclnnAcos                      # ★ 指定 CANN 内核名

custom:      # torch_npu 自定义算子
  - func: npu_grouped_matmul_swiglu_quant(...) -> (Tensor, Tensor, Tensor)
    op_api: all_version

symint:      # SymInt 支持的算子（用于动态 shape）
  - func: zeros(SymInt[] size, ...) -> Tensor
    acl_op: [v2.1, newest]
```

### 4.2 关键字段说明

| 字段 | 含义 | 示例值 |
|------|------|--------|
| `func` | PyTorch 函数签名（ATen IR 格式） | `mm(Tensor self, Tensor mat2) -> Tensor` |
| `acl_op` | ACLOP 支持的版本范围 | `all_version`, `[v2.1, v2.5]`, `[v2.7, newest]` |
| `op_api` | ACLNN 支持的版本范围（优先使用） | 同上 |
| `gen_opapi.exec` | 指定的 CANN 内核名 | `aclnnAcos`, `aclnnConvolution` |
| `internal_format_opapi` | 是否支持 FRACTAL_NZ 等内部格式 | `all_version` |

### 4.3 版本范围语法

- `all_version`：所有版本支持
- `[v2.1, newest]`：v2.1 到最新版本
- `[v2.1, v2.5]`：仅 v2.1 到 v2.5

## 5. C++ 实现文件解读

### 5.1 Dispatcher（自动生成）

`codegen/gen.py` 根据 YAML 生成 `OpInterface.cpp`，dispatcher 模式：

```cpp
// 同时有 acl_op 和 op_api 的算子
at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2) {
    if (is_jit_disable) {
        return op_api::mm(self, mat2);   // 优先使用 ACLNN
    } else {
        return acl_op::mm(self, mat2);   // fallback 到 ACLOP
    }
}
```

### 5.2 ACLNN 实现（以 mm 为例）

文件：`op_plugin/ops/opapi/MmKernelNpuOpApi.cpp`

```cpp
namespace op_api {
at::Tensor mm(const at::Tensor &self, const at::Tensor &mat2) {
    // ...
    if (op_plugin::utils::is_nd_nz_format(self, mat2)) {
        // 权重为 NZ 格式时使用优化内核
        EXEC_NPU_CMD(aclnnMatmulWeightNz, self, mat2, result, cube_math_type);
    } else {
        // 标准格式
        EXEC_NPU_CMD(aclnnMm, self, mat2, result, cube_math_type);
    }
    return result;
}
}
```

**关键**：`EXEC_NPU_CMD` 的第一个参数就是实际的 **CANN 内核名**，也是 Profiling 里 Name 列的 `aclnn` 前缀部分。

### 5.3 aclnn 命名规则

```
aclnn<操作名>[<变体>]

aclnnMm                               # 标准矩阵乘
aclnnMatmulWeightNz                   # NZ 格式权重矩阵乘
aclnnAddRmsNorm                       # Add + RMSNorm 融合
aclnnFusedInferAttentionScoreV2/V3    # 融合 Attention（版本后缀常见）
aclnnGroupedMatmulSwigluQuantWeightNZ # 融合 GroupedMatmul+SwiGlu+Quant
```

## 6. 操作手册：正向映射（PyTorch → NPU Kernel）

给定一个 PyTorch 算子，如何找到它对应的 NPU 内核：

### Step 1: 在 YAML 中查找算子签名

```bash
# 搜索算子名
grep -n "func: mm\b" \
  /home/horacehxw/Projects/op-plugin/op_plugin/config/op_plugin_functions.yaml
```

输出示例：
```
3476:  - func: mm(Tensor self, Tensor mat2) -> Tensor
3481:  - func: mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
```

确认该算子有 `op_api` 字段 → 存在 ACLNN 实现。

### Step 2: 找到对应的实现文件

```bash
# 按算子名搜索实现文件
ls /home/horacehxw/Projects/op-plugin/op_plugin/ops/opapi/ | grep -i "mm\|matmul"
```

输出（部分）：
```
MmKernelNpuOpApi.cpp
MatmulKernelNpuOpApi.cpp
...  # grep -i 会匹配到更多文件，关注文件名直接对应的即可
```

### Step 3: 提取 CANN 内核名

```bash
# 搜索 EXEC_NPU_CMD 调用
grep "EXEC_NPU_CMD" \
  /home/horacehxw/Projects/op-plugin/op_plugin/ops/opapi/MmKernelNpuOpApi.cpp
```

输出：
```
EXEC_NPU_CMD(aclnnMatmulWeightNz, self, mat2, result, cube_math_type);
EXEC_NPU_CMD(aclnnMm, self, mat2, result, cube_math_type);
```

**结论**：`aten::mm.default` → 可能产生 `aclnnMm` 或 `aclnnMatmulWeightNz`（取决于权重格式）。

### Step 4: 确定 Profiling 中的 Type

在 `kernel_details.csv` 中：
- `aclnnMm` → Type = `MatMulV2`
- `aclnnMatmulWeightNz` → Type = `MatMulV2`

### 快速查询脚本

```bash
#!/bin/bash
# find_kernel.sh - 查找 PyTorch 算子对应的 NPU 内核
# 用法: ./find_kernel.sh <算子名>  例: ./find_kernel.sh mm

OP_PLUGIN_DIR="/home/horacehxw/Projects/op-plugin"
OP_NAME="$1"

echo "=== Step 1: YAML 配置 ==="
grep -n "func: ${OP_NAME}\b" \
  "${OP_PLUGIN_DIR}/op_plugin/config/op_plugin_functions.yaml" | head -5

echo ""
echo "=== Step 2: 实现文件 ==="
find "${OP_PLUGIN_DIR}/op_plugin/ops/opapi/" -iname "*${OP_NAME}*" 2>/dev/null

echo ""
echo "=== Step 3: CANN 内核 ==="
grep -r "EXEC_NPU_CMD" "${OP_PLUGIN_DIR}/op_plugin/ops/opapi/" 2>/dev/null \
  | grep -i "${OP_NAME}" | grep -o "EXEC_NPU_CMD([^,]*" | sort -u
```

## 7. 操作手册：反向映射（Profiling Kernel → PyTorch 算子）

给定 Profiling 中的一个内核，如何反向找到 PyTorch 算子：

### Step 1: 从 kernel_details.csv 提取内核信息

```csv
Name,Type,Input Shapes,Input Data Types,Duration(us)
aclnnMatmulWeightNz_MatMulCommon_MatMulV2,MatMulV2,"[1,4096,4096]","DT_BF16",123.45
```

- **Name** 的 `aclnn` 前缀 = `aclnnMatmulWeightNz`
- **Type** = `MatMulV2`

### Step 2: 在实现文件中搜索 aclnn 名称

```bash
# 搜索 aclnn 前缀
grep -r "aclnnMatmulWeightNz" \
  /home/horacehxw/Projects/op-plugin/op_plugin/ops/ --include="*.cpp" -l
```

输出：
```
ops/opapi/MmKernelNpuOpApi.cpp
ops/opapi/MatmulKernelNpuOpApi.cpp
```

### Step 3: 从实现文件确定函数名

```bash
# 查看函数定义
grep -B 10 "aclnnMatmulWeightNz" \
  /home/horacehxw/Projects/op-plugin/op_plugin/ops/opapi/MmKernelNpuOpApi.cpp \
  | grep "at::Tensor.*("
```

输出：
```
at::Tensor mm(const at::Tensor &self, const at::Tensor &mat2) {
```

→ 函数名为 `mm`

### Step 4: 在 YAML 中确认 ATen 签名

```bash
grep -A 3 "func: mm(" \
  /home/horacehxw/Projects/op-plugin/op_plugin/config/op_plugin_functions.yaml
```

**结论**：Profiling `MatMulV2` ← `aclnnMatmulWeightNz` ← `op_api::mm()` ← `aten::mm.default`

### 反向查询脚本

```bash
#!/bin/bash
# find_pytorch_op.sh - 从 Profiling 内核名反向查找 PyTorch 算子
# 用法: ./find_pytorch_op.sh <aclnn名称>  例: ./find_pytorch_op.sh aclnnMatmulWeightNz

OP_PLUGIN_DIR="/home/horacehxw/Projects/op-plugin"
KERNEL_NAME="$1"

echo "=== Step 1: 搜索实现文件 ==="
grep -r "${KERNEL_NAME}" "${OP_PLUGIN_DIR}/op_plugin/ops/" --include="*.cpp" -l

echo ""
echo "=== Step 2: 提取函数名（显示 aclnn 调用的上下文，手动确认所属函数）==="
for f in $(grep -r "${KERNEL_NAME}" "${OP_PLUGIN_DIR}/op_plugin/ops/opapi/" --include="*.cpp" -l); do
    echo "--- ${f} ---"
    # 显示 aclnn 调用前后的上下文，人工确认所属 namespace::function
    grep -B 20 "${KERNEL_NAME}" "$f" | grep -E "^(at::Tensor|void|std::tuple)" | tail -1
done

echo ""
echo "=== Step 3: YAML 签名 ==="
# 从文件名推断算子名
for f in $(grep -r "${KERNEL_NAME}" "${OP_PLUGIN_DIR}/op_plugin/ops/opapi/" --include="*.cpp" -l); do
    basename "$f" | sed 's/KernelNpuOpApi.cpp//' | sed 's/NpuOpapi.cpp//'
done
```

## 8. 常见算子映射速查表

下表列出 LLM 推理中常见的算子映射关系，来源于实际 Profiling 分析：

| PyTorch / TensorCast 算子 | op-plugin 函数 | aclnn 内核 | Profiling Type |
|---|---|---|---|
| `aten::mm.default` | `op_api::mm` | `aclnnMm`, `aclnnMatmulWeightNz` | `MatMulV2` |
| `aten::bmm.default` | `op_api::bmm` | `aclnnBatchMatMul` | `TransposeBatchMatMul` |
| `aten::addmm.default` | `op_api::addmm` | `aclnnAddmm` | `MatMulV2` |
| `torch_npu.npu_grouped_matmul` | `op_api::npu_grouped_matmul` | `aclnnGroupedMatmul` | `GroupedMatmul` |
| `torch_npu.npu_fused_infer_attention_score` | `op_api::npu_fused_infer_attention_score` | `aclnnFusedInferAttentionScoreV2/V3` | `FusedInferAttentionScore` |
| `torch_npu.npu_add_rms_norm` | `op_api::npu_add_rms_norm` | `aclnnAddRmsNorm` | `AddRmsNorm` |
| `torch_npu.npu_swiglu` | `op_api::npu_swiglu` | `aclnnSwiGlu` | `SwiGlu` |
| `torch_npu.npu_dynamic_quant` | `op_api::npu_dynamic_quant` | `aclnnDynamicQuant` | `DynamicQuant` |
| `torch_npu.npu_quantize` | `op_api::npu_quantize` | `aclnnAscendQuant/V3` | `AscendQuantV2` |
| `torch_npu.npu_weight_quant_batchmatmul` | `op_api::npu_weight_quant_batchmatmul` | `aclnnWeightQuantBatchMatmulV2/V3` | `QuantBatchMatmulV3` |
| `torch.distributed.all_reduce` | HCCL | — | `hcom_allReduce_` |
| `torch.distributed.all_gather` | HCCL | — | `HcomAllGather` |
| `torch.distributed.all_to_all` | HCCL | — | `hcom_alltoall_` |

## 9. 与 TensorCast op_mapping.yaml 的关系

`op_mapping.yaml`（见 [`examples/op_mapping_example.yaml`](../examples/op_mapping_example.yaml)）是 TensorCast 用于匹配 Profiling 数据的配置文件，其核心是将 TensorCast 虚拟算子映射到 Profiling 的 **Type 列值**：

```yaml
operator_mappings:
  "aten.mm.default":
    kernel_type: MatMulV2           # ← Profiling Type 列的值
  "tensor_cast.static_quant_linear.default":
    kernel_type: QuantBatchMatmulV3
  "tensor_cast.attention.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
```

**编写流程**：

1. 确定 TensorCast 算子名（如 `tensor_cast.static_quant_linear.default`）
2. 分析该算子在实际 vLLM 推理中对应的 NPU 操作（参考第 8 节速查表）
3. 从 Profiling 数据确认 Type 列值
4. 写入 `op_mapping.yaml`

## 10. 进阶：处理融合算子

### 10.1 一对多映射

某些 TensorCast 算子对应多个 NPU 内核：

```yaml
"tensor_cast.multihead_latent_attention.default":
  composite: true
  sub_kernels: [TransposeBatchMatMul, FusedInferAttentionScore]
```

这种情况需要 TensorCast 的分解 pass 将复合算子拆成子算子后再逐个查询。

### 10.2 多对一映射

多个 PyTorch 算子融合成一个 NPU 内核（如 `DequantSwigluQuant` = dequant + swiglu + quant）：

- TensorCast 需要对应的融合 pass（见 `tensor_cast/compilation/passes/`）
- 或在 `op_mapping.yaml` 中手动处理

### 10.3 查找自定义/融合算子

```bash
# 搜索自定义融合算子的实现
grep -r "grouped_matmul_swiglu" \
  /home/horacehxw/Projects/op-plugin/op_plugin/ops/opapi/ --include="*.cpp" -l

# 查看其 CANN 内核名
grep "EXEC_NPU_CMD" \
  /home/horacehxw/Projects/op-plugin/op_plugin/ops/opapi/GroupedMatmulSwigluQuantNpuOpapi.cpp
```

## 11. 实战示例：从零建立 Qwen3-30B Prefill 的映射

### Step 1: 分析 Profiling 数据

```bash
# 查看 Profiling 中的 Top 算子
# 文件: /mnt/d/Data/Profiling/comparison_qwen3_30b/kernel_details.csv
# 主要 Type: TensorMove(386), hcom_allReduce_(276), MatMulV2(275),
#           AddRmsNorm(131), FusedInferAttentionScore(67), SwiGlu(67)
```

### Step 2: 逐个建立映射

| Profiling Type | 反向查找 aclnn | PyTorch 算子 | TensorCast 映射 |
|---|---|---|---|
| `MatMulV2` | `aclnnMm` | `aten::mm.default` | `aten.mm.default` |
| `AddRmsNorm` | `aclnnAddRmsNorm` | `torch_npu.npu_add_rms_norm` | `tensor_cast.add_rms_norm.default` |
| `FusedInferAttentionScore` | `aclnnFusedInferAttentionScore` | `torch_npu.npu_fused_infer_attention_score` | `tensor_cast.attention.default` |
| `SwiGlu` | `aclnnSwiGlu` | `torch_npu.npu_swiglu` | `tensor_cast.swiglu.default` |
| `hcom_allReduce_` | HCCL | `torch.distributed.all_reduce` | `tensor_cast.all_reduce.default` |

### Step 3: 写入 op_mapping.yaml

```yaml
version: "0.13.0"
device: ATLAS_800_A3_752T_128G_DIE
operator_mappings:
  "aten.mm.default": {kernel_type: MatMulV2}
  "tensor_cast.add_rms_norm.default": {kernel_type: AddRmsNorm}
  "tensor_cast.attention.default": {kernel_type: FusedInferAttentionScore, query_mode: attention_special}
  "tensor_cast.swiglu.default": {kernel_type: SwiGlu}
  "tensor_cast.all_reduce.default": {kernel_type: hcom_allReduce_, category: communication}
```

## 12. 附录：op-plugin 搜索技巧

### 按 gen_opapi.exec 搜索（YAML 中指定了 aclnn 名的算子）

```bash
grep -B 3 "exec: aclnn" \
  /home/horacehxw/Projects/op-plugin/op_plugin/config/op_plugin_functions.yaml \
  | grep -E "func:|exec:"
```

### 统计所有 aclnn 内核

```bash
grep -roh "EXEC_NPU_CMD(aclnn[^,]*" \
  /home/horacehxw/Projects/op-plugin/op_plugin/ops/opapi/ \
  | sed 's/EXEC_NPU_CMD(//' | sort -u | wc -l
```

### 查看 custom 算子（torch_npu 专有）

```bash
grep -A 2 "^custom:" \
  /home/horacehxw/Projects/op-plugin/op_plugin/config/op_plugin_functions.yaml
# 然后搜索具体的 npu_ 前缀算子
grep "func: npu_" \
  /home/horacehxw/Projects/op-plugin/op_plugin/config/op_plugin_functions.yaml | wc -l
```

## 13. 相关文档

- 设计文档：[`OPERATOR_PERF_DATABASE_DESIGN_zh_v1.2.md`](../OPERATOR_PERF_DATABASE_DESIGN_zh_v1.2.md)
- op_mapping 示例：[`examples/op_mapping_example.yaml`](../examples/op_mapping_example.yaml)
- comm_config 示例：[`examples/comm_config_example.yaml`](../examples/comm_config_example.yaml)
- op-plugin 源码：`/home/horacehxw/Projects/op-plugin/`
- op-plugin 配置文档：`/home/horacehxw/Projects/op-plugin/op_plugin/config/README.md`
