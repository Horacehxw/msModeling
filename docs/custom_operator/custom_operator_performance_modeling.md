# Custom Operator Performance Modeling

## Code Organization

**Place custom code in the `tensor_cast/performance_model/custom_op` directory**

Please store your custom performance modeling function implementations in this specified directory and ensure proper import paths are configured.

## Overview

Two different decorators are used for performance modeling:

### `@OpInvokeInfo.register_op_properties`
For capturing computational complexity and memory access patterns of regular operators. It analyzes the operator's computation cost (in operations/second) and memory traffic (in bytes).

### `@register_op_estimator`
For estimating direct execution time of distributed communication operators. It returns the estimated time (in seconds) for operations like all_reduce, all_gather, etc.

## `@OpInvokeInfo.register_op_properties` Template

Use this to capture computational properties:

```python
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
import torch

@OpInvokeInfo.register_op_properties(torch.ops.your_op.Operator)
def _(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    properties = op_invoke_info.get_memory_access_properties()
    
    # Add computation by data type
    compute_ops = properties.compute_ops.setdefault(
        op_invoke_info.args[0].dtype, OpInvokeInfo.ComputeOps())
    compute_ops.mma_ops = calculated_ops
    
    return properties
```

## Parameter Mapping

- `op_invoke_info.args[0]`: First argument (e.g., key tensor)
- `op_invoke_info.args[1]`: Second argument (e.g., value tensor)
- `op_invoke_info.args[2]`: Third argument (e.g., kv_cache tensor)
- `op_invoke_info.args[3]`: Fourth argument (e.g., slot_mapping)

## Example

```python
@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.reshape_and_cache)
def _(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    key, value, kv_cache, _ = op_invoke_info.args
    properties = op_invoke_info.get_memory_access_properties()
    
    # Update memory properties
    properties.memory_read_bytes += key.numel() * key.element_size()
    properties.memory_write_bytes += kv_cache.numel() * kv_cache.element_size()
    
    # Set computation operations
    compute_ops = properties.compute_ops.setdefault(key.dtype, OpInvokeInfo.ComputeOps())
    compute_ops.mma_ops = key.numel() + value.numel()
    
    return properties
```

## `@register_op_estimator` Template

Use this to estimate execution time of communication operations:

```python
from tensor_cast.performance_model.op_estimator_registry import register_op_estimator
from tensor_cast.performance_model.model import PerformanceModel

@register_op_estimator(torch.ops.tensor_cast.all_to_all.default, None, True)
def _estimate_all_to_all(op_invoke_info, device_profile) -> object:
    input_tensor = op_invoke_info.args[0]
    message_size = input_tensor.numel() * input_tensor.element_size()
    
    # Simple communication time estimate
    return PerformanceModel.Result(0.001 + message_size / (10.0 * 1e9))
```

### Common Communication Operators:
- `torch.ops.tensor_cast.all_reduce.default`
- `torch.ops.tensor_cast.all_gather.default`
- `torch.ops.tensor_cast.reduce_scatter.default`
- `torch.ops.tensor_cast.all_to_all.default`

## Summary

- Use `@OpInvokeInfo.register_op_properties` for regular operators to capture computation cost and memory access
- Use `@register_op_estimator` for distributed communication operations to estimate direct execution time
- They are unrelated and serve different purposes in the performance modeling framework

### Parameters for `@register_op_estimator`:
1. **Operator**: The PyTorch communication operator to estimate
2. **Device Profile**: Device-specific configuration (can be `None`)
3. **Override**: Whether to allow override of existing estimators