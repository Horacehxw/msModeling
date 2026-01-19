'''
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
import torch


@OpInvokeInfo.register_op_properties(torch.ops.your_op.Operator)
def _(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    """
    Performance modeling template for custom operators.

    Parameters:
    - op_invoke_info: Contains operator inputs, kwargs, and metadata

    op_invoke_info.args Parameter Mapping:
    - op_invoke_info.args[0]: First argument from operator call (e.g., key tensor)
    - op_invoke_info.args[1]: Second argument from operator call (e.g., value tensor)
    - op_invoke_info.args[2]: Third argument from operator call (e.g., kv_cache tensor)
    - op_invoke_info.args[3]: Fourth argument from operator call (e.g., slot_mapping)

    Example from reshape_and_cache operator:
    ```python
    # Operator call:
    # torch.ops.tensor_cast.reshape_and_cache(key, value, kv_cache, attention_meta.slot_mapping)
    #
    # In performance modeling function:
    assert len(op_invoke_info.args) == 4
    key = op_invoke_info.args[0]      # Corresponds to 'key' in operator call
    value = op_invoke_info.args[1]    # Corresponds to 'value' in operator call
    kv_cache = op_invoke_info.args[2] # Corresponds to 'kv_cache' in operator call
    # op_invoke_info.args[3] would be 'attention_meta.slot_mapping'
    ```

    Tensor Shape Access Examples:
    - Basic tensor: `tensor = op_invoke_info.args[0]  # Get first argument`
    - Shape info: `shape = tensor.shape` or `dims = tensor.size()`
    - Specific dimension: `batch = tensor.size(0)`, `height = tensor.size(1)`, `width = tensor.size(2)`
    - Last dimension: `last_dim = tensor.size(-1)`
    - Number of dimensions: `ndim = tensor.ndim`
    - Total elements: `numel = tensor.numel()`

    Common patterns from existing ops:
    ```python
    # Matrix multiplication example (bmm op)
    mat1, mat2 = op_invoke_info.args[0], op_invoke_info.args[1]
    b, m, k = mat1.size(0), mat1.size(1), mat1.size(2)  # Batch, rows, inner_dim
    _, _, n = mat2.size(2)  # Output cols
    ```

    Properties attributes:

    compute_ops: Dict[torch.dtype, ComputeOps]
       - Stores computation cost per data type
       - Create: properties.compute_ops.setdefault(dtype, OpInvokeInfo.ComputeOps())
       - Matrix ops: compute_ops.mma_ops = calculated_ops
       - Scalar ops: compute_ops.gp_ops = calculated_ops

    memory access bytes:
       - Base: properties = op_invoke_info.get_memory_access_properties()
       - Extended: properties.memory_read_bytes += extra_bytes
       - Exclude: get_memory_access_properties(exclude_input_ids={idx})

    Computation formulas:
       - Linear ops: input_size * output_size * ops_per_element
       - Element-wise: total_elements * 2 (multiply + add)
       - Reduction: input_elements * log2(input_elements)
    """
    # 1. Validate inputs
    assert len(op_invoke_info.args) >= expected_count, "Error message"
    input1, input2 = op_invoke_info.args[0], op_invoke_info.args[1]

    # 2. Get base memory properties
    properties = op_invoke_info.get_memory_access_properties()

    # 3. Calculate compute operations by data type
    compute_ops = properties.compute_ops.setdefault(input1.dtype, OpInvokeInfo.ComputeOps())
    compute_ops.mma_ops = matrix_ops  # Heavy computation (matrix ops)

    # Optional: Add scalar operations for different dtype
    # compute_ops = properties.compute_ops.setdefault(scalar_dtype, OpInvokeInfo.ComputeOps())
    # compute_ops.gp_ops = scalar_ops  # Element-wise computation

    # 4. Optional: Add extra memory for special data structures
    # properties.memory_read_bytes += extra_memory_bytes

    return properties  # Returns complete PerformanceProperties
'''