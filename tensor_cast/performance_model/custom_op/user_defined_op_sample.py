'''
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo
import torch


@OpInvokeInfo.register_op_properties(torch.ops.your_op.Operator)
def _(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    """
    Performance modeling template for custom operators.

    Parameters:
    - op_invoke_info: Contains operator inputs, kwargs, and metadata

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
