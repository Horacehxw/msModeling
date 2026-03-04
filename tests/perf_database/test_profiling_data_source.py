from unittest.mock import MagicMock

import pytest
import torch

from tensor_cast.performance_model.perf_database.data_source import QuerySource

from tensor_cast.performance_model.perf_database.profiling_data_source import (
    DTYPE_MAP,
    fractal_nz_to_nd,
    ProfilingDataSource,
)


# --- fractal_nz_to_nd tests (design doc S4.9, Appendix B) ---


def test_fractal_nz_to_nd_bf16():
    # BF16 MatMulV2: [320,48,16,16] -> K=320*16=5120, N=48*16=768
    assert fractal_nz_to_nd((320, 48, 16, 16)) == (5120, 768)


def test_fractal_nz_to_nd_int8():
    # INT8 QuantBatchMatmulV3: [N/32, K/16, 16, 32]
    # H=48, W=448, block_h=16, block_w=32 -> (H*block_w, W*block_h) = (1536, 7168)
    assert fractal_nz_to_nd((48, 448, 16, 32)) == (1536, 7168)


def test_fractal_nz_to_nd_batched():
    # GroupedMatmul INT8: [E, N/32, K/16, 16, 32]
    # batch=64, H=48, W=448, block_h=16, block_w=32 -> (64, 1536, 7168)
    assert fractal_nz_to_nd((64, 48, 448, 16, 32)) == (64, 1536, 7168)


def test_dtype_map():
    assert DTYPE_MAP[torch.bfloat16] == "DT_BF16"
    assert DTYPE_MAP[torch.float16] == "DT_BF16"
    assert DTYPE_MAP[torch.int8] == "INT8"
    assert DTYPE_MAP[torch.float32] == "FLOAT"


# --- ProfilingDataSource tests ---

SPIKE_OP_MAPPING_YAML = """
version: "0.14.0"
device: TEST_DEVICE

operator_mappings:
  "aten.mm.default":
    kernel_type: MatMulV2
  "aten.bmm.default":
    kernel_type: TransposeBatchMatMul
  "tensor_cast.attention.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
  "tensor_cast.multihead_latent_attention.default":
    composite: true
    sub_kernels: [TransposeBatchMatMul, FusedInferAttentionScore]
  "tensor_cast.all_reduce.default":
    kernel_type: hcom_allReduce_
    category: communication
"""

# CSV with FRACTAL_NZ weight, matching design doc S4.6 query example
SPIKE_MATMUL_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"136,5120;320,48,16,16","DT_BF16;DT_BF16","ND;FRACTAL_NZ","136,768","DT_BF16","ND",45.3
"1,5120;320,48,16,16","DT_BF16;DT_BF16","ND;FRACTAL_NZ","1,768","DT_BF16","ND",12.1
"""


@pytest.fixture
def spike_data_dir(tmp_path):
    data_dir = tmp_path / "spike"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(SPIKE_OP_MAPPING_YAML)
    (data_dir / "MatMulV2.csv").write_text(SPIKE_MATMUL_CSV.strip())
    return data_dir


def _make_op_info(func, input_tensors, output_tensors=None):
    """Create a mock OpInvokeInfo with real torch.ops func and meta tensors."""
    mock = MagicMock()
    mock.func = func
    mock.args = tuple(input_tensors)
    mock.kwargs = {}
    if output_tensors:
        mock.out = (
            output_tensors[0] if len(output_tensors) == 1 else tuple(output_tensors)
        )
    else:
        mock.out = None
    return mock


def test_exact_match_with_fractal_nz(spike_data_dir):
    """Design doc S4.6: aten.mm(A[136,5120], B[5120,768]) matches
    CSV row with FRACTAL_NZ weight [320,48,16,16] after restoration."""
    ds = ProfilingDataSource(spike_data_dir)
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(136, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, 768, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 45.3) < 0.01
    assert result.confidence == 1.0
    assert result.source == QuerySource.MEASURED
    assert result.details.get("kernel_type") == "MatMulV2"


def test_miss_wrong_shape(spike_data_dir):
    ds = ProfilingDataSource(spike_data_dir)
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(256, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, 768, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None


def test_miss_unmapped_op(spike_data_dir):
    ds = ProfilingDataSource(spike_data_dir)
    op = _make_op_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(136, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(136, 5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None


def test_composite_returns_none(spike_data_dir):
    """Composite ops (MLA) return None in spike, fallback to analytic."""
    ds = ProfilingDataSource(spike_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.multihead_latent_attention.default,
        [torch.empty(136, 5120, device="meta", dtype=torch.bfloat16)],
    )
    result = ds.lookup(op)
    assert result is None


def test_communication_returns_none(spike_data_dir):
    """Communication ops return None in spike, fallback to analytic."""
    ds = ProfilingDataSource(spike_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [torch.empty(136, 5120, device="meta", dtype=torch.bfloat16), 0, [0, 1]],
    )
    result = ds.lookup(op)
    assert result is None


def test_attention_special_returns_none(spike_data_dir):
    """attention_special ops return None in spike, fallback to analytic."""
    ds = ProfilingDataSource(spike_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.attention.default,
        [torch.empty(136, 5120, device="meta", dtype=torch.bfloat16)],
    )
    result = ds.lookup(op)
    assert result is None
