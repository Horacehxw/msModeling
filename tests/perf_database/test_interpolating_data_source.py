"""Tests for InterpolatingDataSource."""

from unittest.mock import MagicMock

import pytest
import torch

from tensor_cast.performance_model.perf_database.data_source import QuerySource
from tensor_cast.performance_model.perf_database.interpolating_data_source import (
    InterpolatingDataSource,
)
from tensor_cast.performance_model.perf_database.profiling_data_source import (
    ProfilingDataSource,
)


def _make_op_info(func, input_tensors):
    mock = MagicMock()
    mock.func = func
    mock.args = tuple(input_tensors)
    mock.kwargs = {}
    mock.out = None
    return mock


# --- Fixtures ---

INTERP_COMPUTE_MAPPING = """\
version: "test"
device: TEST_DEVICE
interpolation_policy:
  default_method: linear
  kernel_overrides:
    FusedInferAttentionScore:
      shape_transform: sqrt
operator_mappings:
  "aten.mm.default":
    kernel_type: MatMulV2
  "tensor_cast.all_reduce.default":
    kernel_type: hcom_allReduce_
    category: communication
  "tensor_cast.attention.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
"""

# MatMulV2 CSV with multiple seq lengths for interpolation
INTERP_MATMUL_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"100,512;512,1024","DT_BF16;DT_BF16","ND;ND","100,1024","DT_BF16","ND",10.0
"200,512;512,1024","DT_BF16;DT_BF16","ND;ND","200,1024","DT_BF16","ND",20.0
"400,512;512,1024","DT_BF16;DT_BF16","ND;ND","400,1024","DT_BF16","ND",40.0"""

INTERP_COMM_CSV = """\
message_bytes,num_devices,dtype,topology_tier,Duration(us)
100000,16,DT_BF16,0,100.0
200000,16,DT_BF16,0,200.0
400000,16,DT_BF16,0,400.0"""

INTERP_FIA_CSV = """\
batch_size,avg_seq_len,num_heads,head_dim,dtype,Duration(us)
1,1000,4,128,DT_BF16,100.0
1,4000,4,128,DT_BF16,1600.0"""


@pytest.fixture
def interp_data_dir(tmp_path):
    data_dir = tmp_path / "interp"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(INTERP_COMPUTE_MAPPING)
    (data_dir / "MatMulV2.csv").write_text(INTERP_MATMUL_CSV.strip())
    (data_dir / "hcom_allReduce_.csv").write_text(INTERP_COMM_CSV.strip())
    (data_dir / "FusedInferAttentionScore.csv").write_text(INTERP_FIA_CSV.strip())
    return data_dir


# --- Tests ---


def test_exact_match_passthrough(interp_data_dir):
    """Exact match should pass through from base."""
    base = ProfilingDataSource(interp_data_dir)
    ds = InterpolatingDataSource(base)
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(100, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 10.0) < 0.01
    assert result.source == QuerySource.MEASURED


def test_compute_interpolation_midpoint(interp_data_dir):
    """seq=150 between 100 and 200 should interpolate to ~15.0 us."""
    base = ProfilingDataSource(interp_data_dir)
    ds = InterpolatingDataSource(base)
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(150, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should interpolate between seq=100 and seq=200"
    assert abs(result.latency_us - 15.0) < 0.5
    assert result.source == QuerySource.INTERPOLATED


def test_compute_interpolation_quarter(interp_data_dir):
    """seq=300 between 200 and 400 should interpolate to ~30.0 us."""
    base = ProfilingDataSource(interp_data_dir)
    ds = InterpolatingDataSource(base)
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(300, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 30.0) < 0.5
    assert result.source == QuerySource.INTERPOLATED


def test_compute_no_interpolation_wrong_weight(interp_data_dir):
    """Different weight shape (not just seq dim) should not interpolate."""
    base = ProfilingDataSource(interp_data_dir)
    ds = InterpolatingDataSource(base)
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(150, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(512, 2048, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None, "Can't interpolate when non-seq dims differ"


def test_comm_interpolation(interp_data_dir):
    """Communication: interpolate by message_bytes."""
    base = ProfilingDataSource(interp_data_dir)
    ds = InterpolatingDataSource(base)
    # 150000 bytes between 100000 and 200000 -> 150.0 us
    # Need tensor with 150000 / 2 = 75000 elements (BF16 = 2 bytes)
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(75000, device="meta", dtype=torch.bfloat16),
            0,
            list(range(16)),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should interpolate comm by message_bytes"
    assert abs(result.latency_us - 150.0) < 1.0
    assert result.source == QuerySource.INTERPOLATED


def test_attention_interpolation_sqrt(interp_data_dir):
    """Attention: interpolate avg_seq_len with sqrt transform.

    CSV has: seq=1000 -> 100us, seq=4000 -> 1600us
    sqrt(1000)=31.62, sqrt(4000)=63.25
    For seq=2000: sqrt(2000)=44.72
    In sqrt space: t = (44.72 - 31.62) / (63.25 - 31.62) = 0.4142
    sqrt_interp = 100 + 0.4142 * (1600 - 100) = 721.3
    """
    base = ProfilingDataSource(interp_data_dir)
    ds = InterpolatingDataSource(base)
    op = _make_op_info(
        torch.ops.tensor_cast.attention.default,
        [
            torch.empty(1, 512, device="meta", dtype=torch.bfloat16),  # query
            torch.empty(
                16, 128, 4, 128, device="meta", dtype=torch.bfloat16
            ),  # key
            torch.empty(
                16, 128, 4, 128, device="meta", dtype=torch.bfloat16
            ),  # value
            None,
            None,
            None,
            torch.tensor([2000], dtype=torch.int64),  # seq_lens (CPU)
            torch.tensor([1], dtype=torch.int64),  # query_lens
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should interpolate attention with sqrt transform"
    assert result.source == QuerySource.INTERPOLATED
    # With sqrt transform, expect ~721 (not 600 from linear)
    assert (
        680.0 < result.latency_us < 760.0
    ), f"Expected ~721 with sqrt, got {result.latency_us}"


def test_unmapped_op_no_interpolation(interp_data_dir):
    """Unmapped ops should still return None."""
    base = ProfilingDataSource(interp_data_dir)
    ds = InterpolatingDataSource(base)
    op = _make_op_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(100, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(100, 512, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None
