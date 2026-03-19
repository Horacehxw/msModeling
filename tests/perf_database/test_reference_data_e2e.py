"""E2E tests against CANN 8.5 reference profiling data directory.

Validates that ProfilingDataSource correctly matches TC ops against
real NPU profiling CSVs in the CANN 8.5 data directory.
Reference: Qwen3-32B BF16 TP=16, vllm 0.15.0 / torch 2.9.0 / CANN 8.5.
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock

from tensor_cast.performance_model.perf_database.profiling_data_source import (
    ProfilingDataSource,
)
from tensor_cast.performance_model.perf_database.data_source import QuerySource

CANN85_DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "tensor_cast/performance_model/perf_database/data"
    / "ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5"
)


def _make_op(func, *tensor_specs):
    """Create mock OpInvokeInfo with given func and tensor args.

    tensor_specs: tuples of (shape, dtype) or just shape (defaults to bf16).
    """
    mock = MagicMock()
    mock.func = func
    args = []
    for spec in tensor_specs:
        if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[1], torch.dtype):
            shape, dtype = spec
        else:
            shape, dtype = spec, torch.bfloat16
        args.append(torch.empty(*shape, dtype=dtype, device="meta"))
    mock.args = tuple(args)
    return mock


@pytest.fixture(scope="module")
def ds():
    """ProfilingDataSource with CANN 8.5 reference data directory."""
    if not CANN85_DATA_DIR.exists():
        pytest.skip("CANN 8.5 data directory not found")
    return ProfilingDataSource(CANN85_DATA_DIR)


# === Zero-cost ops ===


@pytest.mark.parametrize(
    "func",
    [
        torch.ops.aten.view.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.split.Tensor,
        torch.ops.aten.split_with_sizes.default,
        torch.ops.aten.select.int,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.alias.default,
    ],
)
def test_zero_cost_ops(ds, func):
    """Zero-cost metadata ops should return QueryResult with latency=0."""
    op = _make_op(func, (336, 5120))
    result = ds.lookup(op)
    assert result is not None
    assert result.latency_us == 0.0
    assert result.details.get("zero_cost") is True


# === Compute ops with matching shapes (from CANN 8.5 CSV data) ===


def test_mm_hit(ds):
    """aten.mm: TC (336,5120)×(5120,3200) matches MatMulV2 with FRACTAL_NZ weight."""
    # CSV has "336,5120;320,200,16,16" (FRACTAL_NZ) = (5120,3200) in ND
    op = _make_op(torch.ops.aten.mm.default, (336, 5120), (5120, 3200))
    result = ds.lookup(op)
    assert result is not None
    assert result.latency_us > 0
    assert result.details["kernel_type"] == "MatMulV2"


def test_swiglu_hit(ds):
    """SwiGlu: TC 2×(1,336,1600) merged→(336,3200) matches CSV (336,3200)."""
    op = _make_op(
        torch.ops.tensor_cast.swiglu.default, (1, 336, 1600), (1, 336, 1600)
    )
    result = ds.lookup(op)
    assert result is not None
    assert result.latency_us > 0
    assert result.details["kernel_type"] == "SwiGlu"


def test_rmsnorm_hidden_hit(ds):
    """RmsNorm: TC (1,336,5120),(5120,) → strip batch → matches CSV (336,5120;5120)."""
    op = _make_op(
        torch.ops.tensor_cast.rms_norm.default, (1, 336, 5120), (5120,)
    )
    result = ds.lookup(op)
    assert result is not None
    assert result.latency_us > 0
    assert result.details["kernel_type"] == "RmsNorm"


def test_rmsnorm_head_hit(ds):
    """RmsNorm: TC (1,336,4,128),(128,) matches CSV (336,4,128;128)."""
    op = _make_op(
        torch.ops.tensor_cast.rms_norm.default, (1, 336, 4, 128), (128,)
    )
    result = ds.lookup(op)
    assert result is not None
    assert result.details["kernel_type"] == "RmsNorm"


def test_rope_miss_shape_mismatch(ds):
    """RoPE: TC dispatches [K(1,1,S,128), Q(1,4,S,128), cos, sin] → _triton_rope.

    CSV has 3 inputs (Q,K,cos_cache) while TC sends 4 (K,Q,cos,sin).
    With tc_input_count=2 only first 2 TC inputs are matched, but CSV
    still has 3 inputs → shape_mismatch. This is a known gap.
    """
    op = _make_op(
        torch.ops.tensor_cast.apply_rope.default,
        (1, 1, 336, 128),
        (1, 4, 336, 128),
        (1, 336, 128),
        (1, 336, 128),
    )
    result = ds.lookup(op)
    assert result is None


def test_add_elementwise_hit(ds):
    """Add (elementwise query_mode): TC (1,7168),(7168) matches CSV (1,7168;7168)."""
    op = _make_op(torch.ops.aten.add.Tensor, (1, 7168), (7168,))
    result = ds.lookup(op)
    assert result is not None
    assert result.details["kernel_type"] == "Add"


def test_matmul_all_reduce_composite_miss(ds):
    """Composite: matmul_all_reduce — shapes not in CSV → graceful None."""
    op = _make_op(
        torch.ops.tensor_cast.matmul_all_reduce.default,
        (144, 512),
        (512, 5120),
    )
    # Add rank_group so _lookup_comm_for_composite can extract it
    op.args = (*op.args, None, 0, [0, 1])
    result = ds.lookup(op)
    assert result is None


# === Known MISSes (structural mismatches) ===


def test_attention_miss(ds):
    """Attention: attention_special query_mode → graceful None (not crash)."""
    op = MagicMock()
    op.func = torch.ops.tensor_cast.attention.default
    op.args = (
        torch.empty(336, 512, dtype=torch.bfloat16, device="meta"),
        torch.empty(2, 128, 1, 128, dtype=torch.bfloat16, device="meta"),
        torch.empty(2, 128, 1, 128, dtype=torch.bfloat16, device="meta"),
        torch.empty(2, 1, dtype=torch.long, device="meta"),
        torch.empty(3, dtype=torch.long, device="meta"),
        torch.empty(2, dtype=torch.long, device="meta"),
        torch.empty(2, dtype=torch.long, device="meta"),
    )
    result = ds.lookup(op)
    assert result is None


def test_comm_op_interpolated(ds):
    """Communication ops use message_bytes interpolation in CANN 8.5 data."""
    op = MagicMock()
    op.func = torch.ops.tensor_cast.all_reduce.default
    op.args = (
        torch.empty(336, 5120, dtype=torch.bfloat16, device="meta"),
        list(range(16)),
    )
    result = ds.lookup(op)
    # CANN 8.5 has proper comm data → interpolation returns a result
    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    assert result.details["kernel_type"] == "hcom_allReduce_"


def test_embedding_miss_input_count(ds):
    """Embedding: TC sends 2 tensors, GatherV2 CSV has 3 → miss."""
    op = _make_op(
        torch.ops.aten.embedding.default,
        (151936, 5120),
        ((1, 336), torch.long),
    )
    result = ds.lookup(op)
    assert result is None


def test_copy_miss_shape_mismatch(ds):
    """copy_: mapped to TensorMove but KV cache shape doesn't match CSV."""
    op = _make_op(
        torch.ops.aten.copy_.default,
        (2, 2, 128, 1, 128),
        (2, 2, 128, 1, 128),
    )
    result = ds.lookup(op)
    assert result is None
