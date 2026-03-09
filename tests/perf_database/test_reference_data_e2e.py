"""E2E tests against reference profiling data directory.

Validates that ProfilingDataSource correctly matches TC ops against
real NPU profiling CSVs in the reference data/ directory.
Reference: Qwen3-32B BF16 TP=16, tokens=144 (block-pads to match CSV 136).
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock

from tensor_cast.performance_model.perf_database.profiling_data_source import (
    ProfilingDataSource,
)
from tensor_cast.performance_model.perf_database.data_source import QuerySource


REFERENCE_DATA_DIR = Path(__file__).parent.parent.parent / (
    "tensor_cast/performance_model/perf_database/data/"
    "ATLAS_800_A3_752T_128G_DIE/vllm_ascend/v0.13.0"
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
    """ProfilingDataSource with reference data directory."""
    if not REFERENCE_DATA_DIR.exists():
        pytest.skip("Reference data directory not found")
    return ProfilingDataSource(REFERENCE_DATA_DIR)


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
    op = _make_op(func, (144, 5120))
    result = ds.lookup(op)
    assert result is not None
    assert result.latency_us == 0.0
    assert result.details.get("zero_cost") is True


# === Compute ops with matching shapes (block-padding tolerance) ===


def test_mm_qkv_hit(ds):
    """aten.mm (qkv_proj): TC (144,5120)×(5120,768) matches MatMulV2 FRACTAL_NZ."""
    op = _make_op(torch.ops.aten.mm.default, (144, 5120), (5120, 768))
    result = ds.lookup(op)
    assert result is not None
    assert result.latency_us > 0
    assert result.details["kernel_type"] == "MatMulV2"


def test_swiglu_hit(ds):
    """SwiGlu: TC 2×(1,144,1600) merged→(144,3200) matches CSV (136,3200)."""
    op = _make_op(
        torch.ops.tensor_cast.swiglu.default, (1, 144, 1600), (1, 144, 1600)
    )
    result = ds.lookup(op)
    assert result is not None
    assert result.latency_us > 0
    assert result.details["kernel_type"] == "SwiGlu"


def test_rmsnorm_hidden_hit(ds):
    """RmsNorm: TC (1,144,5120),(5120,) matches CSV (136,5120),(5120,)."""
    op = _make_op(
        torch.ops.tensor_cast.rms_norm.default, (1, 144, 5120), (5120,)
    )
    result = ds.lookup(op)
    assert result is not None
    assert result.latency_us > 0
    assert result.details["kernel_type"] == "RmsNorm"


def test_rmsnorm_head_hit(ds):
    """RmsNorm: TC (1,144,4,128),(128,) matches CSV (136,4,128),(128,)."""
    op = _make_op(
        torch.ops.tensor_cast.rms_norm.default, (1, 144, 4, 128), (128,)
    )
    result = ds.lookup(op)
    assert result is not None
    assert result.details["kernel_type"] == "RmsNorm"


def test_rope_hit(ds):
    """RoPE: TC dispatches [K(1,1,S,128), Q(1,4,S,128), cos, sin] → ApplyRotaryPosEmb."""
    op = _make_op(
        torch.ops.tensor_cast.apply_rope.default,
        (1, 1, 144, 128),
        (1, 4, 144, 128),
        (1, 144, 128),
        (1, 144, 128),
    )
    result = ds.lookup(op)
    assert result is not None
    assert result.details["kernel_type"] == "ApplyRotaryPosEmb"


def test_add_hit(ds):
    """Add: TC (144,5120),(144,5120) matches CSV (136,5120),(136,5120)."""
    op = _make_op(torch.ops.aten.add.Tensor, (144, 5120), (144, 5120))
    result = ds.lookup(op)
    assert result is not None
    assert result.details["kernel_type"] == "Add"


def test_matmul_all_reduce_composite_hit(ds):
    """Composite: matmul_all_reduce decomposes to MatMulV2 sub-kernel."""
    op = _make_op(
        torch.ops.tensor_cast.matmul_all_reduce.default,
        (144, 512),
        (512, 5120),
    )
    result = ds.lookup(op)
    assert result is not None
    assert result.details.get("composite") is True
    assert result.details["kernel_type"] == "MatMulV2"


# === Known MISSes (structural mismatches) ===


def test_attention_miss_raw_csv(ds):
    """Attention: raw CSV format → graceful None (not crash)."""
    op = MagicMock()
    op.func = torch.ops.tensor_cast.attention.default
    op.args = (
        torch.empty(144, 512, dtype=torch.bfloat16, device="meta"),
        torch.empty(2, 128, 1, 128, dtype=torch.bfloat16, device="meta"),
        torch.empty(2, 128, 1, 128, dtype=torch.bfloat16, device="meta"),
        torch.empty(2, 1, dtype=torch.long, device="meta"),
        torch.empty(3, dtype=torch.long, device="meta"),
        torch.empty(2, dtype=torch.long, device="meta"),
        torch.empty(2, dtype=torch.long, device="meta"),
    )
    result = ds.lookup(op)
    assert result is None


def test_comm_miss_raw_csv(ds):
    """Communication: raw CSV format → graceful None."""
    op = MagicMock()
    op.func = torch.ops.tensor_cast.all_reduce.default
    op.args = (
        torch.empty(144, 5120, dtype=torch.bfloat16, device="meta"),
        list(range(16)),
    )
    result = ds.lookup(op)
    assert result is None


def test_embedding_miss_input_count(ds):
    """Embedding: TC sends 2 tensors, GatherV2 CSV has 3 → miss."""
    op = _make_op(
        torch.ops.aten.embedding.default,
        (151936, 5120),
        ((1, 144), torch.long),
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


def test_mm_lmhead_miss_batch(ds):
    """lm_head mm: TC (2,5120) doesn't match CSV (1,5120)."""
    op = _make_op(torch.ops.aten.mm.default, (2, 5120), (5120, 9496))
    result = ds.lookup(op)
    assert result is None
