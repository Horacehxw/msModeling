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


# --- Weight transpose matching tests ---

LMHEAD_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"1,5120;9496,5120","DT_BF16;DT_BF16","ND;ND","1,9496","DT_BF16","ND",91.753
"""


@pytest.fixture
def lmhead_data_dir(tmp_path):
    data_dir = tmp_path / "lmhead"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "aten.mm.default":\n'
        "    kernel_type: MatMulV2\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "MatMulV2.csv").write_text(LMHEAD_CSV.strip())
    return data_dir


def test_nd_weight_transpose_match(lmhead_data_dir):
    """ND-format matmul weight stored as (N,K) should match TC's (K,N).
    CSV has weight (9496,5120) = (N,K). TC mm receives (5120,9496) = (K,N)
    because F.linear transposes before dispatch."""
    ds = ProfilingDataSource(lmhead_data_dir)
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(1, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, 9496, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match with ND weight transpose"
    assert abs(result.latency_us - 91.753) < 0.01


def test_nd_weight_no_false_positive(lmhead_data_dir):
    """Non-transpose shape mismatches should NOT match."""
    ds = ProfilingDataSource(lmhead_data_dir)
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(1, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, 1234, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None, "Completely different N should not match"


# --- Block-padding tolerance tests ---

ADD_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"136,5120;136,5120","DT_BF16;DT_BF16","ND;ND","136,5120","DT_BF16","ND",16.238
"""


@pytest.fixture
def add_data_dir(tmp_path):
    data_dir = tmp_path / "add"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "aten.add.Tensor":\n'
        "    kernel_type: Add\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "Add.csv").write_text(ADD_CSV.strip())
    return data_dir


def test_block_padding_tolerance(add_data_dir):
    """TC seq=144 (padded from 136 via ceil(136/16)*16) should match CSV seq=136."""
    ds = ProfilingDataSource(add_data_dir)
    op = _make_op_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(144, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(144, 5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match with block-padding tolerance (144 ≈ 136)"
    assert abs(result.latency_us - 16.238) < 0.01


def test_block_padding_no_false_positive(add_data_dir):
    """Shapes that aren't block-padding should NOT match."""
    ds = ProfilingDataSource(add_data_dir)
    op = _make_op_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(256, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(256, 5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None, "256 is not a block-padding of 136"


def test_block_padding_32_alignment(add_data_dir):
    """INT8 uses 32-alignment: ceil(136/32)*32=160 should also match."""
    ds = ProfilingDataSource(add_data_dir)
    op = _make_op_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(160, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(160, 5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match with 32-alignment padding (160 ≈ 136)"


# --- Batch-dim stripping tests ---

RMSNORM_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"136,5120;5120","DT_BF16;DT_BF16","ND;ND","136,5120;136,1","DT_BF16;FLOAT","ND;ND",21.660000
"""

ADD_RMSNORM_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"136,5120;136,5120;5120","DT_BF16;DT_BF16;DT_BF16","ND;ND;ND","136,5120;136,1;136,5120","DT_BF16;FLOAT;DT_BF16","ND;ND;ND",33.150000
"""


@pytest.fixture
def rmsnorm_data_dir(tmp_path):
    data_dir = tmp_path / "rmsnorm"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.rms_norm.default":\n'
        "    kernel_type: RmsNorm\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "RmsNorm.csv").write_text(RMSNORM_CSV.strip())
    return data_dir


@pytest.fixture
def add_rmsnorm_data_dir(tmp_path):
    data_dir = tmp_path / "add_rmsnorm"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.add_rms_norm2.default":\n'
        "    kernel_type: AddRmsNorm\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "AddRmsNorm.csv").write_text(ADD_RMSNORM_CSV.strip())
    return data_dir


def test_batch_dim_stripping_rmsnorm(rmsnorm_data_dir):
    """TC RmsNorm sends (1,144,5120),(5120,) — match CSV (136,5120),(5120) after batch strip + padding."""
    ds = ProfilingDataSource(rmsnorm_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.rms_norm.default,
        [
            torch.empty(1, 144, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match after stripping batch dim=1 + padding tolerance"
    assert abs(result.latency_us - 21.66) < 0.01


def test_batch_dim_stripping_add(add_data_dir):
    """TC Add sends (1,144,5120),(1,144,5120) — match CSV (136,5120),(136,5120)."""
    ds = ProfilingDataSource(add_data_dir)
    op = _make_op_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(1, 144, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 144, 5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match after stripping batch dim=1 + padding"


def test_batch_dim_stripping_add_rmsnorm(add_rmsnorm_data_dir):
    """TC AddRmsNorm sends (1,144,5120),(144,5120),(5120,) — match CSV (136,5120),(136,5120),(5120)."""
    ds = ProfilingDataSource(add_rmsnorm_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.add_rms_norm2.default,
        [
            torch.empty(1, 144, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(144, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match after stripping batch dim + padding"


def test_batch_dim_no_false_positive(add_data_dir):
    """Batch dim > 1 should NOT be stripped."""
    ds = ProfilingDataSource(add_data_dir)
    op = _make_op_info(
        torch.ops.aten.add.Tensor,
        [
            torch.empty(2, 144, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(2, 144, 5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None, "Batch dim > 1 should not match"


# --- SwiGlu input concatenation tests ---

SWIGLU_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"136,3200","DT_BF16","ND","136,1600","DT_BF16","ND",14.871969
"""


@pytest.fixture
def swiglu_data_dir(tmp_path):
    data_dir = tmp_path / "swiglu"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.swiglu.default":\n'
        "    kernel_type: SwiGlu\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "SwiGlu.csv").write_text(SWIGLU_CSV.strip())
    return data_dir


def test_swiglu_input_concat(swiglu_data_dir):
    """TC SwiGlu sends 2 inputs (1,144,1600),(1,144,1600) -> CSV has 1 input (136,3200)."""
    ds = ProfilingDataSource(swiglu_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.swiglu.default,
        [
            torch.empty(1, 144, 1600, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 144, 1600, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match SwiGlu after concatenating 2 inputs into 1"
    assert abs(result.latency_us - 14.871969) < 0.01


def test_swiglu_no_false_positive(swiglu_data_dir):
    """Wrong shape should not match."""
    ds = ProfilingDataSource(swiglu_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.swiglu.default,
        [
            torch.empty(1, 256, 1600, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 256, 1600, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None, "Wrong seq dim should not match"


def test_attention_special_returns_none(spike_data_dir):
    """attention_special ops return None in spike, fallback to analytic."""
    ds = ProfilingDataSource(spike_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.attention.default,
        [torch.empty(136, 5120, device="meta", dtype=torch.bfloat16)],
    )
    result = ds.lookup(op)
    assert result is None


# --- RoPE shape normalization tests ---

ROPE_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"1,136,4,128;1,136,1,128;1,136,1,128;1,136,1,128","DT_BF16;DT_BF16;DT_BF16;DT_BF16","ND;ND;ND;ND","1,136,4,128;1,136,1,128","DT_BF16;DT_BF16","ND;ND",12.500000
"""


@pytest.fixture
def rope_data_dir(tmp_path):
    data_dir = tmp_path / "rope"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.apply_rope.default":\n'
        "    kernel_type: InterleaveRope\n"
        "    alternate_kernel_types: [ApplyRotaryPosEmb]\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "ApplyRotaryPosEmb.csv").write_text(ROPE_CSV.strip())
    return data_dir


def test_rope_shape_normalization(rope_data_dir):
    """TC RoPE sends [Q(1,1,144,128), K(1,4,144,128), cos(1,144,128), sin(1,144,128)]
    CSV expects [K(1,136,4,128), Q(1,136,1,128), cos(1,136,1,128), sin(1,136,1,128)].
    Should match after: reorder Q/K, transpose (B,H,S,D)->(B,S,H,D), insert head dim in cos/sin."""
    ds = ProfilingDataSource(rope_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.apply_rope.default,
        [
            torch.empty(1, 1, 144, 128, device="meta", dtype=torch.bfloat16),  # Q
            torch.empty(1, 4, 144, 128, device="meta", dtype=torch.bfloat16),  # K
            torch.empty(1, 144, 128, device="meta", dtype=torch.bfloat16),     # cos
            torch.empty(1, 144, 128, device="meta", dtype=torch.bfloat16),     # sin
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match RoPE after shape normalization + padding"
    assert abs(result.latency_us - 12.5) < 0.01
    assert result.details.get("kernel_type") == "ApplyRotaryPosEmb"


def test_rope_no_false_positive(rope_data_dir):
    """Wrong head count should not match."""
    ds = ProfilingDataSource(rope_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.apply_rope.default,
        [
            torch.empty(1, 8, 144, 128, device="meta", dtype=torch.bfloat16),  # Q with wrong heads
            torch.empty(1, 8, 144, 128, device="meta", dtype=torch.bfloat16),  # K with wrong heads
            torch.empty(1, 144, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 144, 128, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None, "Wrong head count should not match"


# --- Composite decomposition tests ---

COMPOSITE_MATMUL_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"136,512;32,320,16,16","DT_BF16;DT_BF16","ND;FRACTAL_NZ","136,5120","DT_BF16","ND",14.156
"""


@pytest.fixture
def composite_data_dir(tmp_path):
    data_dir = tmp_path / "composite"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.matmul_all_reduce.default":\n'
        "    composite: true\n"
        "    sub_kernels: [MatMulV2, hcom_allReduce_]\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "MatMulV2.csv").write_text(COMPOSITE_MATMUL_CSV.strip())
    return data_dir


def test_composite_decomposition_matmul(composite_data_dir):
    """matmul_all_reduce should decompose and match MatMulV2 sub-kernel."""
    ds = ProfilingDataSource(composite_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.matmul_all_reduce.default,
        [
            torch.empty(144, 512, device="meta", dtype=torch.bfloat16),   # mat1
            torch.empty(512, 5120, device="meta", dtype=torch.bfloat16),  # mat2
            None,   # bias
            0,      # rank
            [0, 1], # rank_group
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match MatMulV2 sub-kernel via composite decomposition"
    assert abs(result.latency_us - 14.156) < 0.01
    assert result.details.get("composite") is True
    assert result.confidence < 1.0  # Lower confidence for partial match


def test_composite_no_sub_kernels(spike_data_dir):
    """Composite op without matching sub-kernel CSVs returns None."""
    ds = ProfilingDataSource(spike_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.multihead_latent_attention.default,
        [torch.empty(136, 5120, device="meta", dtype=torch.bfloat16)],
    )
    result = ds.lookup(op)
    assert result is None
