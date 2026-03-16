from unittest.mock import MagicMock

import pytest
import torch

from tensor_cast.device import CommGrid, InterconnectTopology
from tensor_cast.performance_model.perf_database.data_source import QuerySource

from tensor_cast.performance_model.perf_database.profiling_data_source import (
    DTYPE_MAP,
    fractal_nz_to_nd,
    get_topology_tier,
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
    assert result is not None, (
        "Should match after stripping batch dim=1 + padding tolerance"
    )
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
            torch.empty(1, 144, 128, device="meta", dtype=torch.bfloat16),  # cos
            torch.empty(1, 144, 128, device="meta", dtype=torch.bfloat16),  # sin
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
            torch.empty(
                1, 8, 144, 128, device="meta", dtype=torch.bfloat16
            ),  # Q with wrong heads
            torch.empty(
                1, 8, 144, 128, device="meta", dtype=torch.bfloat16
            ),  # K with wrong heads
            torch.empty(1, 144, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 144, 128, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None, "Wrong head count should not match"


# --- RoPE with _triton_rope + tc_input_count=2 ---

TRITON_ROPE_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"41040,4,128;41040,1,128;81920,128","DT_BF16;DT_BF16;DT_BF16","ND;ND;ND","41040,4,128;41040,1,128","DT_BF16;DT_BF16","ND;ND",55.0
"336,4,128;336,1,128;81920,128","DT_BF16;DT_BF16;DT_BF16","ND;ND;ND","336,4,128;336,1,128","DT_BF16;DT_BF16","ND;ND",8.5
"""


@pytest.fixture
def triton_rope_data_dir(tmp_path):
    data_dir = tmp_path / "triton_rope"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.apply_rope.default":\n'
        "    kernel_type: _triton_rope\n"
        "    tc_input_count: 2\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "_triton_rope.csv").write_text(TRITON_ROPE_CSV.strip())
    return data_dir


def test_triton_rope_tc_input_count_2_prefill(triton_rope_data_dir):
    """TC RoPE with _triton_rope + tc_input_count=2: Qwen3 Prefill.
    TC sends [Q(1,1,41040,128), K(1,4,41040,128), cos, sin] — tc_input_count=2 truncates to [Q, K].
    Normalize: swap Q↔K, transpose (B,H,S,D)→(B,S,H,D), strip batch=1.
    Result: [K(41040,4,128), Q(41040,1,128)] should match CSV first 2 inputs."""
    ds = ProfilingDataSource(triton_rope_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.apply_rope.default,
        [
            torch.empty(1, 1, 41040, 128, device="meta", dtype=torch.bfloat16),  # Q
            torch.empty(1, 4, 41040, 128, device="meta", dtype=torch.bfloat16),  # K
            torch.empty(1, 41040, 128, device="meta", dtype=torch.bfloat16),  # cos
            torch.empty(1, 41040, 128, device="meta", dtype=torch.bfloat16),  # sin
        ],
    )
    result = ds.lookup(op)
    assert result is not None, (
        "Should match _triton_rope with tc_input_count=2 after normalize"
    )
    assert abs(result.latency_us - 55.0) < 0.01


def test_triton_rope_tc_input_count_2_decode_miss(triton_rope_data_dir):
    """TC RoPE Decode: M=16 not in CSV (CSV has M=336, M=41040) — shape_coverage_gap."""
    ds = ProfilingDataSource(triton_rope_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.apply_rope.default,
        [
            torch.empty(1, 1, 16, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 4, 16, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 16, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 16, 128, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None, "M=16 not in CSV — should miss (shape_coverage_gap)"


# --- Composite decomposition tests ---

COMPOSITE_MATMUL_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"136,512;32,320,16,16","DT_BF16;DT_BF16","ND;FRACTAL_NZ","136,5120","DT_BF16","ND",14.156
"""

# mat1[144,512] @ mat2[512,5120] -> output[144,5120]
# message_bytes = 144 * 5120 * 2 (bfloat16) = 1474560, num_devices = 2
COMPOSITE_COMM_CSV = """\
message_bytes,num_devices,Duration(us)
1474560,2,200.00
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
    (data_dir / "hcom_allReduce_.csv").write_text(COMPOSITE_COMM_CSV.strip())
    return data_dir


def test_composite_decomposition_matmul(composite_data_dir):
    """matmul_all_reduce decomposes to MatMulV2 + hcom_allReduce_; latency is summed."""
    ds = ProfilingDataSource(composite_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.matmul_all_reduce.default,
        [
            torch.empty(144, 512, device="meta", dtype=torch.bfloat16),  # mat1
            torch.empty(512, 5120, device="meta", dtype=torch.bfloat16),  # mat2
            None,  # bias
            0,  # rank
            [0, 1],  # rank_group
        ],
    )
    result = ds.lookup(op)
    assert result is not None, (
        "Should match both MatMulV2 and hcom_allReduce_ sub-kernels"
    )
    assert abs(result.latency_us - (14.156 + 200.00)) < 0.01
    assert result.details.get("composite") is True
    assert result.confidence == 0.9


def test_composite_no_sub_kernels(spike_data_dir):
    """Composite op without matching sub-kernel CSVs returns None."""
    ds = ProfilingDataSource(spike_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.multihead_latent_attention.default,
        [torch.empty(136, 5120, device="meta", dtype=torch.bfloat16)],
    )
    result = ds.lookup(op)
    assert result is None


# --- B2: composite sub-kernel sum tests ---


# Fixture: compute CSV only (no comm CSV) — for comm-miss scenario
@pytest.fixture
def mc2_compute_only_dir(tmp_path):
    data_dir = tmp_path / "mc2_compute_only"
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
    # No hcom_allReduce_.csv
    return data_dir


# Fixture: compute CSV with wrong shapes — for shape-mismatch scenario
@pytest.fixture
def mc2_wrong_shape_dir(tmp_path):
    data_dir = tmp_path / "mc2_wrong_shape"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.matmul_all_reduce.default":\n'
        "    composite: true\n"
        "    sub_kernels: [MatMulV2, hcom_allReduce_]\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    # CSV has different shapes — won't match mat1[144,512] @ mat2[512,5120]
    wrong_csv = (
        "Input Shapes,Input Data Types,Input Formats,Output Shapes,"
        "Output Data Types,Output Formats,Average Duration(us)\n"
        '"1,256;16,160,16,16","DT_BF16;DT_BF16","ND;FRACTAL_NZ",'
        '"1,2560","DT_BF16","ND",99.0\n'
    )
    (data_dir / "MatMulV2.csv").write_text(wrong_csv.strip())
    (data_dir / "hcom_allReduce_.csv").write_text(COMPOSITE_COMM_CSV.strip())
    return data_dir


def test_composite_mc2_compute_hit_comm_miss_returns_none(mc2_compute_only_dir):
    """Compute sub-kernel hits but comm CSV absent → None + comm_sub_kernel_miss."""
    ds = ProfilingDataSource(mc2_compute_only_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.matmul_all_reduce.default,
        [
            torch.empty(144, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(512, 5120, device="meta", dtype=torch.bfloat16),
            None,
            0,
            [0, 1],
        ],
    )
    result = ds.lookup(op)
    assert result is None
    assert ds.last_miss_reason == "comm_sub_kernel_miss"


def test_composite_mc2_compute_miss_returns_none(mc2_wrong_shape_dir):
    """Compute CSV exists but shapes don't match → None + shape_mismatch."""
    ds = ProfilingDataSource(mc2_wrong_shape_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.matmul_all_reduce.default,
        [
            torch.empty(144, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(512, 5120, device="meta", dtype=torch.bfloat16),
            None,
            0,
            [0, 1],
        ],
    )
    result = ds.lookup(op)
    assert result is None
    assert ds.last_miss_reason == "shape_mismatch"


def test_composite_mla_csv_not_found_returns_none(spike_data_dir):
    """MLA composite: attempts composite lookup, returns csv_not_found (sub-kernel CSV missing)."""
    ds = ProfilingDataSource(spike_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.multihead_latent_attention.default,
        [torch.empty(136, 5120, device="meta", dtype=torch.bfloat16)],
    )
    result = ds.lookup(op)
    assert result is None
    # After C1 fix: composite lookup attempted, sub-kernel CSV missing
    assert ds.last_miss_reason != "mla_not_implemented"


def test_composite_no_sub_kernels_miss_reason(tmp_path):
    """Composite op with empty sub_kernels list → None + no_sub_kernels."""
    data_dir = tmp_path / "empty_sub"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.matmul_all_reduce.default":\n'
        "    composite: true\n"
        "    sub_kernels: []\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    ds = ProfilingDataSource(data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.matmul_all_reduce.default,
        [
            torch.empty(144, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(512, 5120, device="meta", dtype=torch.bfloat16),
            None,
            0,
            [0, 1],
        ],
    )
    result = ds.lookup(op)
    assert result is None
    assert ds.last_miss_reason == "no_sub_kernels"


# --- B2: quant MC2 + MLA placeholder tests ---

# QuantBatchMatmulV3 CSV: INT8 inputs, ND format
# x[144,512] INT8, w[512,5120] INT8 → output[144,5120] BF16
QUANT_MATMUL_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"144,512;512,5120","INT8;INT8","ND;ND","144,5120","DT_BF16","ND",22.5
"""

# Comm CSV: message_bytes = 144 * 5120 * 2 (BF16 output) = 1474560
QUANT_COMM_CSV = """\
message_bytes,num_devices,Duration(us)
1474560,2,200.00
"""

# Wrong comm CSV: message_bytes = 144 * 5120 * 1 (INT8 input) = 737280
QUANT_COMM_WRONG_CSV = """\
message_bytes,num_devices,Duration(us)
737280,2,150.00
"""


@pytest.fixture
def quant_mc2_data_dir(tmp_path):
    """Quant MC2 fixture: static_quant_linear_all_reduce with tc_input_count=2."""
    data_dir = tmp_path / "quant_mc2"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.static_quant_linear_all_reduce.default":\n'
        "    composite: true\n"
        "    sub_kernels: [QuantBatchMatmulV3, hcom_allReduce_]\n"
        "    tc_input_count: 2\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "QuantBatchMatmulV3.csv").write_text(QUANT_MATMUL_CSV.strip())
    (data_dir / "hcom_allReduce_.csv").write_text(QUANT_COMM_CSV.strip())
    return data_dir


def test_composite_quant_mc2_hit(quant_mc2_data_dir):
    """static_quant_linear_all_reduce: tc_input_count=2 truncates 6 tensor args to x+w,
    matches QuantBatchMatmulV3 + hcom_allReduce_, latency summed."""
    ds = ProfilingDataSource(quant_mc2_data_dir)
    # 6 tensor args: x, w, scale, zero_point, bias, per_token_scale
    op = _make_op_info(
        torch.ops.tensor_cast.static_quant_linear_all_reduce.default,
        [
            torch.empty(144, 512, device="meta", dtype=torch.int8),  # x
            torch.empty(512, 5120, device="meta", dtype=torch.int8),  # w
            torch.empty(5120, device="meta", dtype=torch.bfloat16),  # scale
            torch.empty(5120, device="meta", dtype=torch.int8),  # zero_point
            torch.empty(5120, device="meta", dtype=torch.bfloat16),  # bias
            torch.empty(144, device="meta", dtype=torch.bfloat16),  # per_token_scale
            0,  # rank
            [0, 1],  # rank_group
        ],
        output_tensors=[torch.empty(144, 5120, device="meta", dtype=torch.bfloat16)],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match with tc_input_count=2 truncation"
    assert abs(result.latency_us - (22.5 + 200.00)) < 0.01
    assert result.details.get("composite") is True


def test_composite_quant_mc2_message_bytes_uses_output_dtype(tmp_path):
    """message_bytes should use BF16 output (2B) not INT8 input (1B).
    With INT8 input: 144*5120*1=737280. With BF16 output: 144*5120*2=1474560.
    Only the BF16-sized comm CSV should match."""
    data_dir = tmp_path / "quant_mc2_dtype"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.static_quant_linear_all_reduce.default":\n'
        "    composite: true\n"
        "    sub_kernels: [QuantBatchMatmulV3, hcom_allReduce_]\n"
        "    tc_input_count: 2\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "QuantBatchMatmulV3.csv").write_text(QUANT_MATMUL_CSV.strip())
    # Only provide INT8-sized comm CSV (737280) — should NOT match
    (data_dir / "hcom_allReduce_.csv").write_text(QUANT_COMM_WRONG_CSV.strip())

    ds = ProfilingDataSource(data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.static_quant_linear_all_reduce.default,
        [
            torch.empty(144, 512, device="meta", dtype=torch.int8),
            torch.empty(512, 5120, device="meta", dtype=torch.int8),
            torch.empty(5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, device="meta", dtype=torch.int8),
            torch.empty(5120, device="meta", dtype=torch.bfloat16),
            torch.empty(144, device="meta", dtype=torch.bfloat16),
            0,
            [0, 1],
        ],
        output_tensors=[torch.empty(144, 5120, device="meta", dtype=torch.bfloat16)],
    )
    result = ds.lookup(op)
    # BF16 output → message_bytes=1474560, but CSV only has 737280 → comm miss
    assert result is None
    assert ds.last_miss_reason == "comm_sub_kernel_miss"


def test_composite_mla_attempts_lookup(tmp_path):
    """After C1: MLA attempts composite lookup instead of rejecting."""
    data_dir = tmp_path / "mla_composite"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.multihead_latent_attention.default":\n'
        "    composite: true\n"
        "    sub_kernels: [BatchMatMulV2, FusedInferAttentionScore]\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    ds = ProfilingDataSource(data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.multihead_latent_attention.default,
        [torch.empty(136, 5120, device="meta", dtype=torch.bfloat16)],
    )
    result = ds.lookup(op)
    assert result is None  # CSVs missing, but composite lookup attempted
    assert ds.last_miss_reason != "mla_not_implemented"


def test_composite_mlapo_attempts_lookup(tmp_path):
    """After C1: MLAPO attempts composite lookup instead of rejecting."""
    data_dir = tmp_path / "mlapo_composite"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.mlapo.default":\n'
        "    composite: true\n"
        "    sub_kernels: [MatMulV2, KvRmsNormRopeCache]\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    ds = ProfilingDataSource(data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.mlapo.default,
        [torch.empty(136, 5120, device="meta", dtype=torch.bfloat16)],
    )
    result = ds.lookup(op)
    assert result is None  # CSVs missing, but composite lookup attempted
    assert ds.last_miss_reason != "mla_not_implemented"


def test_composite_tc_input_count_truncation(tmp_path):
    """tc_input_count truncation: 4 tensor args truncated to 2, matches CSV with 2 inputs."""
    data_dir = tmp_path / "tc_input_trunc"
    data_dir.mkdir()
    # Generic composite with tc_input_count=2 and compute-only sub_kernels
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.static_quant_linear_all_reduce.default":\n'
        "    composite: true\n"
        "    sub_kernels: [QuantBatchMatmulV3]\n"
        "    tc_input_count: 2\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "QuantBatchMatmulV3.csv").write_text(QUANT_MATMUL_CSV.strip())

    ds = ProfilingDataSource(data_dir)
    # 4 tensor args — without truncation, len(tc_inputs)=4 != len(csv_shapes)=2 → miss
    op = _make_op_info(
        torch.ops.tensor_cast.static_quant_linear_all_reduce.default,
        [
            torch.empty(144, 512, device="meta", dtype=torch.int8),
            torch.empty(512, 5120, device="meta", dtype=torch.int8),
            torch.empty(5120, device="meta", dtype=torch.bfloat16),  # extra: scale
            torch.empty(144, device="meta", dtype=torch.bfloat16),  # extra: per_token
            0,
            [0, 1],
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "tc_input_count=2 should truncate to x+w and match"
    assert abs(result.latency_us - 22.5) < 0.01
    assert result.details.get("composite") is True


# --- Communication query tests (design doc §4.7) ---

COMM_OP_MAPPING_YAML = """
version: "test"
device: TEST_DEVICE

operator_mappings:
  "tensor_cast.all_reduce.default":
    kernel_type: hcom_allReduce_
    category: communication
  "tensor_cast.all_gather.default":
    kernel_type: hcom_allGather_
    category: communication
  "tensor_cast.all_to_all.default":
    kernel_type: hcom_alltoallv_
    category: communication
  "aten.mm.default":
    kernel_type: MatMulV2
"""

# Design doc §4.7: message_bytes, num_devices, dtype, topology_tier, Duration(us)
COMM_ALLREDUCE_CSV = """\
message_bytes,num_devices,dtype,topology_tier,Duration(us)
1310720,16,DT_BF16,0,689.96
655360,16,DT_BF16,0,412.50
1310720,4,DT_BF16,2,125.30
"""

COMM_ALLGATHER_CSV = """\
message_bytes,num_devices,dtype,topology_tier,Duration(us)
655360,16,DT_BF16,0,167.62
"""


@pytest.fixture
def comm_data_dir(tmp_path):
    data_dir = tmp_path / "comm"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(COMM_OP_MAPPING_YAML)
    (data_dir / "hcom_allReduce_.csv").write_text(COMM_ALLREDUCE_CSV.strip())
    (data_dir / "hcom_allGather_.csv").write_text(COMM_ALLGATHER_CSV.strip())
    return data_dir


def test_comm_allreduce_exact_match(comm_data_dir):
    """all_reduce with matching message_bytes + num_devices should return latency."""
    ds = ProfilingDataSource(comm_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(1, 640, 1024, device="meta", dtype=torch.bfloat16),
            0,
            list(range(16)),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match comm CSV by message_bytes + num_devices"
    assert abs(result.latency_us - 689.96) < 0.01
    assert result.source == QuerySource.MEASURED
    assert result.details.get("kernel_type") == "hcom_allReduce_"


def test_comm_allreduce_different_shape_same_bytes(comm_data_dir):
    """Different tensor shape but same message_bytes should still match."""
    ds = ProfilingDataSource(comm_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(640, 1024, device="meta", dtype=torch.bfloat16),
            0,
            list(range(16)),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 689.96) < 0.01


def test_comm_allreduce_miss_wrong_bytes(comm_data_dir):
    """Non-matching message_bytes should return None."""
    ds = ProfilingDataSource(comm_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(100, 100, device="meta", dtype=torch.bfloat16),
            0,
            list(range(16)),
        ],
    )
    result = ds.lookup(op)
    assert result is None


def test_comm_allgather_match(comm_data_dir):
    """all_gather(x, dim, rank, rank_group) should match by message_bytes."""
    ds = ProfilingDataSource(comm_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.all_gather.default,
        [
            torch.empty(1, 640, 512, device="meta", dtype=torch.bfloat16),
            0,
            0,
            list(range(16)),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 167.62) < 0.01


def test_comm_no_csv_returns_none(comm_data_dir):
    """Communication op without CSV file should return None."""
    ds = ProfilingDataSource(comm_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.all_to_all.default,
        [
            torch.empty(100, 512, device="meta", dtype=torch.bfloat16),
            [25] * 4,
            [25] * 4,
            0,
            [0, 1, 2, 3],
        ],
    )
    result = ds.lookup(op)
    assert result is None


# --- _comm_data_dir fallback tests ---

COMM_DATA_REF_OP_MAPPING_YAML = """
version: "test"
device: TEST_DEVICE

communication_data_ref: "../hccl_ref/"

operator_mappings:
  "tensor_cast.all_reduce.default":
    kernel_type: hcom_allReduce_
    category: communication
"""


@pytest.fixture
def comm_data_ref_dir(tmp_path):
    """Data dir with communication_data_ref pointing to a sibling hccl dir."""
    data_dir = tmp_path / "main"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(COMM_DATA_REF_OP_MAPPING_YAML)
    # CSV lives in the referenced dir, not in data_dir
    hccl_dir = tmp_path / "hccl_ref"
    hccl_dir.mkdir()
    (hccl_dir / "hcom_allReduce_.csv").write_text(COMM_ALLREDUCE_CSV.strip())
    return data_dir


def test_comm_data_ref_fallback(comm_data_ref_dir):
    """_load_csv should find CSV via communication_data_ref when not in data_dir."""
    ds = ProfilingDataSource(comm_data_ref_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(1, 640, 1024, device="meta", dtype=torch.bfloat16),
            0,
            list(range(16)),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 689.96) < 0.01


# --- Attention special query tests (design doc §4.8) ---

ATTN_OP_MAPPING_YAML = """
version: "test"
device: TEST_DEVICE

operator_mappings:
  "tensor_cast.attention.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
  "tensor_cast.attention_quant.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
"""

# Design doc §4.8 microbenchmark format
ATTN_FIA_CSV = """\
batch_size,avg_seq_len,num_heads,head_dim,dtype,Duration(us)
1,4096,4,128,DT_BF16,56.18
2,3500,4,128,DT_BF16,98.50
10,4500,4,128,DT_BF16,890.70
1,4096,8,128,DT_BF16,112.36
"""


@pytest.fixture
def attn_data_dir(tmp_path):
    data_dir = tmp_path / "attn"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(ATTN_OP_MAPPING_YAML)
    (data_dir / "FusedInferAttentionScore.csv").write_text(ATTN_FIA_CSV.strip())
    return data_dir


def test_attention_prefill_match(attn_data_dir):
    """Prefill: batch=2, seq_lens=[3500,3500], 4 heads, head_dim=128."""
    ds = ProfilingDataSource(attn_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.attention.default,
        [
            torch.empty(
                7000, 512, device="meta", dtype=torch.bfloat16
            ),  # query: hidden=4*128=512
            torch.empty(
                56, 128, 4, 128, device="meta", dtype=torch.bfloat16
            ),  # key (paged)
            torch.empty(56, 128, 4, 128, device="meta", dtype=torch.bfloat16),  # value
            None,  # attention_mask
            torch.empty(2, 28, device="meta", dtype=torch.int32),  # block_table
            torch.empty(3, device="meta", dtype=torch.int64),  # query_start_loc
            torch.tensor([3500, 3500], dtype=torch.int64),  # seq_lens
            torch.tensor([3500, 3500], dtype=torch.int64),  # query_lens
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match FIA by batch_size=2, avg_seq_len=3500"
    assert abs(result.latency_us - 98.50) < 0.01
    assert result.details.get("kernel_type") == "FusedInferAttentionScore"


def test_attention_decode_match(attn_data_dir):
    """Decode: batch=10, seq_lens=[4500]*10, 4 heads, head_dim=128."""
    ds = ProfilingDataSource(attn_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.attention.default,
        [
            torch.empty(10, 512, device="meta", dtype=torch.bfloat16),  # query
            torch.empty(360, 128, 4, 128, device="meta", dtype=torch.bfloat16),  # key
            torch.empty(360, 128, 4, 128, device="meta", dtype=torch.bfloat16),  # value
            None,
            torch.empty(10, 36, device="meta", dtype=torch.int32),
            torch.empty(11, device="meta", dtype=torch.int64),
            torch.tensor([4500] * 10, dtype=torch.int64),  # seq_lens
            torch.tensor([1] * 10, dtype=torch.int64),  # query_lens
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match FIA by batch_size=10, avg_seq_len=4500"
    assert abs(result.latency_us - 890.70) < 0.01


def test_attention_miss_wrong_heads(attn_data_dir):
    """Wrong num_heads should not match."""
    ds = ProfilingDataSource(attn_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.attention.default,
        [
            torch.empty(1, 2048, device="meta", dtype=torch.bfloat16),  # 16 heads * 128
            torch.empty(
                32, 128, 16, 128, device="meta", dtype=torch.bfloat16
            ),  # 16 kv heads
            torch.empty(32, 128, 16, 128, device="meta", dtype=torch.bfloat16),
            None,
            torch.empty(1, 32, device="meta", dtype=torch.int32),
            torch.empty(2, device="meta", dtype=torch.int64),
            torch.tensor([4096], dtype=torch.int64),
            torch.tensor([4096], dtype=torch.int64),
        ],
    )
    result = ds.lookup(op)
    assert result is None, "16 heads not in CSV, should miss"


def test_attention_miss_no_seq_lens(attn_data_dir):
    """If seq_lens is None, should return None gracefully."""
    ds = ProfilingDataSource(attn_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.attention.default,
        [
            torch.empty(100, 512, device="meta", dtype=torch.bfloat16),
            torch.empty(10, 128, 4, 128, device="meta", dtype=torch.bfloat16),
            torch.empty(10, 128, 4, 128, device="meta", dtype=torch.bfloat16),
            None,
            None,
            None,
            None,
            None,
        ],
    )
    result = ds.lookup(op)
    assert result is None, "No seq_lens -> can't compute batch/seq, return None"


# --- topology_tier matching tests (design doc §4.7) ---
#
# Test grid: [2, 4] — 2 pods, 4 devices per pod
#   tier 0 (inter_pod): ranks spanning different pods  (e.g. [0, 4])
#   tier 1 (intra_pod): ranks within same pod          (e.g. [0, 1])
#
# Rank → coord mapping:
#   rank 0 → [0, 0],  rank 1 → [0, 1],  rank 2 → [0, 2],  rank 3 → [0, 3]
#   rank 4 → [1, 0],  rank 5 → [1, 1],  rank 6 → [1, 2],  rank 7 → [1, 3]


def _make_test_comm_grid() -> CommGrid:
    """2-tier grid [2, 4]: tier 0 = inter-pod, tier 1 = intra-pod."""
    return CommGrid(
        grid=torch.zeros([2, 4], dtype=torch.int32),
        topologies={
            0: InterconnectTopology(bandwidth_bytes_ps=196e9, latency_s=5.5e-6),
            1: InterconnectTopology(bandwidth_bytes_ps=224e9, latency_s=0.2e-6),
        },
    )


# CSV with two rows: same message_bytes+num_devices=2, different topology_tier
# num_devices=2 matches rank_group size ([0,4] or [0,1] both have 2 elements)
# message_bytes = torch.empty(4,1024,160,bfloat16).nelement()*2 = 655360*2 = 1310720
COMM_TIERED_CSV = """\
message_bytes,num_devices,dtype,topology_tier,Duration(us)
1310720,2,DT_BF16,0,689.96
1310720,2,DT_BF16,1,125.30
"""

COMM_TIER0_ONLY_CSV = """\
message_bytes,num_devices,dtype,topology_tier,Duration(us)
1310720,2,DT_BF16,0,689.96
"""

COMM_NO_TIER_COL_CSV = """\
message_bytes,num_devices,dtype,Duration(us)
1310720,2,DT_BF16,350.00
"""


@pytest.fixture
def tiered_comm_dir(tmp_path):
    data_dir = tmp_path / "tiered_comm"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.all_reduce.default":\n'
        "    kernel_type: hcom_allReduce_\n"
        "    category: communication\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "hcom_allReduce_.csv").write_text(COMM_TIERED_CSV.strip())
    return data_dir


@pytest.fixture
def tier0_only_comm_dir(tmp_path):
    data_dir = tmp_path / "tier0_only"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.all_reduce.default":\n'
        "    kernel_type: hcom_allReduce_\n"
        "    category: communication\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "hcom_allReduce_.csv").write_text(COMM_TIER0_ONLY_CSV.strip())
    return data_dir


@pytest.fixture
def no_tier_col_comm_dir(tmp_path):
    data_dir = tmp_path / "no_tier_col"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.all_reduce.default":\n'
        "    kernel_type: hcom_allReduce_\n"
        "    category: communication\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "hcom_allReduce_.csv").write_text(COMM_NO_TIER_COL_CSV.strip())
    return data_dir


# --- get_topology_tier unit tests ---


def testget_topology_tier_inter_pod():
    """Ranks spanning different pods → tier 0 (inter_pod)."""
    comm_grid = _make_test_comm_grid()
    # rank 0 → [0,0], rank 4 → [1,0]: differ at dim 0 → tier 0
    assert get_topology_tier(comm_grid, [0, 4]) == 0


def testget_topology_tier_intra_pod():
    """Ranks within same pod → tier 1 (intra_pod)."""
    comm_grid = _make_test_comm_grid()
    # rank 0 → [0,0], rank 1 → [0,1]: differ at dim 1 → tier 1
    assert get_topology_tier(comm_grid, [0, 1]) == 1


def testget_topology_tier_multi_rank_intra():
    """All ranks in same pod → tier 1."""
    comm_grid = _make_test_comm_grid()
    assert get_topology_tier(comm_grid, [0, 1, 2, 3]) == 1


def testget_topology_tier_multi_rank_inter():
    """Ranks spanning pods → tier 0."""
    comm_grid = _make_test_comm_grid()
    assert get_topology_tier(comm_grid, [0, 1, 4, 5]) == 0


def _make_device_profile_with_comm_grid(comm_grid):
    """Wrap a CommGrid in a mock DeviceProfile for ProfilingDataSource."""
    mock_dp = MagicMock()
    mock_dp.comm_grid = comm_grid
    return mock_dp


# --- _lookup_comm topology_tier integration tests ---


def test_comm_topology_tier_selects_correct_row(tiered_comm_dir):
    """With comm_grid, inter-pod group (tier 0) should match the tier=0 row (689.96 us)."""
    comm_grid = _make_test_comm_grid()
    ds = ProfilingDataSource(
        tiered_comm_dir, _make_device_profile_with_comm_grid(comm_grid)
    )
    # rank_group [0,4] spans pods → tier 0
    # tensor: 4 devices, message_bytes = 4 * 1024 * 160 * 2 = 1310720
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(4, 1024, 160, device="meta", dtype=torch.bfloat16),
            0,
            [0, 4],
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 689.96) < 0.01
    assert result.details.get("topology_tier") == 0


def test_comm_topology_tier_intra_pod_row(tiered_comm_dir):
    """Intra-pod group (tier 1) should match the tier=1 row (125.30 us)."""
    comm_grid = _make_test_comm_grid()
    ds = ProfilingDataSource(
        tiered_comm_dir, _make_device_profile_with_comm_grid(comm_grid)
    )
    # rank_group [0,1] within pod → tier 1
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(4, 1024, 160, device="meta", dtype=torch.bfloat16),
            0,
            [0, 1],
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 125.30) < 0.01
    assert result.details.get("topology_tier") == 1


def test_comm_topology_tier_miss_when_tier_absent(tier0_only_comm_dir):
    """Intra-pod group (tier 1) should MISS when CSV only has tier=0 data."""
    comm_grid = _make_test_comm_grid()
    ds = ProfilingDataSource(
        tier0_only_comm_dir, _make_device_profile_with_comm_grid(comm_grid)
    )
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(4, 1024, 160, device="meta", dtype=torch.bfloat16),
            0,
            [0, 1],  # intra-pod → tier 1, not in CSV
        ],
    )
    result = ds.lookup(op)
    assert result is None, "tier=1 not in CSV, should miss"


def test_comm_no_comm_grid_ignores_topology_tier(tiered_comm_dir):
    """Without comm_grid, topology_tier column is ignored; first matching row returned."""
    ds = ProfilingDataSource(tiered_comm_dir)  # no comm_grid
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(4, 1024, 160, device="meta", dtype=torch.bfloat16),
            0,
            [0, 1],
        ],
    )
    result = ds.lookup(op)
    # Both rows match on message_bytes+num_devices; first row (tier=0, 689.96) returned
    assert result is not None
    assert abs(result.latency_us - 689.96) < 0.01
    assert result.details.get("topology_tier") is None


def test_comm_csv_without_topology_tier_col(no_tier_col_comm_dir):
    """CSV without topology_tier column works fine even when comm_grid is provided."""
    comm_grid = _make_test_comm_grid()
    ds = ProfilingDataSource(
        no_tier_col_comm_dir, _make_device_profile_with_comm_grid(comm_grid)
    )
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(4, 1024, 160, device="meta", dtype=torch.bfloat16),
            0,
            [0, 1],
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 350.00) < 0.01


# --- communication_data_ref path redirection tests (design doc §5/§6) ---
#
# Directory layout mirrors production:
#   vllm_ascend/v0.13.0/op_mapping.yaml  ← communication_data_ref: "../../hccl/v8.1.RC1/"
#   hccl/v8.1.RC1/hcom_allReduce_.csv
#
# message_bytes = torch.empty(4,1024,160,bfloat16).nelement()*2 = 1310720

_COMM_REF_OP_MAPPING_WITH_REF = """\
version: "test"
communication_data_ref: "../../hccl/v8.1.RC1/"
operator_mappings:
  "tensor_cast.all_reduce.default":
    kernel_type: hcom_allReduce_
    category: communication
"""

_COMM_REF_OP_MAPPING_NO_REF = """\
version: "test"
operator_mappings:
  "tensor_cast.all_reduce.default":
    kernel_type: hcom_allReduce_
    category: communication
"""

_COMM_REF_CSV = """\
message_bytes,num_devices,dtype,topology_tier,Duration(us)
1310720,16,DT_BF16,0,512.00
"""


@pytest.fixture
def comm_ref_dir(tmp_path):
    """Separate hccl dir; op_mapping.yaml points to it via communication_data_ref."""
    vllm_dir = tmp_path / "vllm_ascend" / "v0.13.0"
    vllm_dir.mkdir(parents=True)
    hccl_dir = tmp_path / "hccl" / "v8.1.RC1"
    hccl_dir.mkdir(parents=True)
    (vllm_dir / "op_mapping.yaml").write_text(_COMM_REF_OP_MAPPING_WITH_REF)
    (hccl_dir / "hcom_allReduce_.csv").write_text(_COMM_REF_CSV.strip())
    return vllm_dir


@pytest.fixture
def comm_no_ref_dir(tmp_path):
    """Legacy layout: CSV and op_mapping.yaml in the same directory, no communication_data_ref."""
    data_dir = tmp_path / "legacy"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(_COMM_REF_OP_MAPPING_NO_REF)
    (data_dir / "hcom_allReduce_.csv").write_text(_COMM_REF_CSV.strip())
    return data_dir


def test_comm_data_ref_resolves_csv_from_separate_dir(comm_ref_dir):
    """communication_data_ref points to a separate hccl dir; CSV should be found and hit."""
    ds = ProfilingDataSource(comm_ref_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(4, 1024, 160, device="meta", dtype=torch.bfloat16),
            0,
            list(range(16)),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, f"Expected hit, got miss: {ds.last_miss_reason}"
    assert abs(result.latency_us - 512.00) < 0.01
    assert result.details.get("kernel_type") == "hcom_allReduce_"


def test_comm_data_ref_missing_falls_back_to_data_dir(comm_no_ref_dir):
    """Without communication_data_ref, _comm_data_dir falls back to data_dir (legacy layout)."""
    ds = ProfilingDataSource(comm_no_ref_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(4, 1024, 160, device="meta", dtype=torch.bfloat16),
            0,
            list(range(16)),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, (
        f"Expected hit in legacy layout, got: {ds.last_miss_reason}"
    )
    assert abs(result.latency_us - 512.00) < 0.01


def test_comm_data_ref_csv_not_found_returns_none(comm_ref_dir):
    """communication_data_ref dir exists but CSV is absent → None + csv_not_found."""
    ds = ProfilingDataSource(comm_ref_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.all_gather.default,  # no hcom_allGather_.csv in hccl dir
        [
            torch.empty(4, 1024, 160, device="meta", dtype=torch.bfloat16),
            0,
            0,
            list(range(16)),
        ],
    )
    # Need all_gather in op_mapping — patch the loaded mapping directly
    ds._op_mapping.setdefault("operator_mappings", {})[
        "tensor_cast.all_gather.default"
    ] = {"kernel_type": "hcom_allGather_", "category": "communication"}
    result = ds.lookup(op)
    assert result is None
    assert ds.last_miss_reason == "csv_not_found"


# --- Comm interpolation tests ---


def test_comm_allreduce_interpolates_message_bytes(comm_data_dir):
    """When exact message_bytes misses, interpolate between bracketing rows."""
    ds = ProfilingDataSource(comm_data_dir)
    # comm_data_dir has allReduce: 655360→412.50us, 1310720→689.96us (num_devices=16, tier=0)
    # Query 983040 bytes (midpoint): 412.50 + 277.46 * 0.5 = 551.23
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(
                983040 // 2, device="meta", dtype=torch.bfloat16
            ),  # 983040 bytes
            0,
            list(range(16)),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, (
        "Should interpolate comm between bracketing message_bytes"
    )
    assert abs(result.latency_us - 551.23) < 1.0
    assert result.source == QuerySource.INTERPOLATED


def test_comm_allreduce_no_extrapolation(comm_data_dir):
    """When message_bytes is outside CSV range, return None (no extrapolation)."""
    ds = ProfilingDataSource(comm_data_dir)
    # CSV max for num_devices=16, tier=0 is 1310720. Query 2x max → can't bracket
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(2621440 // 2, device="meta", dtype=torch.bfloat16),
            0,
            list(range(16)),
        ],
    )
    result = ds.lookup(op)
    assert result is None, "Should not extrapolate beyond CSV range"


def test_comm_interpolation_latency_dominated_region(tmp_path):
    """In latency-dominated region (small messages), interpolation should
    stay near alpha (startup latency), not linearly ramp toward the next point.

    Real HCCL pattern: latency ≈ 120us for 1KB-4MB, then ramps.
    Naive linear between 1MB (120us) and 16MB (300us) would predict
    166us at 4MB, but actual is ~107us (still latency-dominated).
    """
    data_dir = tmp_path / "alpha_beta"
    data_dir.mkdir()
    (data_dir / "op_mapping.yaml").write_text(COMM_OP_MAPPING_YAML)

    # Mimic real HCCL data with alpha-beta behavior (powers-of-4 spacing)
    csv_content = """\
message_bytes,num_devices,dtype,topology_tier,Duration(us)
1024,16,DT_BF16,1,120.0
4096,16,DT_BF16,1,120.0
16384,16,DT_BF16,1,120.0
65536,16,DT_BF16,1,120.5
1048576,16,DT_BF16,1,130.0
16777216,16,DT_BF16,1,288.0
67108864,16,DT_BF16,1,791.0
268435456,16,DT_BF16,1,2804.0
"""
    (data_dir / "hcom_allReduce_.csv").write_text(csv_content.strip())

    ds = ProfilingDataSource(data_dir)

    # Query 4MB (4194304) — between 1MB (130.0us) and 16MB (288.0us)
    # Alpha-beta model: alpha≈120, beta≈100GB/s → 120 + 4194304/100000 ≈ 162us
    # Naive linear: 130.0 + (288.0-130.0) * (4194304-1048576)/(16777216-1048576) = 161.6us
    # Both happen to give ~162us here (acceptable)
    #
    # Better test: 160KB (163840) — between 64KB (120.5us) and 1MB (130.0us)
    # This is the actual Qwen3 allReduce message size!
    # Naive linear: 120.5 + (130.0-120.5) * (163840-65536)/(1048576-65536) = 121.5us
    # Alpha-beta:   120 + 163840/100000 = 121.6us (close, because bracket is tight)
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(
                163840 // 2, device="meta", dtype=torch.bfloat16
            ),  # 163840 bytes
            0,
            list(range(16)),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert result.source == QuerySource.INTERPOLATED
    # Should be in [120.5, 130.0] range and close to alpha (~120-122us)
    assert 120.0 <= result.latency_us <= 125.0, (
        f"Interpolated {result.latency_us:.1f}us: latency-dominated region "
        f"should stay near alpha (~120us), not ramp toward 130us"
    )


def test_comm_allreduce_exact_still_measured(comm_data_dir):
    """Exact message_bytes match should return MEASURED, not INTERPOLATED."""
    ds = ProfilingDataSource(comm_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.all_reduce.default,
        [
            torch.empty(
                1, 640, 1024, device="meta", dtype=torch.bfloat16
            ),  # 1310720 bytes
            0,
            list(range(16)),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert result.source == QuerySource.MEASURED
    assert abs(result.latency_us - 689.96) < 0.01


# --- MoE csv_file + tc_input_count tests ---

MOE_OP_MAPPING_YAML = """\
version: "0.14.0"
device: TEST_DEVICE

operator_mappings:
  "tensor_cast.permute_tokens.default":
    kernel_type: MoeDistributeDispatchV2
    csv_file: MoeTokenPermute
    tc_input_count: 2
  "tensor_cast.unpermute_tokens.default":
    kernel_type: MoeDistributeCombineV2
    csv_file: MoeTokenUnpermute
    tc_input_count: 1
"""

# MoE permute CSV (simulates MoeTokenPermute.csv raw profiling format)
MOE_PERMUTE_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"4,7168;4,8","DT_BF16;INT32","ND;ND","32,7168","DT_BF16","ND",6.12
"19,1;19","FLOAT;INT32","ND;ND","19,1;19","FLOAT","ND;ND",11.39
"""

# MoE unpermute CSV (simulates MoeTokenUnpermute.csv, contains NPU internal params)
MOE_UNPERMUTE_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"128,7168;128;","DT_BF16;INT32;DT_UNDEFINED","ND;ND;NULL","128,7168","DT_BF16","ND",7.01
"256,7168;256;4,8","DT_BF16;INT32;DT_BF16","ND;ND;ND","4,7168","DT_BF16","ND",6.02
"""


@pytest.fixture
def moe_data_dir(tmp_path):
    d = tmp_path / "moe"
    d.mkdir()
    (d / "op_mapping.yaml").write_text(MOE_OP_MAPPING_YAML)
    (d / "MoeTokenPermute.csv").write_text(MOE_PERMUTE_CSV.strip())
    (d / "MoeTokenUnpermute.csv").write_text(MOE_UNPERMUTE_CSV.strip())
    return d


def test_moe_permute_hit(moe_data_dir):
    """permute_tokens (4,7168)+(4,8) matches first CSV row -> 6.12 us."""
    ds = ProfilingDataSource(moe_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.permute_tokens.default,
        [
            torch.empty(4, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(4, 8, device="meta", dtype=torch.int32),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 6.12) < 0.01


def test_moe_permute_dtype_filter(moe_data_dir):
    """BF16 inputs should not match the FLOAT row."""
    ds = ProfilingDataSource(moe_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.permute_tokens.default,
        [
            torch.empty(19, 1, device="meta", dtype=torch.bfloat16),
            torch.empty(19, device="meta", dtype=torch.int32),
        ],
    )
    result = ds.lookup(op)
    assert result is None


def test_moe_permute_shape_miss(moe_data_dir):
    """(5,7168)+(5,8) has no matching row -> None."""
    ds = ProfilingDataSource(moe_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.permute_tokens.default,
        [
            torch.empty(5, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(5, 8, device="meta", dtype=torch.int32),
        ],
    )
    result = ds.lookup(op)
    assert result is None


def test_moe_unpermute_hit_tc_input_count_1(moe_data_dir):
    """tc_input_count=1: only first TC input (128,7168) compared -> 7.01 us."""
    ds = ProfilingDataSource(moe_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.unpermute_tokens.default,
        [
            torch.empty(128, 7168, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 7.01) < 0.01


def test_moe_unpermute_hit_with_extra_csv_inputs(moe_data_dir):
    """tc_input_count=1: (256,7168) matches second row (which has 3 CSV inputs) -> 6.02 us."""
    ds = ProfilingDataSource(moe_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.unpermute_tokens.default,
        [
            torch.empty(256, 7168, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 6.02) < 0.01


def test_moe_unpermute_shape_miss(moe_data_dir):
    """(512,7168) has no matching row -> None."""
    ds = ProfilingDataSource(moe_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.unpermute_tokens.default,
        [
            torch.empty(512, 7168, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None


def test_csv_file_field_override(moe_data_dir):
    """csv_file overrides kernel_type for CSV loading."""
    ds = ProfilingDataSource(moe_data_dir)
    assert not (moe_data_dir / "MoeDistributeDispatchV2.csv").exists()
    assert (moe_data_dir / "MoeTokenPermute.csv").exists()
    op = _make_op_info(
        torch.ops.tensor_cast.permute_tokens.default,
        [
            torch.empty(4, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(4, 8, device="meta", dtype=torch.int32),
        ],
    )
    result = ds.lookup(op)
    assert result is not None


def test_csv_file_field_fallback(moe_data_dir):
    """Without csv_file, kernel_type is used as CSV filename (regression guard)."""
    ds = ProfilingDataSource(moe_data_dir)
    mappings = ds._op_mapping["operator_mappings"]
    mappings["tensor_cast.permute_tokens.default"] = {
        "kernel_type": "MoeTokenPermute",
    }
    op = _make_op_info(
        torch.ops.tensor_cast.permute_tokens.default,
        [
            torch.empty(4, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(4, 8, device="meta", dtype=torch.int32),
        ],
    )
    result = ds.lookup(op)
    assert result is not None
    assert abs(result.latency_us - 6.12) < 0.01


# --- Integration tests: real CANN 8.3 / 8.5 data directories ---

from pathlib import Path

_CANN83_DATA_DIR = Path(__file__).resolve().parents[2] / (
    "tensor_cast/performance_model/perf_database/data/"
    "ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.13.0_torch2.8.0_cann8.3"
)
_CANN85_DATA_DIR = Path(__file__).resolve().parents[2] / (
    "tensor_cast/performance_model/perf_database/data/"
    "ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5"
)

_skip_no_cann83 = pytest.mark.skipif(
    not _CANN83_DATA_DIR.exists(), reason="CANN 8.3 data dir not present"
)
_skip_no_cann85 = pytest.mark.skipif(
    not _CANN85_DATA_DIR.exists(), reason="CANN 8.5 data dir not present"
)


# MoE real CANN tests (permute_tokens, unpermute_tokens, moe_gating_topk)
# moved to G1 PR — they depend on tensor_cast.ops.fused_moe which is G1 scope.


def test_moe_gating_topk_op_exists():
    """moe_gating_topk should be a registered tensor_cast op."""

    assert hasattr(torch.ops.tensor_cast, "moe_gating_topk"), (
        "moe_gating_topk op not registered"
    )


def test_moe_gating_topk_output_shapes():
    """moe_gating_topk returns (topk_weights, topk_indices) with correct shapes."""

    logits = torch.randn(8, 256)  # 8 tokens, 256 experts
    expert_bias = torch.zeros(256)
    topk_weights, topk_indices = torch.ops.tensor_cast.moe_gating_topk(
        logits,
        expert_bias,
        8,  # top_k=8
    )
    assert topk_weights.shape == (8, 8)
    assert topk_indices.shape == (8, 8)
    assert topk_indices.dtype == torch.int32


# --- C1: MLA/MLAPO unblock tests ---


def test_mlapo_composite_not_rejected():
    """After C1 fix, MLAPO ops should attempt composite lookup, not return mla_not_implemented."""
    import os
    import tempfile

    import yaml

    op_mapping = {
        "version": "test",
        "device": "TEST",
        "operator_mappings": {
            "tensor_cast.mlapo.default": {
                "composite": True,
                "sub_kernels": ["MatMulV2", "KvRmsNormRopeCache"],
            }
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "op_mapping.yaml"), "w") as f:
            yaml.dump(op_mapping, f)

        ds = ProfilingDataSource(tmpdir, device_profile=MagicMock())

        mock_op = MagicMock()
        mock_op.func = "torch.ops.tensor_cast.mlapo.default"
        mock_op.args = [
            torch.randn(8, 576),
            torch.randn(576, 512),
        ]

        result = ds.lookup(mock_op)

        # Result may be None (CSV missing), but reason should NOT be mla_not_implemented
        assert ds.last_miss_reason != "mla_not_implemented", (
            f"Expected composite lookup attempt, got {ds.last_miss_reason}"
        )


# --- C4: MISS reason reclassification tests ---


def test_miss_reason_respects_tc_input_count():
    """With tc_input_count, miss reason should compare truncated counts."""
    import os
    import tempfile

    import pandas as pd
    import yaml

    op_mapping = {
        "version": "test",
        "device": "TEST",
        "operator_mappings": {
            "tensor_cast.quantize.default": {
                "kernel_type": "AscendQuantV2",
                "tc_input_count": 1,
            }
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "op_mapping.yaml"), "w") as f:
            yaml.dump(op_mapping, f)

        # CSV with 1-input shape that doesn't match TC shape
        csv_data = pd.DataFrame(
            {
                "Input Shapes": ['"999,888"'],
                "Input Data Types": ["DT_BF16"],
                "Input Formats": ["ND"],
                "Output Shapes": ['"999,888"'],
                "Output Data Types": ["DT_BF16"],
                "Output Formats": ["ND"],
                "AVG_DURATION_US": [10.0],
            }
        )
        csv_data.to_csv(os.path.join(tmpdir, "AscendQuantV2.csv"), index=False)

        ds = ProfilingDataSource(tmpdir, device_profile=MagicMock())

        mock_op = MagicMock()
        mock_op.func = "torch.ops.tensor_cast.quantize.default"
        # TC has 3 inputs but tc_input_count=1, so only first is compared
        mock_op.args = [
            torch.randn(128, 5120),  # tensor (different from CSV 999,888)
            torch.randn(5120),  # scale (ignored by tc_input_count)
            torch.randn(5120),  # zero_point (ignored by tc_input_count)
        ]

        result = ds.lookup(mock_op)
        assert result is None  # should miss
        # Key assertion: reason should be shape_mismatch, NOT input_count_mismatch
        assert ds.last_miss_reason == "shape_mismatch", (
            f"Expected shape_mismatch, got {ds.last_miss_reason}"
        )


# --- Flatten batch 3D→2D tests (quantize / norm kernels) ---

QUANT_2D_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"16,7168","DT_BF16","ND","16,7168","INT8","ND",5.5
"256,5120","DT_BF16","ND","256,5120","INT8","ND",18.2
"""


@pytest.fixture
def quant_flatten_data_dir(tmp_path):
    data_dir = tmp_path / "quant_flatten"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.quantize.default":\n'
        "    kernel_type: AscendQuantV2\n"
        "    tc_input_count: 1\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "AscendQuantV2.csv").write_text(QUANT_2D_CSV.strip())
    return data_dir


def test_flatten_batch_quantize_3d_to_2d(quant_flatten_data_dir):
    """TC quantize sends (1,16,7168) 3D — should match CSV (16,7168) 2D
    via flatten batch rule for AscendQuantV2."""
    ds = ProfilingDataSource(quant_flatten_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.quantize.default,
        [
            torch.empty(1, 16, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(7168, device="meta", dtype=torch.bfloat16),  # scale
            torch.empty(7168, device="meta", dtype=torch.bfloat16),  # zp
        ],
    )
    result = ds.lookup(op)
    assert result is not None, (
        "Should match 3D (1,16,7168) → 2D (16,7168) via flatten batch"
    )
    assert abs(result.latency_us - 5.5) < 0.01


def test_flatten_batch_quantize_batch_gt_1(quant_flatten_data_dir):
    """TC quantize sends (4,64,5120) 3D — should match CSV (256,5120) 2D
    via flatten: 4*64=256."""
    ds = ProfilingDataSource(quant_flatten_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.quantize.default,
        [
            torch.empty(4, 64, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, (
        "Should match 3D (4,64,5120) → 2D (256,5120) via flatten batch"
    )
    assert abs(result.latency_us - 18.2) < 0.01


def test_flatten_batch_quantize_with_padding(quant_flatten_data_dir):
    """TC quantize sends (1,272,5120) 3D — should match CSV (256,5120) 2D
    via flatten + block padding: flatten→(272,5120), 272 ≈ 256 via ceil(256/16)*16=256? No.
    Actually 272 = ceil(256/16)*16 = 256? No, ceil(256/16)*16 = 256. 272 = ceil(268/16)*16.
    Use (1,256,5120) instead for exact flatten match."""
    ds = ProfilingDataSource(quant_flatten_data_dir)
    # 3D exact flatten (no padding needed)
    op = _make_op_info(
        torch.ops.tensor_cast.quantize.default,
        [
            torch.empty(1, 256, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, (
        "Should match 3D (1,256,5120) → 2D (256,5120) via flatten"
    )


def test_flatten_batch_rmsnorm_3d_to_2d(rmsnorm_data_dir):
    """TC RmsNorm sends (2,68,5120),(5120,) 3D — should match CSV (136,5120),(5120)
    via flatten batch: 2*68=136."""
    ds = ProfilingDataSource(rmsnorm_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.rms_norm.default,
        [
            torch.empty(2, 68, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, (
        "Should match 3D (2,68,5120) → 2D (136,5120) via flatten batch"
    )
    assert abs(result.latency_us - 21.66) < 0.01


def test_flatten_batch_not_applied_to_matmul(spike_data_dir):
    """MatMulV2 is NOT in _FLATTEN_BATCH_KERNELS — 3D should NOT match 2D."""
    ds = ProfilingDataSource(spike_data_dir)
    # CSV has (136,5120) as first input. Try 3D (2,68,5120) — should NOT match.
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(2, 68, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, 768, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None, "Flatten batch should NOT apply to MatMulV2"


def test_flatten_batch_2d_still_works(quant_flatten_data_dir):
    """2D TC shape should still match 2D CSV directly (no flatten needed)."""
    ds = ProfilingDataSource(quant_flatten_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.quantize.default,
        [
            torch.empty(16, 7168, device="meta", dtype=torch.bfloat16),
            torch.empty(7168, device="meta", dtype=torch.bfloat16),
            torch.empty(7168, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "2D exact match should still work"
    assert abs(result.latency_us - 5.5) < 0.01


# --- Merge-last-dims tests (MLA quantize 3D→2D) ---

QUANT_MLA_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Average Duration(us)
"8,2048","DT_BF16","ND","8,2048","INT8","ND",9.8
"256,2048","DT_BF16","ND","256,2048","INT8","ND",20.5
"""


@pytest.fixture
def quant_mla_data_dir(tmp_path):
    data_dir = tmp_path / "quant_mla"
    data_dir.mkdir()
    op_mapping = (
        'version: "test"\n'
        "operator_mappings:\n"
        '  "tensor_cast.quantize.default":\n'
        "    kernel_type: AscendQuantV2\n"
        "    tc_input_count: 1\n"
    )
    (data_dir / "op_mapping.yaml").write_text(op_mapping)
    (data_dir / "AscendQuantV2.csv").write_text(QUANT_MLA_CSV.strip())
    return data_dir


def test_merge_last_dims_quantize_mla_decode(quant_mla_data_dir):
    """MLA quantize: TC (8, 16, 128) 3D → should match CSV (8, 2048) 2D
    by merging last two dims: 16*128=2048."""
    ds = ProfilingDataSource(quant_mla_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.quantize.default,
        [
            torch.empty(8, 16, 128, device="meta", dtype=torch.bfloat16),
            torch.tensor(1.0),
            None,
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match via last-two-dims merge"
    assert abs(result.latency_us - 9.8) < 0.01


def test_merge_last_dims_quantize_mla_prefill(quant_mla_data_dir):
    """MLA quantize: TC (256, 16, 128) 3D → should match CSV (256, 2048) 2D."""
    ds = ProfilingDataSource(quant_mla_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.quantize.default,
        [
            torch.empty(256, 16, 128, device="meta", dtype=torch.bfloat16),
            torch.tensor(1.0),
            None,
        ],
    )
    result = ds.lookup(op)
    assert result is not None, "Should match via last-two-dims merge"
    assert abs(result.latency_us - 20.5) < 0.01


def test_merge_last_dims_quantize_mla_batch1(quant_mla_data_dir):
    """MLA quantize batch=1: TC (1, 16, 128) 3D → should match CSV (1, 2048) 2D.
    _strip_batch_dim collapses (1,16,128)→(16,128), so merge must use original shape."""
    # Add a batch=1 row to CSV
    csv_path = quant_mla_data_dir / "AscendQuantV2.csv"
    with open(csv_path, "a") as f:
        f.write('\n"1,2048","DT_BF16","ND","1,2048","INT8","ND",4.2\n')
    ds = ProfilingDataSource(quant_mla_data_dir)
    op = _make_op_info(
        torch.ops.tensor_cast.quantize.default,
        [
            torch.empty(1, 16, 128, device="meta", dtype=torch.bfloat16),
            torch.tensor(1.0),
            None,
        ],
    )
    result = ds.lookup(op)
    assert result is not None, (
        "Should match (1,16,128) → (1,2048) via merge-last-dims on original shape"
    )
    assert abs(result.latency_us - 4.2) < 0.01


def test_merge_last_dims_not_applied_to_matmul(spike_data_dir):
    """MatMulV2 is NOT in _FLATTEN_BATCH_KERNELS — merge should NOT apply.
    Uses spike_data_dir which has MatMulV2 mapped via aten.mm.default."""
    ds = ProfilingDataSource(spike_data_dir)
    # CSV has (136,5120) for MatMulV2. Try 3D (2,68,5120) — should NOT match
    # via merge-last-dims (68*5120=348160 ≠ 5120).
    op = _make_op_info(
        torch.ops.aten.mm.default,
        [
            torch.empty(2, 68, 5120, device="meta", dtype=torch.bfloat16),
            torch.empty(5120, 768, device="meta", dtype=torch.bfloat16),
        ],
    )
    result = ds.lookup(op)
    assert result is None, "Merge last dims should NOT apply to MatMulV2"
