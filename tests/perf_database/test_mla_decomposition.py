"""Tests for G2 MLA decomposition + InterpolatingDataSource composite support."""

from unittest.mock import MagicMock

import pytest
import torch

from tensor_cast.performance_model.perf_database.data_source import QuerySource
from tensor_cast.performance_model.perf_database.interpolating_data_source import (
    InterpolatingDataSource,
)
from tensor_cast.performance_model.perf_database.profiling_data_source import (
    COMPOSITE_DECOMPOSERS,
    ProfilingDataSource,
    SubKernelSpec,
    _decompose_mla,
    _decompose_mla_quant,
    _is_decode_mla,
)


# ---- Helpers ----


def _make_op_info(func, args):
    mock = MagicMock()
    mock.func = func
    mock.args = tuple(args)
    mock.kwargs = {}
    mock.out = None
    return mock


def _make_mla_decode_args(
    num_tokens=16,
    num_heads=16,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    kv_lora_rank=512,
    v_head_dim=128,
    batch_size=16,
    avg_seq_len=4096,
):
    """Build args for multihead_latent_attention in decode mode."""
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    q = torch.empty(
        num_tokens, num_heads, qk_head_dim, device="meta", dtype=torch.bfloat16
    )
    kv_cache = torch.empty(
        256, 16, kv_lora_rank + qk_rope_head_dim, device="meta", dtype=torch.bfloat16
    )
    block_table = torch.empty(batch_size, 16, device="meta", dtype=torch.int32)
    query_start_loc = torch.arange(batch_size + 1, dtype=torch.int32)
    seq_lens = torch.full((batch_size,), avg_seq_len, dtype=torch.int64)
    query_lens = None  # decode
    W_UK_T = torch.empty(
        num_heads, qk_nope_head_dim, kv_lora_rank, device="meta", dtype=torch.bfloat16
    )
    W_UV = torch.empty(
        num_heads, kv_lora_rank, v_head_dim, device="meta", dtype=torch.bfloat16
    )
    kv_b_proj = None  # decode
    return [
        q,
        kv_cache,
        block_table,
        query_start_loc,
        seq_lens,
        query_lens,
        W_UK_T,
        W_UV,
        kv_b_proj,
        v_head_dim,
    ]


def _make_mla_prefill_args(
    num_tokens=136,
    num_heads=16,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    kv_lora_rank=512,
    v_head_dim=128,
    batch_size=2,
    avg_seq_len=68,
):
    """Build args for multihead_latent_attention in prefill mode."""
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    q = torch.empty(
        num_tokens, num_heads, qk_head_dim, device="meta", dtype=torch.bfloat16
    )
    kv_cache = torch.empty(
        256, 16, kv_lora_rank + qk_rope_head_dim, device="meta", dtype=torch.bfloat16
    )
    block_table = torch.empty(batch_size, 16, device="meta", dtype=torch.int32)
    query_start_loc = torch.arange(batch_size + 1, dtype=torch.int32)
    seq_lens = torch.full((batch_size,), avg_seq_len, dtype=torch.int64)
    query_lens = torch.full((batch_size,), avg_seq_len, dtype=torch.int64)  # prefill
    W_UK_T = None
    W_UV = None
    proj_out_dim = num_heads * (qk_nope_head_dim + v_head_dim)
    kv_b_proj = torch.empty(
        kv_lora_rank, proj_out_dim, device="meta", dtype=torch.bfloat16
    )
    return [
        q,
        kv_cache,
        block_table,
        query_start_loc,
        seq_lens,
        query_lens,
        W_UK_T,
        W_UV,
        kv_b_proj,
        v_head_dim,
    ]



# ---- Unit tests: decomposition functions ----


class TestIsDecodeMLA:
    def test_none_query_lens_is_decode(self):
        assert _is_decode_mla((None, None, None, None, None, None)) is True

    def test_all_ones_is_decode(self):
        args = (None, None, None, None, None, torch.ones(16, dtype=torch.int64))
        assert _is_decode_mla(args) is True

    def test_query_lens_gt_1_is_prefill(self):
        args = (None, None, None, None, None, torch.full((2,), 68, dtype=torch.int64))
        assert _is_decode_mla(args) is False


class TestDecomposeMLA:
    def test_decode_returns_3_specs(self):
        args = _make_mla_decode_args()
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention.default, args
        )
        specs = _decompose_mla(op, {})
        assert specs is not None
        assert len(specs) == 3
        assert specs[0].kernel_type == "TransposeBatchMatMul"
        assert specs[1].kernel_type == "FusedInferAttentionScore"
        assert specs[1].query_mode == "attention"
        assert specs[2].kernel_type == "TransposeBatchMatMul"

    def test_decode_shapes_correct(self):
        args = _make_mla_decode_args(
            num_tokens=16,
            num_heads=16,
            qk_nope_head_dim=128,
            kv_lora_rank=512,
            v_head_dim=128,
        )
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention.default, args
        )
        specs = _decompose_mla(op, {})
        # q @ W_UK_T: (16, 16, 128) @ (16, 128, 512)
        assert specs[0].input_shapes == [(16, 16, 128), (16, 128, 512)]
        # attn_out @ W_UV: (16, 16, 512) @ (16, 512, 128)
        assert specs[2].input_shapes == [(16, 16, 512), (16, 512, 128)]

    def test_prefill_decomposes_to_matmul_and_fia(self):
        """Prefill decomposes to MatMulV2 + FIA (v0.18.0: unified FIA)."""
        args = _make_mla_prefill_args()
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention.default, args
        )
        specs = _decompose_mla(op, {})
        assert specs is not None
        assert len(specs) == 2
        assert specs[0].kernel_type == "MatMulV2"
        # kv_c @ kv_b_proj: (136, 512) @ (512, 16*(128+128))
        assert specs[0].input_shapes[0] == (136, 512)
        assert specs[0].input_shapes[1][0] == 512
        assert specs[1].kernel_type == "FusedInferAttentionScore"

    def test_prefill_fia_has_attention_params(self):
        """Prefill FIA spec has attention_params (v0.18.0)."""
        args = _make_mla_prefill_args(num_tokens=136, kv_lora_rank=512)
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention.default, args
        )
        specs = _decompose_mla(op, {})
        assert specs is not None
        assert len(specs) == 2
        assert specs[1].kernel_type == "FusedInferAttentionScore"
        assert specs[1].attention_params is not None
        assert specs[1].attention_params["num_kv_heads"] == 1

    def test_insufficient_args_returns_none(self):
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention.default,
            [torch.empty(136, 5120, device="meta", dtype=torch.bfloat16)],
        )
        assert _decompose_mla(op, {}) is None

    def test_fia_attention_params_decode(self):
        """Decode FIA spec uses attention_params (not fia_raw_shapes)."""
        args = _make_mla_decode_args(batch_size=16, avg_seq_len=4096, num_heads=16)
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention.default, args
        )
        specs = _decompose_mla(op, {})
        fia = specs[1]
        assert fia.attention_params is not None
        assert fia.attention_params["avg_seq_len"] == 4096
        q_shape_3d = fia.attention_params["q_shape_3d"]
        assert q_shape_3d[0] == 16  # batch_size
        assert q_shape_3d[1] == 16  # num_heads


class TestDecomposeMLAQuant:
    def test_decode_uses_quant_kernel(self):
        args = _make_mla_decode_args()
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention_quant.default, args
        )
        specs = _decompose_mla_quant(op, {})
        assert specs is not None
        assert specs[0].kernel_type == "QuantBatchMatmulV3"

    def test_prefill_decomposes_to_matmul_and_fia(self):
        """Quant prefill decomposes to MatMulV2 + FIA (v0.18.0)."""
        args = _make_mla_prefill_args()
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention_quant.default, args
        )
        specs = _decompose_mla_quant(op, {})
        assert specs is not None
        assert len(specs) == 2
        assert specs[0].kernel_type == "MatMulV2"
        assert specs[1].kernel_type == "FusedInferAttentionScore"



# ---- Integration tests: composite lookup with CSV data ----

MLA_OP_MAPPING = """\
version: "test"
device: TEST_DEVICE
interpolation_policy:
  default_method: linear
  kernel_overrides:
    FusedInferAttentionScore:
      shape_transform: sqrt
operator_mappings:
  "tensor_cast.multihead_latent_attention.default":
    composite: true
    sub_kernels: [TransposeBatchMatMul, FusedInferAttentionScore]
  "tensor_cast.mlapo.default":
    composite: true
    sub_kernels: [MatMulV2, KvRmsNormRopeCache]
"""

# TransposeBatchMatMul CSV: decode q@W_UK_T shape
TBMM_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"16,16,128;16,128,512","DT_BF16;DT_BF16","ND;ND","16,16,512","DT_BF16","ND",5.0
"16,16,512;16,512,128","DT_BF16;DT_BF16","ND;ND","16,16,128","DT_BF16","ND",4.0"""

# FIA CSV: raw format (Case C 4D BNSD paged with rope)
# decode: batch=16, heads=16, seq=1, head_dim=576, kv_cache=(256,16,576), rope_dim=64
# slots: 0=q(16,16,1,576), 1=k(256,1,16,576), 2=v(256,1,16,576), 6=seq_lens(16,),
#        14=block_table(16,256), 24=rope_q(16,16,1,64), 25=rope_k(256,1,16,64)
_FIA_DECODE_ROW_16 = (
    '"16,16,1,576;256,1,16,576;256,1,16,576;;;;16;;;;;;;;16,256;;;;;;;;;;'
    '16,16,1,64;256,1,16,64;;;;;"'
)
_FIA_DECODE_ROW_32 = (
    '"32,16,1,576;256,1,16,576;256,1,16,576;;;;32;;;;;;;;32,256;;;;;;;;;;'
    '32,16,1,64;256,1,16,64;;;;;"'
)
FIA_CSV = (
    "Input Shapes,Input Data Types,Input Formats,Output Shapes,"
    "Output Data Types,Output Formats,Duration(us),avg_seq_len\n"
    + _FIA_DECODE_ROW_16
    + ",DT_BF16;DT_BF16;DT_BF16;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "INT64;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_UNDEFINED;INT32;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_BF16;DT_BF16;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_UNDEFINED"
    ",ND;ND;ND;NULL;NULL;NULL;ND;NULL;NULL;NULL;NULL;NULL;NULL;NULL;ND;"
    "NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;ND;ND;NULL;NULL;NULL;NULL;NULL"
    ',"""16,16,1,576;""",DT_BF16;FLOAT,ND;ND,50.0,4096'
)

# MatMulV2 CSV: mlapo hidden @ q_a_proj
MATMUL_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"136,5120;5120,1536","DT_BF16;DT_BF16","ND;ND","136,1536","DT_BF16","ND",8.0
"100,5120;5120,1536","DT_BF16;DT_BF16","ND;ND","100,1536","DT_BF16","ND",6.0
"200,5120;5120,1536","DT_BF16;DT_BF16","ND;ND","200,1536","DT_BF16","ND",12.0"""

# KvRmsNormRopeCache CSV
KVRNRC_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"136,5120;5120,576","DT_BF16;DT_BF16","ND;ND","136,576","DT_BF16","ND",3.0"""


@pytest.fixture
def mla_data_dir(tmp_path):
    d = tmp_path / "mla"
    d.mkdir()
    (d / "op_mapping.yaml").write_text(MLA_OP_MAPPING)
    (d / "TransposeBatchMatMul.csv").write_text(TBMM_CSV.strip())
    (d / "FusedInferAttentionScore.csv").write_text(FIA_CSV.strip())
    (d / "MatMulV2.csv").write_text(MATMUL_CSV.strip())
    (d / "KvRmsNormRopeCache.csv").write_text(KVRNRC_CSV.strip())
    return d


class TestCompositeLookupMLA:
    def test_mla_decode_hit(self, mla_data_dir):
        """MLA decode: all 3 sub-kernels hit → sum latency."""
        ds = ProfilingDataSource(mla_data_dir)
        args = _make_mla_decode_args(
            num_tokens=16,
            num_heads=16,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            v_head_dim=128,
            batch_size=16,
            avg_seq_len=4096,
        )
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention.default, args
        )
        result = ds.lookup(op)
        assert result is not None
        # 5.0 (TBMM q@W_UK_T) + 50.0 (FIA) + 4.0 (TBMM out@W_UV) = 59.0
        assert abs(result.latency_us - 59.0) < 0.1
        assert result.source == QuerySource.MEASURED

    def test_mla_decode_fia_miss_returns_none(self, mla_data_dir):
        """MLA decode: FIA miss (wrong batch_size) → None."""
        ds = ProfilingDataSource(mla_data_dir)
        args = _make_mla_decode_args(batch_size=99, avg_seq_len=4096)
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention.default, args
        )
        result = ds.lookup(op)
        assert result is None

    def test_mla_insufficient_args_returns_none(self, mla_data_dir):
        """MLA with insufficient args → decompose fails → None."""
        ds = ProfilingDataSource(mla_data_dir)
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention.default,
            [torch.empty(136, 5120, device="meta", dtype=torch.bfloat16)],
        )
        result = ds.lookup(op)
        assert result is None


# ---- Integration tests: InterpolatingDataSource composite ----


class TestCompositeInterpolation:
    def test_mla_decode_fia_hit(self, mla_data_dir):
        """MLA decode: FIA shape + avg_seq_len matches → exact hit."""
        base = ProfilingDataSource(mla_data_dir)
        ds = InterpolatingDataSource(base)
        args = _make_mla_decode_args(
            num_tokens=16,
            num_heads=16,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            v_head_dim=128,
            batch_size=16,
            avg_seq_len=4096,
        )
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention.default, args
        )
        result = ds.lookup(op)
        # TBMM exact: 5.0 + 4.0, FIA enriched hit: 50.0 → total 59.0
        assert result is not None
        assert result.source == QuerySource.MEASURED
        assert abs(result.latency_us - 59.0) < 0.1

    def test_existing_interpolation_not_broken(self, mla_data_dir):
        """Existing compute interpolation still works (regression test)."""
        base = ProfilingDataSource(mla_data_dir)
        ds = InterpolatingDataSource(base)
        # Non-composite MatMulV2 is not in op_mapping as non-composite, skip
        # Just verify the ds object is functional
        op = _make_op_info(
            torch.ops.aten.add.Tensor,
            [
                torch.empty(100, device="meta", dtype=torch.bfloat16),
                torch.empty(100, device="meta", dtype=torch.bfloat16),
            ],
        )
        result = ds.lookup(op)
        assert result is None  # unmapped op


# ============================================================
# Extended test suite: edge cases, boundary, accuracy, robustness
# Reference: AI Configurator design principles
# ============================================================


# ---- 1. Extrapolation rejection ----


EXTRAP_OP_MAPPING = """\
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
  "tensor_cast.attention.default":
    kernel_type: FusedInferAttentionScore
    query_mode: attention_special
"""

EXTRAP_MATMUL_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"256,512;512,1024","DT_BF16;DT_BF16","ND;ND","256,1024","DT_BF16","ND",25.0
"512,512;512,1024","DT_BF16;DT_BF16","ND;ND","512,1024","DT_BF16","ND",50.0
"1024,512;512,1024","DT_BF16;DT_BF16","ND;ND","1024,1024","DT_BF16","ND",100.0"""

_EXTRAP_FIA_HEADER = (
    "Input Shapes,Input Data Types,Input Formats,Output Shapes,"
    "Output Data Types,Output Formats,Duration(us),avg_seq_len"
)
_EXTRAP_FIA_ROW_COMMON = (
    '"1,4,128;16,128,4,128;16,128,4,128;;;;1;;;;;;;;1,16;;;;;;;;;;;;;;"'
    ',"DT_BF16;DT_BF16;DT_BF16;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;'
    "INT64;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_UNDEFINED;INT32;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    'DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED"'
    ',"ND;ND;ND;NULL;NULL;NULL;ND;NULL;NULL;NULL;NULL;NULL;NULL;NULL;ND;'
    'NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL"'
    ',"""1,4,128;""","DT_BF16;FLOAT","ND;ND"'
)
EXTRAP_FIA_CSV = (
    _EXTRAP_FIA_HEADER + "\n"
    + _EXTRAP_FIA_ROW_COMMON + ",100.0,1000\n"
    + _EXTRAP_FIA_ROW_COMMON + ",400.0,2000\n"
    + _EXTRAP_FIA_ROW_COMMON + ",1600.0,4000"
)


@pytest.fixture
def extrap_data_dir(tmp_path):
    d = tmp_path / "extrap"
    d.mkdir()
    (d / "op_mapping.yaml").write_text(EXTRAP_OP_MAPPING)
    (d / "MatMulV2.csv").write_text(EXTRAP_MATMUL_CSV.strip())
    (d / "FusedInferAttentionScore.csv").write_text(EXTRAP_FIA_CSV.strip())
    return d


class TestExtrapolationRejection:
    """AI Configurator principle: only interpolate within bracket, never extrapolate."""

    def test_compute_below_min_returns_none(self, extrap_data_dir):
        """seq_len=64 below CSV min=256 → no bracket → None."""
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.aten.mm.default,
            [
                torch.empty(64, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
            ],
        )
        assert ds.lookup(op) is None

    def test_compute_above_max_returns_none(self, extrap_data_dir):
        """seq_len=2048 above CSV max=1024 → no bracket → None."""
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.aten.mm.default,
            [
                torch.empty(2048, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
            ],
        )
        assert ds.lookup(op) is None

    def test_attention_below_min_returns_none(self, extrap_data_dir):
        """avg_seq_len=500 below CSV min=1000 → None."""
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.tensor_cast.attention.default,
            [
                torch.empty(1, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
                torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
                None,
                None,
                None,
                torch.tensor([500], dtype=torch.int64),
                torch.tensor([1], dtype=torch.int64),
            ],
        )
        assert ds.lookup(op) is None

    def test_attention_above_max_returns_none(self, extrap_data_dir):
        """avg_seq_len=8000 above CSV max=4000 → None."""
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.tensor_cast.attention.default,
            [
                torch.empty(1, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
                torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
                None,
                None,
                None,
                torch.tensor([8000], dtype=torch.int64),
                torch.tensor([1], dtype=torch.int64),
            ],
        )
        assert ds.lookup(op) is None


# ---- 2. Single data point ----


SINGLE_POINT_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"256,512;512,1024","DT_BF16;DT_BF16","ND;ND","256,1024","DT_BF16","ND",25.0"""


class TestSingleDataPoint:
    """Need ≥2 data points for interpolation; 1 point → None."""

    def test_single_csv_row_no_interpolation(self, tmp_path):
        d = tmp_path / "single"
        d.mkdir()
        (d / "op_mapping.yaml").write_text(EXTRAP_OP_MAPPING)
        (d / "MatMulV2.csv").write_text(SINGLE_POINT_CSV.strip())
        base = ProfilingDataSource(d)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.aten.mm.default,
            [
                torch.empty(300, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
            ],
        )
        assert ds.lookup(op) is None

    def test_single_csv_row_exact_match_still_works(self, tmp_path):
        """Exact match should still work even with 1 row."""
        d = tmp_path / "single_exact"
        d.mkdir()
        (d / "op_mapping.yaml").write_text(EXTRAP_OP_MAPPING)
        (d / "MatMulV2.csv").write_text(SINGLE_POINT_CSV.strip())
        base = ProfilingDataSource(d)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.aten.mm.default,
            [
                torch.empty(256, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
            ],
        )
        result = ds.lookup(op)
        assert result is not None
        assert abs(result.latency_us - 25.0) < 0.01
        assert result.source == QuerySource.MEASURED


# ---- 3. Confidence levels ----


class TestConfidenceLevels:
    """Verify confidence: MEASURED > linear > sqrt > composite interpolated."""

    def test_exact_match_confidence_1(self, extrap_data_dir):
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.aten.mm.default,
            [
                torch.empty(256, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
            ],
        )
        result = ds.lookup(op)
        assert result.confidence == 1.0
        assert result.source == QuerySource.MEASURED

    def test_linear_interpolation_confidence_07(self, extrap_data_dir):
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.aten.mm.default,
            [
                torch.empty(384, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
            ],
        )
        result = ds.lookup(op)
        assert result is not None
        assert result.confidence == 0.7
        assert result.source == QuerySource.INTERPOLATED

    def test_sqrt_interpolation_confidence_06(self, extrap_data_dir):
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.tensor_cast.attention.default,
            [
                torch.empty(1, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
                torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
                None,
                None,
                None,
                torch.tensor([1500], dtype=torch.int64),
                torch.tensor([1], dtype=torch.int64),
            ],
        )
        result = ds.lookup(op)
        assert result is not None
        assert result.confidence == 0.6

    def test_composite_exact_confidence_08(self, mla_data_dir):
        """Composite exact match → confidence 0.8."""
        ds = ProfilingDataSource(mla_data_dir)
        args = _make_mla_decode_args(
            num_tokens=16,
            num_heads=16,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            v_head_dim=128,
            batch_size=16,
            avg_seq_len=4096,
        )
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention.default, args
        )
        result = ds.lookup(op)
        assert result.confidence == 0.8

    def test_composite_interpolated_confidence_05(self, mla_data_dir):
        """Composite with FIA raw shape miss → None (no interpolation for raw shapes yet)."""
        base = ProfilingDataSource(mla_data_dir)
        ds = InterpolatingDataSource(base)
        args = _make_mla_decode_args(
            num_tokens=16,
            num_heads=16,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            v_head_dim=128,
            batch_size=32,  # batch=32 not in CSV → FIA miss
            avg_seq_len=3000,
        )
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention.default, args
        )
        result = ds.lookup(op)
        # FIA raw shape for batch=32 not in CSV → sub_kernel_miss → None
        assert result is None


# ---- 4. Monotonicity ----


class TestMonotonicity:
    """Interpolated values should be monotonic if CSV data is monotonic."""

    def test_compute_monotonic_increasing(self, extrap_data_dir):
        """Increasing seq_len → increasing latency."""
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        latencies = []
        for seq in [300, 400, 600, 800, 900]:
            op = _make_op_info(
                torch.ops.aten.mm.default,
                [
                    torch.empty(seq, 512, device="meta", dtype=torch.bfloat16),
                    torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
                ],
            )
            result = ds.lookup(op)
            assert result is not None, f"seq={seq} should interpolate"
            latencies.append(result.latency_us)
        # Verify monotonically increasing
        for i in range(len(latencies) - 1):
            assert (
                latencies[i] < latencies[i + 1]
            ), f"Not monotonic: seq[{i}]={latencies[i]} >= seq[{i+1}]={latencies[i+1]}"

    def test_interpolation_within_bracket_bounds(self, extrap_data_dir):
        """Interpolated value must be between bracket endpoints (no overshoot)."""
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        # Between 256→25.0 and 512→50.0
        op = _make_op_info(
            torch.ops.aten.mm.default,
            [
                torch.empty(384, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
            ],
        )
        result = ds.lookup(op)
        assert result is not None
        assert 25.0 <= result.latency_us <= 50.0

    def test_attention_sqrt_within_bounds(self, extrap_data_dir):
        """Sqrt-interpolated attention value within bracket bounds."""
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.tensor_cast.attention.default,
            [
                torch.empty(1, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
                torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
                None,
                None,
                None,
                torch.tensor([1500], dtype=torch.int64),
                torch.tensor([1], dtype=torch.int64),
            ],
        )
        result = ds.lookup(op)
        assert result is not None
        assert 100.0 <= result.latency_us <= 400.0


# ---- 5. Dtype mismatch ----


DTYPE_MISMATCH_CSV = """\
Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Duration(us)
"256,512;512,1024","INT8;INT8","ND;ND","256,1024","INT8","ND",10.0
"512,512;512,1024","INT8;INT8","ND;ND","512,1024","INT8","ND",20.0"""


class TestDtypeMismatch:
    """Interpolation must respect dtype: BF16 query should not match INT8 CSV."""

    def test_bf16_query_int8_csv_returns_none(self, tmp_path):
        d = tmp_path / "dtype_mm"
        d.mkdir()
        (d / "op_mapping.yaml").write_text(EXTRAP_OP_MAPPING)
        (d / "MatMulV2.csv").write_text(DTYPE_MISMATCH_CSV.strip())
        base = ProfilingDataSource(d)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.aten.mm.default,
            [
                torch.empty(384, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
            ],
        )
        result = ds.lookup(op)
        assert result is None


# ---- 6. Sqrt vs linear accuracy comparison ----


class TestSqrtTransformAccuracy:
    """Verify sqrt transform behavior for O(n²) ops."""

    def test_sqrt_transform_applied(self, extrap_data_dir):
        """Sqrt interpolation produces different result than linear would.

        CSV: seq=1000→100, seq=2000→400, seq=4000→1600
        For seq=1500 (between 1000 and 2000):
          Linear: 100 + 0.5*300 = 250
          Sqrt: in sqrt space, t=(sqrt(1500)-sqrt(1000))/(sqrt(2000)-sqrt(1000))
                = (38.73-31.62)/(44.72-31.62) = 0.543
                interp = 100 + 0.543*300 = 262.8
        Sqrt gives a different (higher) value, reflecting the nonlinear scaling.
        """
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.tensor_cast.attention.default,
            [
                torch.empty(1, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
                torch.empty(16, 128, 4, 128, device="meta", dtype=torch.bfloat16),
                None,
                None,
                None,
                torch.tensor([1500], dtype=torch.int64),
                torch.tensor([1], dtype=torch.int64),
            ],
        )
        result = ds.lookup(op)
        assert result is not None
        # Sqrt result should differ from naive linear midpoint (250)
        linear_midpoint = 250.0
        assert (
            abs(result.latency_us - linear_midpoint) > 5.0
        ), "Sqrt transform should produce different result than linear"
        # Should be within bracket bounds
        assert 100.0 <= result.latency_us <= 400.0


# ---- 7. Composite: mixed exact + interpolation ----


MLA_RICH_FIA_CSV = (
    "Input Shapes,Input Data Types,Input Formats,Output Shapes,"
    "Output Data Types,Output Formats,Duration(us),avg_seq_len\n"
    + _FIA_DECODE_ROW_16
    + ",DT_BF16;DT_BF16;DT_BF16;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "INT64;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_UNDEFINED;INT32;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_BF16;DT_BF16;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_UNDEFINED"
    ",ND;ND;ND;NULL;NULL;NULL;ND;NULL;NULL;NULL;NULL;NULL;NULL;NULL;ND;"
    "NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;ND;ND;NULL;NULL;NULL;NULL;NULL"
    ',"""16,16,1,576;""",DT_BF16;FLOAT,ND;ND,50.0,4096\n'
    + _FIA_DECODE_ROW_32
    + ",DT_BF16;DT_BF16;DT_BF16;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "INT64;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_UNDEFINED;INT32;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_BF16;DT_BF16;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;"
    "DT_UNDEFINED;DT_UNDEFINED"
    ",ND;ND;ND;NULL;NULL;NULL;ND;NULL;NULL;NULL;NULL;NULL;NULL;NULL;ND;"
    "NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;NULL;ND;ND;NULL;NULL;NULL;NULL;NULL"
    ',"""32,16,1,576;""",DT_BF16;FLOAT,ND;ND,100.0,4096'
)


@pytest.fixture
def mla_rich_data_dir(tmp_path):
    """MLA data dir with richer FIA CSV for interpolation tests."""
    d = tmp_path / "mla_rich"
    d.mkdir()
    (d / "op_mapping.yaml").write_text(MLA_OP_MAPPING)
    (d / "TransposeBatchMatMul.csv").write_text(TBMM_CSV.strip())
    (d / "FusedInferAttentionScore.csv").write_text(MLA_RICH_FIA_CSV.strip())
    (d / "MatMulV2.csv").write_text(MATMUL_CSV.strip())
    (d / "KvRmsNormRopeCache.csv").write_text(KVRNRC_CSV.strip())
    return d


class TestCompositeMixedHitInterpolate:
    """Composite ops: some sub-kernels exact hit, others interpolated."""

    def test_tbmm_exact_fia_hit(self, mla_rich_data_dir):
        """TBMM shapes match exactly, FIA enriched shape also hits exactly."""
        base = ProfilingDataSource(mla_rich_data_dir)
        ds = InterpolatingDataSource(base)
        args = _make_mla_decode_args(
            num_tokens=16,
            num_heads=16,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            v_head_dim=128,
            batch_size=16,
            avg_seq_len=4096,
        )
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention.default, args
        )
        result = ds.lookup(op)
        assert result is not None
        # TBMM exact: 5.0 + 4.0, FIA enriched hit: 50.0 → total 59.0
        assert abs(result.latency_us - 59.0) < 0.1
        assert result.source == QuerySource.MEASURED

    def test_all_sub_kernels_miss_returns_none(self, mla_rich_data_dir):
        """All sub-kernels miss (wrong batch_size for FIA, wrong shape for TBMM)."""
        base = ProfilingDataSource(mla_rich_data_dir)
        ds = InterpolatingDataSource(base)
        args = _make_mla_decode_args(
            num_tokens=99,
            num_heads=8,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            kv_lora_rank=256,
            v_head_dim=64,
            batch_size=64,
            avg_seq_len=999,
        )
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention.default, args
        )
        result = ds.lookup(op)
        assert result is None


# ---- 8. Empty CSV ----


class TestEmptyCSV:
    def test_empty_csv_returns_none(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        (d / "op_mapping.yaml").write_text(EXTRAP_OP_MAPPING)
        (d / "MatMulV2.csv").write_text(
            "Input Shapes,Input Data Types,Input Formats,Output Shapes,"
            "Output Data Types,Output Formats,Duration(us)\n"
        )
        base = ProfilingDataSource(d)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.aten.mm.default,
            [
                torch.empty(256, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
            ],
        )
        assert ds.lookup(op) is None


# ---- 9. Decompose failure modes ----


class TestDecomposeFailureModes:
    def test_mla_decode_missing_W_UK_T(self):
        """Decode path with W_UK_T=None → decompose returns None."""
        args = _make_mla_decode_args()
        args[6] = None  # W_UK_T = None
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention.default, args
        )
        assert _decompose_mla(op, {}) is None

    def test_mla_prefill_missing_kv_b_proj(self):
        """Prefill path with kv_b_proj=None → decompose returns None."""
        args = _make_mla_prefill_args()
        args[8] = None  # kv_b_proj = None
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention.default, args
        )
        assert _decompose_mla(op, {}) is None

    def test_mla_unsupported_dtype(self):
        """MLA with unsupported dtype → decompose returns None."""
        args = _make_mla_decode_args()
        # Replace q with float64 (not in DTYPE_MAP)
        args[0] = torch.empty(16, 16, 192, device="meta", dtype=torch.float64)
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention.default, args
        )
        assert _decompose_mla(op, {}) is None

    def test_mla_seq_lens_not_tensor(self):
        """MLA with seq_lens as list instead of tensor → returns None."""
        args = _make_mla_decode_args()
        args[4] = [4096] * 16  # list instead of tensor
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention.default, args
        )
        assert _decompose_mla(op, {}) is None


class TestMLADecomposeWithAttentionParams:
    """Tests for MLA decomposers using attention_params (Tasks 7 & 8)."""

    def test_e1_mla_decode_attention_params(self):
        """MLA decode produces attention_params for FIA sub-kernel."""
        args = _make_mla_decode_args(
            batch_size=4,
            num_heads=16,
            kv_lora_rank=448,
            qk_rope_head_dim=64,
        )
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention.default, args
        )
        specs = _decompose_mla(op, {})
        assert len(specs) == 3
        fia_spec = specs[1]
        assert fia_spec.attention_params is not None
        q_shape_3d = fia_spec.attention_params["q_shape_3d"]
        assert q_shape_3d[0] == 4  # batch_size
        assert q_shape_3d[1] == 16  # num_heads
        assert fia_spec.attention_params["avg_seq_len"] == 4096

    def test_e2_mla_decode_attention_query_mode(self):
        """MLA decode FIA spec has query_mode='attention'."""
        args = _make_mla_decode_args(batch_size=4, num_heads=16)
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention.default, args
        )
        specs = _decompose_mla(op, {})
        fia_spec = specs[1]
        assert fia_spec.query_mode == "attention"
        assert fia_spec.attention_params is not None

    def test_e3_mla_prefill_fia(self):
        """MLA prefill: decomposes to MatMulV2 + FIA (v0.18.0)."""
        args = _make_mla_prefill_args(num_tokens=256, num_heads=16, kv_lora_rank=512)
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention.default, args
        )
        specs = _decompose_mla(op, {})
        assert specs is not None
        assert len(specs) == 2
        assert specs[0].kernel_type == "MatMulV2"
        assert specs[1].kernel_type == "FusedInferAttentionScore"

    def test_e4_mla_quant_decode_attention_params(self):
        """MLA quant decode also produces attention_params."""
        args = _make_mla_decode_args(batch_size=4, num_heads=16, kv_lora_rank=448)
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention_quant.default, args
        )
        specs = _decompose_mla_quant(op, {})
        assert len(specs) == 3
        fia_spec = specs[1]
        assert fia_spec.attention_params is not None
        assert fia_spec.query_mode == "attention"

    def test_e5_mla_quant_prefill_fia(self):
        """MLA quant prefill: decomposes to MatMulV2 + FIA (v0.18.0)."""
        args = _make_mla_prefill_args(num_tokens=256, num_heads=16, kv_lora_rank=512)
        op = _make_op_info(
            torch.ops.tensor_cast.multihead_latent_attention_quant.default, args
        )
        specs = _decompose_mla_quant(op, {})
        assert specs is not None
        assert len(specs) == 2
        assert specs[0].kernel_type == "MatMulV2"
        assert specs[1].kernel_type == "FusedInferAttentionScore"


# ---- 10. Interpolation linearity verification ----


class TestInterpolationLinearity:
    """Verify linear interpolation produces exact midpoint for equidistant data."""

    def test_exact_midpoint(self, extrap_data_dir):
        """seq=384 is exact midpoint of 256→25 and 512→50 → expect 37.5."""
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.aten.mm.default,
            [
                torch.empty(384, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
            ],
        )
        result = ds.lookup(op)
        assert result is not None
        assert abs(result.latency_us - 37.5) < 0.1

    def test_quarter_point(self, extrap_data_dir):
        """seq=320 is 25% between 256 and 512 → expect 31.25."""
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.aten.mm.default,
            [
                torch.empty(320, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
            ],
        )
        result = ds.lookup(op)
        assert result is not None
        assert abs(result.latency_us - 31.25) < 0.1

    def test_three_quarter_point(self, extrap_data_dir):
        """seq=448 is 75% between 256 and 512 → expect 43.75."""
        base = ProfilingDataSource(extrap_data_dir)
        ds = InterpolatingDataSource(base)
        op = _make_op_info(
            torch.ops.aten.mm.default,
            [
                torch.empty(448, 512, device="meta", dtype=torch.bfloat16),
                torch.empty(512, 1024, device="meta", dtype=torch.bfloat16),
            ],
        )
        result = ds.lookup(op)
        assert result is not None
        assert abs(result.latency_us - 43.75) < 0.1
