"""Tests for tools/perf_data_collection/generate_shape_grid.py (TCX CSV mutation version)."""

import ast
import random
import sys
from pathlib import Path


sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "tools" / "perf_data_collection")
)
from generate_shape_grid import (
    build_shape_text,
    generate_elementwise_binary_shapes,
    generate_matmul_shapes,
    generate_rmsnorm_shapes,
    generate_rope_shapes,
    generate_swiglu_shapes,
    mutate_shape,
    parse_shape_text,
)

RNG = random.Random(42)
MIN_VAL, MAX_VAL = 1, 20_000


class TestParseShapeText:
    def test_basic(self):
        result = parse_shape_text("136,7168;7168,3584")
        assert result == [(136, 7168), (7168, 3584)]

    def test_empty(self):
        assert parse_shape_text("") == []
        assert parse_shape_text("N/A") == []

    def test_roundtrip(self):
        original = "136,7168;7168,3584"
        parsed = parse_shape_text(original)
        rebuilt = build_shape_text(parsed)
        assert parse_shape_text(rebuilt) == parsed


class TestMatmulShapes:
    def test_2d_nd_shapes(self):
        template_inputs = [(136, 7168), (7168, 3584)]
        inputs, outputs = generate_matmul_shapes(
            template_inputs, ["ND", "ND"], RNG, MIN_VAL, MAX_VAL
        )
        assert len(inputs) == 2
        assert len(outputs) == 1
        m, k = inputs[0]
        out_m, out_n = outputs[0]
        assert out_m == m

    def test_m_alignment_8(self):
        for _ in range(20):
            inputs, _ = generate_matmul_shapes(
                [(136, 7168), (7168, 3584)], ["ND", "ND"], RNG, MIN_VAL, MAX_VAL
            )
            assert inputs[0][0] % 8 == 0, f"M={inputs[0][0]} not aligned to 8"

    def test_k_alignment_16(self):
        for _ in range(20):
            inputs, _ = generate_matmul_shapes(
                [(136, 7168), (7168, 3584)], ["ND", "ND"], RNG, MIN_VAL, MAX_VAL
            )
            assert inputs[0][1] % 16 == 0, f"K={inputs[0][1]} not aligned to 16"


class TestSwiGluShapes:
    def test_output_half_of_input(self):
        for _ in range(20):
            inputs, outputs = generate_swiglu_shapes(
                [(136, 14336)], RNG, MIN_VAL, MAX_VAL
            )
            assert inputs[0][-1] == outputs[0][-1] * 2


class TestRoPEShapes:
    def test_4d_structure(self):
        inputs, outputs = generate_rope_shapes(
            [
                (1, 136, 40, 128),
                (1, 136, 1, 128),
                (1, 136, 1, 128),
                (1, 136, 1, 128),
            ],
            RNG,
            MIN_VAL,
            MAX_VAL,
        )
        assert len(inputs) == 4
        assert len(inputs[0]) == 4  # (B, S, H, D)
        for side in inputs[1:]:
            assert side[2] == 1  # side inputs have H=1


class TestRmsNormShapes:
    def test_2d_structure(self):
        inputs, outputs = generate_rmsnorm_shapes(
            [(136, 7168), (7168,)], RNG, MIN_VAL, MAX_VAL
        )
        assert len(inputs[0]) == 2  # (seq, hidden)
        assert len(inputs[1]) == 1  # (hidden,)
        assert inputs[0][1] == inputs[1][0]  # hidden matches


class TestElementwiseBinary:
    def test_output_matches_lhs(self):
        for _ in range(20):
            inputs, outputs = generate_elementwise_binary_shapes(
                [(136, 7168), (136, 7168)], RNG, MIN_VAL, MAX_VAL
            )
            assert outputs[0] == inputs[0]


class TestMutateShape:
    def test_preserves_dim_1(self):
        result = mutate_shape((1, 7168), RNG, MIN_VAL, MAX_VAL)
        assert result[0] == 1

    def test_shared_dims(self):
        """When two shapes share the same dim value, shared_dims reuses the mapping."""
        # Pre-populate shared_dims so `or {}` doesn't create a new dict
        shared = {128: 256}
        s1 = mutate_shape((128, 7168), RNG, MIN_VAL, MAX_VAL, shared)
        # dim=128 should reuse the pre-populated value
        assert s1[0] == 256
        # dim=7168 should be added to shared
        assert 7168 in shared
        # Second call reuses both cached values
        s2 = mutate_shape((7168, 128), RNG, MIN_VAL, MAX_VAL, shared)
        assert s2 == (shared[7168], 256)


class TestGenerateShapeGridFromModel:
    """Smoke test for the preserved feat version (model-config-driven)."""

    def test_module_importable(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "tools"
            / "perf_data_collection"
            / "generate_shape_grid_from_model.py"
        )
        assert path.is_file(), f"Preserved file not found: {path}"
        source = path.read_text()
        ast.parse(source)
