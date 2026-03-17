"""Unit tests for M4/M5/M6 evaluation metrics."""

import pytest

from tensor_cast.performance_model.empirical import compute_per_shape_stats


class TestM4PerShapeMatchRate:
    """M4: Per-Shape Match HR -- unique (func_name, shape) pairs, excl zero_cost."""

    def test_mixed_hit_miss(self):
        hit_details = [
            ("aten.mm.default", "MatMulV2", ((2048, 5120), (5120, 5120)), 45.3e-6),
            ("tensor_cast.swiglu.default", "SwiGlu", ((2048, 6912),), 12.1e-6),
        ]
        miss_details = [
            ("aten.mm.default", "shape_mismatch", [(4096, 5120), (5120, 5120)]),
            ("tensor_cast.swiglu.default", "shape_mismatch", [(4096, 6912)]),
        ]
        stats = compute_per_shape_stats(hit_details, miss_details)
        assert stats["hit_shapes"] == 2
        assert stats["total_shapes"] == 4
        assert abs(stats["m4"] - 0.5) < 1e-9

    def test_all_hit(self):
        hit_details = [
            ("aten.mm.default", "MatMulV2", ((2048, 5120), (5120, 5120)), 45.3e-6),
        ]
        stats = compute_per_shape_stats(hit_details, [])
        assert stats["m4"] == 1.0

    def test_all_miss(self):
        miss_details = [
            ("aten.mm.default", "shape_mismatch", [(2048, 5120), (5120, 5120)]),
        ]
        stats = compute_per_shape_stats([], miss_details)
        assert stats["m4"] == 0.0

    def test_zero_cost_excluded(self):
        hit_details = [
            ("aten.mm.default", "MatMulV2", ((2048, 5120), (5120, 5120)), 45.3e-6),
            ("aten.view.default", "zero_cost", ((2048, 5120),), 0.0),
            ("aten.permute.default", "zero_cost", ((2048, 5120),), 0.0),
        ]
        miss_details = [
            ("aten.mm.default", "shape_mismatch", [(4096, 5120), (5120, 5120)]),
        ]
        stats = compute_per_shape_stats(hit_details, miss_details)
        assert stats["hit_shapes"] == 1
        assert stats["total_shapes"] == 2
        assert abs(stats["m4"] - 0.5) < 1e-9

    def test_duplicate_shape_calls_unique(self):
        hit_details = [
            ("aten.mm.default", "MatMulV2", ((2048, 5120), (5120, 5120)), 45.3e-6),
            ("aten.mm.default", "MatMulV2", ((2048, 5120), (5120, 5120)), 45.3e-6),
            ("aten.mm.default", "MatMulV2", ((2048, 5120), (5120, 5120)), 45.3e-6),
        ]
        stats = compute_per_shape_stats(hit_details, [])
        assert stats["hit_shapes"] == 1
        assert stats["total_shapes"] == 1

    def test_empty_inputs(self):
        stats = compute_per_shape_stats([], [])
        assert stats["m4"] == 0.0
        assert stats["hit_shapes"] == 0
        assert stats["total_shapes"] == 0

    def test_miss_shape_list_sorted(self):
        miss_details = [
            ("z_op", "unmapped", [(10, 20)]),
            ("a_op", "unmapped", [(30, 40)]),
        ]
        stats = compute_per_shape_stats([], miss_details)
        assert stats["miss_shape_list"][0][0] == "a_op"
        assert stats["miss_shape_list"][1][0] == "z_op"
