"""Tests for sequence matcher module."""

import tempfile
from pathlib import Path

import pytest
import yaml

from tensor_cast.scripts.profiling_comparison.alignment.sequence_matcher import (
    DecompositionConfig,
    load_decomposition_config,
    load_decomposition_config_from_path,
    match_by_sequence,
    merge_decomposition_configs,
    SequenceMatch,
)
from tensor_cast.scripts.profiling_comparison.parsers.chrome_trace_parser import (
    TraceEvent,
)
from tensor_cast.scripts.profiling_comparison.parsers.kernel_details_parser import (
    KernelOp,
)


def _make_vllm_op(op_type, start=0, dur=100):
    return KernelOp(
        name=f"kernel_{op_type}",
        op_type=op_type,
        start_time_us=start,
        duration_us=dur,
        end_time_us=start + dur,
        core_type="",
        input_shapes="",
        output_shapes="",
    )


def _make_tc_event(name, start=0, dur=50):
    return TraceEvent(name=name, duration_us=dur, start_us=start)


class TestDecompositionConfig:
    """Tests for DecompositionConfig."""

    def test_default(self):
        config = DecompositionConfig()
        assert config.version == "2.0"
        assert config.decompositions == {}
        assert config.ignored_vllm_ops == set()
        assert config.ignored_tc_ops == set()

    def test_load_default(self):
        config = load_decomposition_config("default")
        assert config.version == "2.0"
        assert "AddRmsNorm" in config.decompositions
        assert "MatMulV2" in config.decompositions
        assert "TensorMove" in config.ignored_vllm_ops
        assert "aten.view" in config.ignored_tc_ops

    def test_load_from_path(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "version": "2.0",
                    "decompositions": {
                        "TestOp": {"tc_ops": ["aten.test"]},
                    },
                    "ignored_vllm_ops": ["Ignore1"],
                    "ignored_tc_ops": ["aten.ignore"],
                },
                f,
            )
            f.flush()
            config = load_decomposition_config_from_path(Path(f.name))

        assert "TestOp" in config.decompositions
        assert config.decompositions["TestOp"] == ["aten.test"]
        assert "Ignore1" in config.ignored_vllm_ops
        assert "aten.ignore" in config.ignored_tc_ops

    def test_load_list_format(self):
        """Test loading decompositions as plain list (shorthand)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "version": "2.0",
                    "decompositions": {
                        "TestOp": ["aten.a", "aten.b"],
                    },
                },
                f,
            )
            f.flush()
            config = load_decomposition_config_from_path(Path(f.name))

        assert config.decompositions["TestOp"] == ["aten.a", "aten.b"]

    def test_merge_configs(self):
        config1 = DecompositionConfig(
            decompositions={"Op1": ["tc1"]},
            ignored_vllm_ops={"A"},
        )
        config2 = DecompositionConfig(
            decompositions={"Op2": ["tc2"], "Op1": ["tc1_override"]},
            ignored_vllm_ops={"B"},
        )
        merged = merge_decomposition_configs(config1, config2)
        assert merged.decompositions["Op1"] == ["tc1_override"]
        assert merged.decompositions["Op2"] == ["tc2"]
        assert merged.ignored_vllm_ops == {"A", "B"}

    def test_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_decomposition_config("nonexistent_mapping_xyz")


class TestMatchBySequence:
    """Tests for the core matching algorithm."""

    def test_simple_match(self):
        """Test matching a simple sequence."""
        vllm_ops = [_make_vllm_op("MatMulV2", dur=100)]
        tc_events = [_make_tc_event("aten.mm", dur=95)]
        decompositions = {"MatMulV2": ["aten.mm"]}

        matches = match_by_sequence(
            vllm_ops, tc_events, decompositions, set(), set()
        )
        assert len(matches) == 1
        assert matches[0].match_status == "matched"
        assert matches[0].vllm_duration_us == 100
        assert matches[0].tc_duration_us == 95
        assert abs(matches[0].difference_pct - (-5.0)) < 0.01

    def test_multi_op_decomposition(self):
        """Test matching a fused VLLM op to multiple TC ops."""
        vllm_ops = [_make_vllm_op("AddRmsNorm", dur=200)]
        tc_events = [
            _make_tc_event("aten.add", dur=50),
            _make_tc_event("aten.pow", dur=30),
            _make_tc_event("aten.mean", dur=40),
            _make_tc_event("aten.rsqrt", dur=60),
        ]
        decompositions = {
            "AddRmsNorm": ["aten.add", "aten.pow", "aten.mean", "aten.rsqrt"]
        }

        matches = match_by_sequence(
            vllm_ops, tc_events, decompositions, set(), set()
        )
        assert len(matches) == 1
        assert matches[0].match_status == "matched"
        assert matches[0].tc_duration_us == 180  # 50+30+40+60
        assert len(matches[0].tc_events) == 4

    def test_ignored_vllm_ops(self):
        """Test that ignored VLLM ops are skipped."""
        vllm_ops = [
            _make_vllm_op("TensorMove", dur=10),
            _make_vllm_op("MatMulV2", dur=100),
        ]
        tc_events = [_make_tc_event("aten.mm", dur=95)]
        decompositions = {"MatMulV2": ["aten.mm"]}

        matches = match_by_sequence(
            vllm_ops, tc_events, decompositions, {"TensorMove"}, set()
        )
        assert len(matches) == 1
        assert matches[0].vllm_op.op_type == "MatMulV2"

    def test_ignored_tc_ops(self):
        """Test that ignored TC ops are skipped during matching."""
        vllm_ops = [_make_vllm_op("MatMulV2", dur=100)]
        tc_events = [
            _make_tc_event("aten.view", dur=0),
            _make_tc_event("aten.mm", dur=95),
        ]
        decompositions = {"MatMulV2": ["aten.mm"]}

        matches = match_by_sequence(
            vllm_ops, tc_events, decompositions, set(), {"aten.view"}
        )
        assert len(matches) == 1
        assert matches[0].match_status == "matched"

    def test_unmatched_vllm(self):
        """Test VLLM op with no decomposition mapping."""
        vllm_ops = [_make_vllm_op("UnknownOp", dur=50)]
        tc_events = []
        decompositions = {}

        matches = match_by_sequence(
            vllm_ops, tc_events, decompositions, set(), set()
        )
        assert len(matches) == 1
        assert matches[0].match_status == "unmatched_vllm"

    def test_unmatched_tc(self):
        """Test remaining TC ops after all VLLM ops consumed."""
        vllm_ops = []
        tc_events = [_make_tc_event("aten.extra", dur=10)]
        decompositions = {}

        matches = match_by_sequence(
            vllm_ops, tc_events, decompositions, set(), set()
        )
        assert len(matches) == 1
        assert matches[0].match_status == "unmatched_tc"

    def test_mismatch(self):
        """Test when TC op doesn't match expected."""
        vllm_ops = [_make_vllm_op("AddRmsNorm", dur=100)]
        tc_events = [
            _make_tc_event("aten.add", dur=50),
            _make_tc_event("aten.wrong_op", dur=30),
        ]
        decompositions = {
            "AddRmsNorm": ["aten.add", "aten.pow", "aten.mean", "aten.rsqrt"]
        }

        matches = match_by_sequence(
            vllm_ops, tc_events, decompositions, set(), set()
        )
        assert matches[0].match_status == "mismatch"
        # Only consumed the first matching TC op
        assert len(matches[0].tc_events) == 1

    def test_no_double_counting(self):
        """Test that each TC op is consumed exactly once."""
        vllm_ops = [
            _make_vllm_op("MatMulV2", dur=100),
            _make_vllm_op("MatMulV2", dur=120),
        ]
        tc_events = [
            _make_tc_event("aten.mm", dur=95),
            _make_tc_event("aten.mm", dur=110),
        ]
        decompositions = {"MatMulV2": ["aten.mm"]}

        matches = match_by_sequence(
            vllm_ops, tc_events, decompositions, set(), set()
        )
        assert len(matches) == 2
        # First match gets first TC op
        assert matches[0].tc_events[0].duration_us == 95
        # Second match gets second TC op
        assert matches[1].tc_events[0].duration_us == 110

    def test_mixed_sequence(self):
        """Test a realistic mixed sequence."""
        vllm_ops = [
            _make_vllm_op("AddRmsNorm", dur=200),
            _make_vllm_op("MatMulV2", dur=500),
            _make_vllm_op("SwiGlu", dur=100),
        ]
        tc_events = [
            _make_tc_event("aten.add", dur=50),
            _make_tc_event("aten.pow", dur=30),
            _make_tc_event("aten.mean", dur=40),
            _make_tc_event("aten.rsqrt", dur=60),
            _make_tc_event("aten.mm", dur=480),
            _make_tc_event("aten.silu", dur=40),
            _make_tc_event("aten.mul", dur=50),
        ]
        decompositions = {
            "AddRmsNorm": ["aten.add", "aten.pow", "aten.mean", "aten.rsqrt"],
            "MatMulV2": ["aten.mm"],
            "SwiGlu": ["aten.silu", "aten.mul"],
        }

        matches = match_by_sequence(
            vllm_ops, tc_events, decompositions, set(), set()
        )
        assert len(matches) == 3
        assert all(m.match_status == "matched" for m in matches)
        assert matches[0].tc_duration_us == 180  # add+pow+mean+rsqrt
        assert matches[1].tc_duration_us == 480  # mm
        assert matches[2].tc_duration_us == 90   # silu+mul

    def test_position_tracking(self):
        """Test that position indices are assigned correctly."""
        vllm_ops = [
            _make_vllm_op("MatMulV2", dur=100),
            _make_vllm_op("MatMulV2", dur=120),
        ]
        tc_events = [
            _make_tc_event("aten.mm", dur=95),
            _make_tc_event("aten.mm", dur=110),
        ]
        decompositions = {"MatMulV2": ["aten.mm"]}

        matches = match_by_sequence(
            vllm_ops, tc_events, decompositions, set(), set()
        )
        assert matches[0].position == 0
        assert matches[1].position == 1

    def test_ran_out_of_tc_events(self):
        """Test when TC events run out before all VLLM ops are matched."""
        vllm_ops = [
            _make_vllm_op("MatMulV2", dur=100),
            _make_vllm_op("MatMulV2", dur=120),
        ]
        tc_events = [
            _make_tc_event("aten.mm", dur=95),
        ]
        decompositions = {"MatMulV2": ["aten.mm"]}

        matches = match_by_sequence(
            vllm_ops, tc_events, decompositions, set(), set()
        )
        assert matches[0].match_status == "matched"
        assert matches[1].match_status == "mismatch"
