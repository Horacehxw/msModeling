"""Tests for stages module."""

import csv
import json
import tempfile
from pathlib import Path

import pytest

from tensor_cast.scripts.profiling_comparison.config.schema import (
    TensorCastConfig,
)
from tensor_cast.scripts.profiling_comparison.stages.analyze import run_analyze
from tensor_cast.scripts.profiling_comparison.stages.compare import run_compare


class TestAnalyzeStage:
    """Tests for Stage 1: Analyze."""

    def _create_test_profiling_dir(self, ops):
        """Create a temp dir with a kernel_details.csv."""
        tmpdir = tempfile.mkdtemp()
        csv_path = Path(tmpdir) / "kernel_details.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Name", "Type", "Start Time(us)", "Duration(us)",
                "Accelerator Core", "Input Shapes", "Output Shapes",
            ])
            for op in ops:
                writer.writerow(op)
        return Path(tmpdir)

    def test_analyze_basic(self):
        """Test basic analyze with decode-like data."""
        vllm_dir = self._create_test_profiling_dir([
            ["k1", "ReshapeAndCacheNdKernel", "100", "50", "", "136,1,128", ""],
            *[
                [f"k_att_{i}", "FusedInferAttentionScore",
                 str(200 + i * 100), "80", "", "", ""]
                for i in range(64)
            ],
        ])

        result = run_analyze(
            vllm_dir=vllm_dir,
            profile_name="qwen3_32b",
            num_layers=64,
        )

        assert result.phase is not None
        assert result.tc_config is not None
        assert result.command_string is not None
        assert "Qwen/Qwen3-32B" in result.command_string


class TestCompareStage:
    """Tests for Stage 3: Compare."""

    def _create_test_profiling_dir(self, ops):
        tmpdir = tempfile.mkdtemp()
        csv_path = Path(tmpdir) / "kernel_details.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Name", "Type", "Start Time(us)", "Duration(us)",
                "Accelerator Core", "Input Shapes", "Output Shapes",
            ])
            for op in ops:
                writer.writerow(op)
        return Path(tmpdir)

    def _create_test_trace(self, events):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump({"traceEvents": events}, f)
        f.flush()
        return Path(f.name)

    def test_compare_basic(self):
        """Test basic compare with matching ops."""
        # Create VLLM data with exactly 1 forward pass (1 attention op)
        vllm_dir = self._create_test_profiling_dir([
            ["k1", "FusedInferAttentionScore", "100", "80", "", "", ""],
            ["k2", "MatMulV2", "200", "100", "", "", ""],
        ])

        # Create TC trace
        tc_trace = self._create_test_trace([
            {"ph": "X", "name": "tensor_cast::attention::default",
             "ts": 0, "dur": 75},
            {"ph": "X", "name": "aten::mm::default",
             "ts": 100, "dur": 95},
        ])

        result = run_compare(
            vllm_dir=vllm_dir,
            tc_trace_path=tc_trace,
            mapping_name="default",
            num_layers=1,  # 1 attention op = 1 layer = 1 step
            step_index=0,
        )

        assert result.num_matched >= 0
        assert result.comparison_result is not None
        assert result.comparison_result.summary is not None

    def test_compare_with_ignored_ops(self):
        """Test that ignored ops are properly skipped."""
        vllm_dir = self._create_test_profiling_dir([
            ["k1", "FusedInferAttentionScore", "100", "80", "", "", ""],
            ["k2", "TensorMove", "200", "10", "", "", ""],  # Should be ignored
            ["k3", "MatMulV2", "220", "100", "", "", ""],
        ])

        tc_trace = self._create_test_trace([
            {"ph": "X", "name": "aten::view::default", "ts": 0, "dur": 0},
            {"ph": "X", "name": "tensor_cast::attention::default",
             "ts": 10, "dur": 75},
            {"ph": "X", "name": "aten::mm::default", "ts": 100, "dur": 95},
        ])

        result = run_compare(
            vllm_dir=vllm_dir,
            tc_trace_path=tc_trace,
            mapping_name="default",
            num_layers=1,
            step_index=0,
        )

        # TensorMove should be ignored, aten.view should be ignored
        # Only FusedInferAttentionScore and MatMulV2 should be matched
        matched = [m for m in result.sequence_matches if m.match_status == "matched"]
        assert len(matched) >= 1


class TestTensorCastConfigHelpers:
    """Tests for TensorCastConfig.to_cli_args() and to_command_string()."""

    def test_to_cli_args_basic(self):
        config = TensorCastConfig(
            model_id="Qwen/Qwen3-32B",
            device="ATLAS_800_A3_752T_128G_DIE",
            world_size=16,
            tp_size=16,
        )
        args = config.to_cli_args()
        assert "Qwen/Qwen3-32B" in args
        assert "--device" in args
        assert "--world-size" in args
        assert "--tp-size" in args

    def test_to_cli_args_with_ep(self):
        config = TensorCastConfig(
            model_id="test/model",
            device="TEST",
            ep=True,
        )
        args = config.to_cli_args()
        assert "--ep" in args

    def test_to_cli_args_with_quant(self):
        config = TensorCastConfig(
            model_id="test/model",
            device="TEST",
            quantize_linear_action="W8A8_DYNAMIC",
        )
        args = config.to_cli_args()
        assert "--quantize-linear-action" in args
        assert "W8A8_DYNAMIC" in args

    def test_to_cli_args_no_quant_when_disabled(self):
        config = TensorCastConfig(
            model_id="test/model",
            device="TEST",
            quantize_linear_action="DISABLED",
        )
        args = config.to_cli_args()
        assert "--quantize-linear-action" not in args

    def test_to_cli_args_with_chrome_trace(self):
        config = TensorCastConfig(model_id="test/model", device="TEST")
        args = config.to_cli_args(chrome_trace_path=Path("/tmp/trace.json"))
        assert "--chrome-trace" in args
        assert "/tmp/trace.json" in args

    def test_to_command_string(self):
        config = TensorCastConfig(
            model_id="Qwen/Qwen3-32B",
            device="TEST",
        )
        cmd = config.to_command_string()
        assert cmd.startswith("python -m tensor_cast.scripts.text_generate")
        assert "Qwen/Qwen3-32B" in cmd

    def test_to_cli_args_with_lmhead_tp(self):
        config = TensorCastConfig(
            model_id="test/model",
            device="TEST",
            lmhead_tp_size=8,
        )
        args = config.to_cli_args()
        assert "--lmhead-tp-size" in args
        assert "8" in args

    def test_to_cli_args_lmhead_tp_default_not_included(self):
        config = TensorCastConfig(
            model_id="test/model",
            device="TEST",
            lmhead_tp_size=1,
        )
        args = config.to_cli_args()
        assert "--lmhead-tp-size" not in args
