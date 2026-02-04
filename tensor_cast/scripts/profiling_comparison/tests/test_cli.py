"""Tests for CLI module."""

import pytest
from pathlib import Path
import tempfile
import csv

from tensor_cast.scripts.profiling_comparison.cli import (
    create_parser,
    cmd_list_profiles,
    cmd_list_mappings,
)


class TestCreateParser:
    """Tests for argument parser creation."""

    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = create_parser()
        assert parser is not None

    def test_parser_help(self):
        """Test that parser has help text."""
        parser = create_parser()
        assert parser.description is not None

    def test_compare_subcommand(self):
        """Test compare subcommand arguments."""
        parser = create_parser()

        # Test with required arguments only
        args = parser.parse_args([
            "compare",
            "--vllm-dir", "/tmp/test",
            "--model-id", "test/model",
        ])

        assert args.command == "compare"
        assert args.vllm_dir == Path("/tmp/test")
        assert args.model_id == "test/model"

    def test_compare_with_profile(self):
        """Test compare with profile argument."""
        parser = create_parser()

        args = parser.parse_args([
            "compare",
            "--profile", "qwen3-32b",
            "--vllm-dir", "/tmp/test",
        ])

        assert args.profile == "qwen3-32b"
        assert args.vllm_dir == Path("/tmp/test")

    def test_compare_all_options(self):
        """Test compare with all options."""
        parser = create_parser()

        args = parser.parse_args([
            "compare",
            "--vllm-dir", "/tmp/test",
            "--model-id", "Qwen/Qwen3-32B",
            "--device", "ATLAS_800_A3_752T_128G_DIE",
            "--world-size", "16",
            "--tp-size", "16",
            "--dp-size", "1",
            "--num-queries", "136",
            "--context-length", "4096",
            "--phase", "decode",
            "--step-index", "50",
            "--mapping", "qwen3",
            "--output", "result.xlsx",
            "--format", "excel",
        ])

        assert args.model_id == "Qwen/Qwen3-32B"
        assert args.world_size == 16
        assert args.tp_size == 16
        assert args.num_queries == 136
        assert args.context_length == 4096
        assert args.phase == "decode"
        assert args.step_index == 50
        assert args.mapping == "qwen3"

    def test_list_profiles_subcommand(self):
        """Test list-profiles subcommand."""
        parser = create_parser()
        args = parser.parse_args(["list-profiles"])
        assert args.command == "list-profiles"

    def test_list_mappings_subcommand(self):
        """Test list-mappings subcommand."""
        parser = create_parser()
        args = parser.parse_args(["list-mappings"])
        assert args.command == "list-mappings"


class TestListCommands:
    """Tests for list commands."""

    def test_list_profiles_runs(self):
        """Test that list-profiles command runs without error."""
        import argparse
        args = argparse.Namespace()
        result = cmd_list_profiles(args)
        assert result == 0

    def test_list_mappings_runs(self):
        """Test that list-mappings command runs without error."""
        import argparse
        args = argparse.Namespace()
        result = cmd_list_mappings(args)
        assert result == 0


class TestPhaseDetection:
    """Tests for phase detection in CLI."""

    def test_phase_choices(self):
        """Test that phase accepts valid choices."""
        parser = create_parser()

        for phase in ["auto", "prefill", "decode"]:
            args = parser.parse_args([
                "compare",
                "--vllm-dir", "/tmp/test",
                "--model-id", "test",
                "--phase", phase,
            ])
            assert args.phase == phase

    def test_phase_default(self):
        """Test default phase value."""
        parser = create_parser()
        args = parser.parse_args([
            "compare",
            "--vllm-dir", "/tmp/test",
            "--model-id", "test",
        ])
        assert args.phase == "auto"


class TestQuantizationOptions:
    """Tests for quantization options in CLI."""

    def test_quantize_default(self):
        """Test default quantization value."""
        parser = create_parser()
        args = parser.parse_args([
            "compare",
            "--vllm-dir", "/tmp/test",
            "--model-id", "test",
        ])
        assert args.quantize_linear_action == "DISABLED"

    def test_quantize_custom(self):
        """Test custom quantization value."""
        parser = create_parser()
        args = parser.parse_args([
            "compare",
            "--vllm-dir", "/tmp/test",
            "--model-id", "test",
            "--quantize-linear-action", "W8A8_DYNAMIC",
        ])
        assert args.quantize_linear_action == "W8A8_DYNAMIC"


class TestExpertParallelism:
    """Tests for expert parallelism options."""

    def test_ep_default_disabled(self):
        """Test EP is disabled by default."""
        parser = create_parser()
        args = parser.parse_args([
            "compare",
            "--vllm-dir", "/tmp/test",
            "--model-id", "test",
        ])
        assert args.ep is False

    def test_ep_enabled(self):
        """Test enabling EP."""
        parser = create_parser()
        args = parser.parse_args([
            "compare",
            "--vllm-dir", "/tmp/test",
            "--model-id", "test",
            "--ep",
        ])
        assert args.ep is True


class TestOutputOptions:
    """Tests for output options in CLI."""

    def test_output_format_choices(self):
        """Test valid output format choices."""
        parser = create_parser()

        for fmt in ["excel", "json"]:
            args = parser.parse_args([
                "compare",
                "--vllm-dir", "/tmp/test",
                "--model-id", "test",
                "--format", fmt,
            ])
            assert args.format == fmt

    def test_output_format_default(self):
        """Test default output format."""
        parser = create_parser()
        args = parser.parse_args([
            "compare",
            "--vllm-dir", "/tmp/test",
            "--model-id", "test",
        ])
        assert args.format == "excel"

    def test_mode_choices(self):
        """Test valid mode choices."""
        parser = create_parser()

        for mode in ["single-step", "full"]:
            args = parser.parse_args([
                "compare",
                "--vllm-dir", "/tmp/test",
                "--model-id", "test",
                "--mode", mode,
            ])
            assert args.mode == mode
