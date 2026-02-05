"""Tests for CLI module."""

from pathlib import Path

from tensor_cast.scripts.profiling_comparison.cli import (
    cmd_list_mappings,
    cmd_list_profiles,
    create_parser,
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

    def test_analyze_subcommand(self):
        """Test analyze subcommand arguments."""
        parser = create_parser()

        args = parser.parse_args(
            [
                "analyze",
                "--vllm-dir", "/tmp/test",
                "--profile", "qwen3-32b",
            ]
        )
        assert args.command == "analyze"
        assert args.vllm_dir == Path("/tmp/test")
        assert args.profile == "qwen3-32b"
        assert args.phase == "auto"
        assert args.step_index == 0

    def test_simulate_subcommand(self):
        """Test simulate subcommand arguments."""
        parser = create_parser()

        args = parser.parse_args(
            [
                "simulate",
                "--profile", "qwen3-32b",
                "--phase", "decode",
                "--output-dir", "/tmp/tc_results",
            ]
        )
        assert args.command == "simulate"
        assert args.profile == "qwen3-32b"
        assert args.phase == "decode"
        assert args.output_dir == Path("/tmp/tc_results")

    def test_compare_subcommand(self):
        """Test compare subcommand arguments."""
        parser = create_parser()

        args = parser.parse_args(
            [
                "compare",
                "--vllm-dir", "/tmp/test",
                "--tc-trace", "/tmp/trace.json",
                "--profile", "qwen3-32b",
            ]
        )
        assert args.command == "compare"
        assert args.vllm_dir == Path("/tmp/test")
        assert args.tc_trace == Path("/tmp/trace.json")
        assert args.profile == "qwen3-32b"

    def test_compare_with_options(self):
        """Test compare with all options."""
        parser = create_parser()

        args = parser.parse_args(
            [
                "compare",
                "--vllm-dir", "/tmp/test",
                "--tc-trace", "/tmp/trace.json",
                "--profile", "qwen3-32b",
                "--mapping", "qwen3",
                "--num-layers", "64",
                "--step-index", "1",
                "--output", "/tmp/result.xlsx",
            ]
        )
        assert args.mapping == "qwen3"
        assert args.num_layers == 64
        assert args.step_index == 1
        assert args.output == Path("/tmp/result.xlsx")

    def test_run_all_subcommand(self):
        """Test run-all subcommand arguments."""
        parser = create_parser()

        args = parser.parse_args(
            [
                "run-all",
                "--vllm-dir", "/tmp/test",
                "--profile", "qwen3-32b",
                "--output-dir", "/tmp/results",
            ]
        )
        assert args.command == "run-all"
        assert args.vllm_dir == Path("/tmp/test")
        assert args.profile == "qwen3-32b"
        assert args.output_dir == Path("/tmp/results")

    def test_run_all_with_options(self):
        """Test run-all with all options."""
        parser = create_parser()

        args = parser.parse_args(
            [
                "run-all",
                "--vllm-dir", "/tmp/test",
                "--profile", "qwen3-32b",
                "--output-dir", "/tmp/results",
                "--phase", "decode",
                "--mapping", "default",
                "--num-layers", "64",
                "--step-index", "2",
            ]
        )
        assert args.phase == "decode"
        assert args.mapping == "default"
        assert args.num_layers == 64
        assert args.step_index == 2

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


class TestAnalyzeOptions:
    """Tests for analyze subcommand options."""

    def test_analyze_phase_choices(self):
        """Test that analyze accepts valid phase choices."""
        parser = create_parser()

        for phase in ["auto", "prefill", "decode"]:
            args = parser.parse_args(
                [
                    "analyze",
                    "--vllm-dir", "/tmp/test",
                    "--profile", "test",
                    "--phase", phase,
                ]
            )
            assert args.phase == phase

    def test_analyze_default_phase(self):
        """Test default phase value for analyze."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "analyze",
                "--vllm-dir", "/tmp/test",
                "--profile", "test",
            ]
        )
        assert args.phase == "auto"

    def test_analyze_num_layers(self):
        """Test num-layers option for analyze."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "analyze",
                "--vllm-dir", "/tmp/test",
                "--profile", "test",
                "--num-layers", "64",
            ]
        )
        assert args.num_layers == 64


class TestSimulateOptions:
    """Tests for simulate subcommand options."""

    def test_simulate_phase_choices(self):
        """Test that simulate accepts valid phase choices."""
        parser = create_parser()

        for phase in ["prefill", "decode"]:
            args = parser.parse_args(
                [
                    "simulate",
                    "--profile", "test",
                    "--phase", phase,
                    "--output-dir", "/tmp/out",
                ]
            )
            assert args.phase == phase

    def test_simulate_default_phase(self):
        """Test default phase for simulate."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "simulate",
                "--profile", "test",
                "--output-dir", "/tmp/out",
            ]
        )
        assert args.phase == "decode"

    def test_simulate_overrides(self):
        """Test simulate with override options."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "simulate",
                "--profile", "test",
                "--output-dir", "/tmp/out",
                "--num-queries", "64",
                "--query-length", "4096",
                "--context-length", "8192",
            ]
        )
        assert args.num_queries == 64
        assert args.query_length == 4096
        assert args.context_length == 8192
