#!/usr/bin/env python3
"""Unified CLI for 3-stage profiling comparison tool.

This module provides a command-line interface with 4 subcommands:

  analyze   - Parse VLLM profiling, detect phase, print TC command
  simulate  - Run TensorCast simulation (subprocess), produce chrome trace
  compare   - Sequence-match VLLM ops vs TC trace, produce Excel report
  run-all   - Chain analyze -> simulate -> compare

Additional utility commands:
  list-profiles  - List available model profiles
  list-mappings  - List available operator mappings

Usage:
    python -m tensor_cast.scripts.profiling_comparison.cli analyze \
        --vllm-dir /path/to/ASCEND_PROFILER_OUTPUT --profile qwen3-32b

    python -m tensor_cast.scripts.profiling_comparison.cli simulate \
        --profile qwen3-32b --phase decode --output-dir /tmp/tc_results/

    python -m tensor_cast.scripts.profiling_comparison.cli compare \
        --vllm-dir /path/to/ASCEND_PROFILER_OUTPUT \
        --tc-trace /tmp/tc_results/chrome_trace.json \
        --profile qwen3-32b --output /tmp/comparison.xlsx

    python -m tensor_cast.scripts.profiling_comparison.cli run-all \
        --vllm-dir /path/to/ASCEND_PROFILER_OUTPUT \
        --profile qwen3-32b --output-dir /tmp/results/
"""

import argparse
import sys
from pathlib import Path

from .config import list_profiles, load_profile, PhaseType
from .output import ExcelFormatter
from .stages.analyze import run_analyze
from .stages.compare import run_compare
from .stages.simulate import run_simulate


def cmd_analyze(args: argparse.Namespace) -> int:
    """Run Stage 1: Analyze VLLM profiling data."""
    phase = PhaseType(args.phase) if args.phase else PhaseType.AUTO
    num_layers = getattr(args, "num_layers", None)

    # Resolve num_layers from profile if not specified
    if num_layers is None and args.profile:
        profile = load_profile(args.profile)
        num_layers = profile.num_layers

    result = run_analyze(
        vllm_dir=args.vllm_dir,
        profile_name=args.profile,
        phase=phase,
        num_layers=num_layers,
        step_index=args.step_index,
    )

    print(f"\n=== Analyze Complete ===")
    print(f"Phase: {result.phase.value}")
    print(f"Steps detected: {result.num_steps}")
    print(f"\nTo run TensorCast simulation:")
    print(f"  {result.command_string}")

    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    """Run Stage 2: TensorCast simulation."""
    profile = load_profile(args.profile)
    phase = PhaseType(args.phase) if args.phase else PhaseType.DECODE

    # Resolve TC config from profile + phase
    if phase == PhaseType.PREFILL:
        defaults = profile.prefill_defaults
    else:
        defaults = profile.decode_defaults

    from .config.schema import TensorCastConfig

    tc_config = TensorCastConfig(
        model_id=profile.tensorcast.model_id,
        device=profile.tensorcast.device,
        world_size=profile.tensorcast.world_size,
        tp_size=profile.tensorcast.tp_size,
        dp_size=profile.tensorcast.dp_size,
        ep=profile.tensorcast.ep,
        quantize_linear_action=profile.tensorcast.quantize_linear_action,
        num_queries=defaults.num_queries,
        query_length=defaults.query_length,
        context_length=defaults.context_length,
        word_embedding_tp=profile.tensorcast.word_embedding_tp,
        lmhead_tp_size=profile.tensorcast.lmhead_tp_size,
        enable_external_shared_experts=profile.tensorcast.enable_external_shared_experts,
    )

    # Override with CLI args if provided
    if args.num_queries is not None:
        tc_config.num_queries = args.num_queries
    if args.query_length is not None:
        tc_config.query_length = args.query_length
    if args.context_length is not None:
        tc_config.context_length = args.context_length

    result = run_simulate(
        tc_config=tc_config,
        output_dir=args.output_dir,
    )

    print(f"\n=== Simulate Complete ===")
    print(f"Chrome trace: {result.chrome_trace_path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Run Stage 3: Sequence-based comparison."""
    num_layers = getattr(args, "num_layers", None)
    mapping_name = args.mapping or "default"
    model_mapping_name = None
    config_dict = {}

    if args.profile:
        profile = load_profile(args.profile)
        if num_layers is None:
            num_layers = profile.num_layers
        if profile.mapping_file:
            model_mapping_name = profile.mapping_file
        config_dict = profile.tensorcast.to_dict()

    result = run_compare(
        vllm_dir=args.vllm_dir,
        tc_trace_path=args.tc_trace,
        mapping_name=mapping_name,
        model_mapping_name=model_mapping_name,
        num_layers=num_layers,
        step_index=args.step_index,
        config_dict=config_dict,
    )

    # Write output
    output_path = args.output
    if output_path is None:
        profile_name = args.profile or "comparison"
        output_path = Path(f"{profile_name}_comparison.xlsx")

    formatter = ExcelFormatter()
    formatter.format(result.comparison_result, output_path)

    # Print summary
    summary = result.comparison_result.summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"VLLM total time:     {summary.vllm_total_time_us:.2f} us")
    print(f"TensorCast total:    {summary.tc_total_time_us:.2f} us")
    print(
        f"Difference:          {summary.overall_diff_us:+.2f} us "
        f"({summary.overall_diff_pct:+.1f}%)"
    )
    print(f"Matched operations:  {result.num_matched}")
    print(f"Unmatched VLLM:      {result.num_unmatched_vllm}")
    print(f"Unmatched TC:        {result.num_unmatched_tc}")
    print(f"Mismatches:          {result.num_mismatch}")
    print(f"Coverage:            {summary.coverage_pct:.1f}%")
    print(f"Output saved to:     {output_path}")

    return 0


def cmd_run_all(args: argparse.Namespace) -> int:
    """Run all 3 stages sequentially."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase = PhaseType(args.phase) if args.phase else PhaseType.AUTO
    num_layers = getattr(args, "num_layers", None)

    # Stage 1: Analyze
    print("=" * 60)
    print("STAGE 1: Analyze VLLM Profiling")
    print("=" * 60)

    if num_layers is None and args.profile:
        profile = load_profile(args.profile)
        num_layers = profile.num_layers

    analyze_result = run_analyze(
        vllm_dir=args.vllm_dir,
        profile_name=args.profile,
        phase=phase,
        num_layers=num_layers,
        step_index=args.step_index,
    )

    # Stage 2: Simulate
    print(f"\n{'=' * 60}")
    print("STAGE 2: Run TensorCast Simulation")
    print("=" * 60)

    simulate_result = run_simulate(
        tc_config=analyze_result.tc_config,
        output_dir=output_dir,
    )

    # Stage 3: Compare
    print(f"\n{'=' * 60}")
    print("STAGE 3: Compare by Execution Order")
    print("=" * 60)

    mapping_name = args.mapping or "default"
    model_mapping_name = None
    profile = load_profile(args.profile)
    if profile.mapping_file:
        model_mapping_name = profile.mapping_file

    compare_result = run_compare(
        vllm_dir=args.vllm_dir,
        tc_trace_path=simulate_result.chrome_trace_path,
        mapping_name=mapping_name,
        model_mapping_name=model_mapping_name,
        num_layers=num_layers,
        step_index=args.step_index,
        config_dict=analyze_result.tc_config.to_dict(),
    )

    # Write Excel report
    output_path = output_dir / f"{args.profile}_comparison.xlsx"
    formatter = ExcelFormatter()
    formatter.format(compare_result.comparison_result, output_path)

    # Print final summary
    summary = compare_result.comparison_result.summary
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"VLLM total time:     {summary.vllm_total_time_us:.2f} us")
    print(f"TensorCast total:    {summary.tc_total_time_us:.2f} us")
    print(
        f"Difference:          {summary.overall_diff_us:+.2f} us "
        f"({summary.overall_diff_pct:+.1f}%)"
    )
    print(f"Matched:             {compare_result.num_matched}")
    print(f"Coverage:            {summary.coverage_pct:.1f}%")
    print(f"\nOutputs:")
    print(f"  Chrome trace:  {simulate_result.chrome_trace_path}")
    print(f"  Excel report:  {output_path}")

    return 0


def cmd_list_profiles(args: argparse.Namespace) -> int:
    """List available profiles."""
    profiles = list_profiles()

    if not profiles:
        print("No profiles found.")
        return 0

    print("Available profiles:")
    print("-" * 40)

    for name in profiles:
        try:
            profile = load_profile(name)
            desc = profile.description or "No description"
            print(f"  {name}")
            print(f"    Model: {profile.tensorcast.model_id}")
            print(f"    {desc}")
            print()
        except Exception as e:
            print(f"  {name} (error loading: {e})")

    return 0


def cmd_list_mappings(args: argparse.Namespace) -> int:
    """List available mappings."""
    from .alignment.sequence_matcher import MAPPINGS_DIR, load_decomposition_config

    print("Available decomposition mappings:")
    print("-" * 40)

    if MAPPINGS_DIR.exists():
        for path in sorted(MAPPINGS_DIR.glob("*.yaml")):
            name = path.stem
            try:
                config = load_decomposition_config(name)
                num_decomp = len(config.decompositions)
                num_ignored_vllm = len(config.ignored_vllm_ops)
                num_ignored_tc = len(config.ignored_tc_ops)
                print(f"  {name}")
                print(f"    Decompositions: {num_decomp}, "
                      f"Ignored VLLM ops: {num_ignored_vllm}, "
                      f"Ignored TC ops: {num_ignored_tc}")
            except Exception as e:
                print(f"  {name} (error: {e})")
    else:
        print("  (none found)")

    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="3-Stage Profiling Comparison: VLLM vs TensorCast",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # === analyze ===
    analyze_p = subparsers.add_parser(
        "analyze",
        help="Stage 1: Analyze VLLM profiling data, detect phase, print TC command",
    )
    analyze_p.add_argument(
        "--vllm-dir", type=Path, required=True,
        help="Path to ASCEND_PROFILER_OUTPUT directory",
    )
    analyze_p.add_argument(
        "--profile", type=str, required=True,
        help="Model profile name (e.g., qwen3-32b)",
    )
    analyze_p.add_argument(
        "--phase", choices=["auto", "prefill", "decode"], default="auto",
        help="Phase type (default: auto-detect)",
    )
    analyze_p.add_argument(
        "--num-layers", type=int,
        help="Number of decoder layers (auto-detected from profile if not specified)",
    )
    analyze_p.add_argument(
        "--step-index", type=int, default=0,
        help="Forward pass index to extract (default: 0)",
    )

    # === simulate ===
    simulate_p = subparsers.add_parser(
        "simulate",
        help="Stage 2: Run TensorCast simulation, produce chrome trace",
    )
    simulate_p.add_argument(
        "--profile", type=str, required=True,
        help="Model profile name",
    )
    simulate_p.add_argument(
        "--phase", choices=["prefill", "decode"], default="decode",
        help="Phase to simulate (default: decode)",
    )
    simulate_p.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for simulation output",
    )
    simulate_p.add_argument("--num-queries", type=int, help="Override batch size")
    simulate_p.add_argument("--query-length", type=int, help="Override query length")
    simulate_p.add_argument("--context-length", type=int, help="Override context length")

    # === compare ===
    compare_p = subparsers.add_parser(
        "compare",
        help="Stage 3: Sequence-match VLLM ops vs TC trace, produce Excel report",
    )
    compare_p.add_argument(
        "--vllm-dir", type=Path, required=True,
        help="Path to ASCEND_PROFILER_OUTPUT directory",
    )
    compare_p.add_argument(
        "--tc-trace", type=Path, required=True,
        help="Path to TensorCast chrome trace JSON",
    )
    compare_p.add_argument(
        "--profile", type=str,
        help="Model profile name (for num_layers and mapping)",
    )
    compare_p.add_argument(
        "--mapping", type=str,
        help="Mapping file to use (default: default)",
    )
    compare_p.add_argument(
        "--num-layers", type=int,
        help="Number of decoder layers",
    )
    compare_p.add_argument(
        "--step-index", type=int, default=0,
        help="Forward pass index to extract (default: 0)",
    )
    compare_p.add_argument(
        "--output", type=Path,
        help="Output Excel file path",
    )

    # === run-all ===
    runall_p = subparsers.add_parser(
        "run-all",
        help="Run all 3 stages: analyze -> simulate -> compare",
    )
    runall_p.add_argument(
        "--vllm-dir", type=Path, required=True,
        help="Path to ASCEND_PROFILER_OUTPUT directory",
    )
    runall_p.add_argument(
        "--profile", type=str, required=True,
        help="Model profile name",
    )
    runall_p.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for all outputs",
    )
    runall_p.add_argument(
        "--phase", choices=["auto", "prefill", "decode"], default="auto",
        help="Phase type (default: auto-detect)",
    )
    runall_p.add_argument(
        "--mapping", type=str,
        help="Mapping file to use (default: default)",
    )
    runall_p.add_argument(
        "--num-layers", type=int,
        help="Number of decoder layers",
    )
    runall_p.add_argument(
        "--step-index", type=int, default=0,
        help="Forward pass index to extract (default: 0)",
    )

    # === list-profiles ===
    subparsers.add_parser("list-profiles", help="List available model profiles")

    # === list-mappings ===
    subparsers.add_parser("list-mappings", help="List available operator mappings")

    return parser


def main() -> int:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    commands = {
        "analyze": cmd_analyze,
        "simulate": cmd_simulate,
        "compare": cmd_compare,
        "run-all": cmd_run_all,
        "list-profiles": cmd_list_profiles,
        "list-mappings": cmd_list_mappings,
    }

    handler = commands.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}")
        return 1

    try:
        return handler(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
