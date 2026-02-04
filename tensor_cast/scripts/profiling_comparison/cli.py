#!/usr/bin/env python3
"""Unified CLI for profiling comparison tool.

This module provides a command-line interface for comparing VLLM profiling
data with TensorCast simulation results.

Usage:
    # Compare using a model profile
    python -m tensor_cast.scripts.profiling_comparison.cli compare \
        --profile qwen3-32b \
        --vllm-dir /path/to/profiling \
        --output comparison.xlsx

    # Compare with custom TensorCast config
    python -m tensor_cast.scripts.profiling_comparison.cli compare \
        --vllm-dir /path/to/profiling \
        --model-id Qwen/Qwen3-32B \
        --device ATLAS_800_A3_752T_128G_DIE \
        --world-size 16 --tp-size 16 \
        --num-queries 136 --context-length 4096

    # List available profiles
    python -m tensor_cast.scripts.profiling_comparison.cli list-profiles

    # List available mappings
    python -m tensor_cast.scripts.profiling_comparison.cli list-mappings
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

from .alignment import FusionAwareMapper, list_mappings, load_mappings

from .config import (
    ComparisonConfig,
    list_profiles,
    load_profile,
    OutputConfig,
    PhaseType,
    TensorCastConfig,
    VLLMConfig,
)
from .output import ComparisonResult, ComparisonSummary, ExcelFormatter, OperationMatch
from .parsers import KernelDetailsParser, PhaseDetector, TensorCastAdapter


def run_comparison(
    config: ComparisonConfig, mapper: FusionAwareMapper
) -> ComparisonResult:
    """Run profiling comparison with the given configuration.

    Args:
        config: Comparison configuration
        mapper: Operation mapper to use

    Returns:
        ComparisonResult with all comparison data
    """
    # Parse VLLM profiling data
    print("\n=== Parsing VLLM Profiling Data ===")
    kernel_csv = config.vllm.profiling_dir / "kernel_details.csv"

    if not kernel_csv.exists():
        raise FileNotFoundError(f"kernel_details.csv not found at {kernel_csv}")

    # Auto-detect phase if needed
    if config.vllm.phase == PhaseType.AUTO:
        detector = PhaseDetector(kernel_csv)
        phase_info = detector.detect_phase()
        print(
            f"Detected phase: {phase_info.phase.value} (confidence: {phase_info.confidence:.0%})"
        )
        print(f"Detection method: {phase_info.method}")

        if phase_info.query_length:
            print(f"Detected query length: {phase_info.query_length}")

        # Update config based on detected phase
        if phase_info.phase == PhaseType.PREFILL and phase_info.query_length:
            config.tensorcast.query_length = phase_info.query_length
            config.tensorcast.context_length = 0
        elif phase_info.phase == PhaseType.DECODE:
            config.tensorcast.query_length = 1
            # Keep existing context_length

    # Parse VLLM single step
    parser = KernelDetailsParser(kernel_csv)
    stats = parser.get_step_statistics()
    print(f"Total decode steps detected: {stats['num_steps']}")
    print(f"Average step duration: {stats['avg_step_duration_us']:.2f} us")

    step_data = parser.extract_single_step(config.vllm.step_index)
    print(f"\nExtracted step {config.vllm.step_index}:")
    print(f"  Operations: {len(step_data.operations)}")
    print(f"  Total duration: {step_data.total_duration_us:.2f} us")

    vllm_stats = step_data.get_aggregated_stats()

    # Run TensorCast simulation
    print("\n=== Running TensorCast Simulation ===")
    tc_adapter = TensorCastAdapter(config.tensorcast)
    tc_result = tc_adapter.parse()

    # Build TC stats lookup
    tc_stats: Dict[str, dict] = {}
    for op in tc_result.operations:
        base_name = op.op_type.replace(".default", "")
        tc_stats[base_name] = {
            "total_duration_us": op.total_time_us,
            "count": op.count,
            "full_name": op.op_type,
            "input_shapes": op.input_shapes,
            "output_shapes": op.output_shapes,
            "bound_classification": op.bound_classification,
        }

    # Match operations
    print("\n=== Matching Operations ===")
    matches_data = mapper.match_single_step_ops(vllm_stats, tc_stats)

    # Convert to OperationMatch objects
    matches = []
    for m in matches_data:
        matches.append(
            OperationMatch(
                vllm_op=m["vllm_op"],
                tc_ops=m["tc_ops"],
                vllm_duration_us=m["vllm_duration_us"],
                tc_duration_us=m["tc_duration_us"],
                vllm_count=m.get("vllm_count", 0),
                tc_count=m.get("tc_count", 0),
                difference_us=m["difference_us"],
                difference_pct=m["difference_pct"],
                match_status=m["match_status"],
            )
        )

    # Calculate summary statistics
    vllm_total = sum(d["total_duration_us"] for d in vllm_stats.values())
    tc_total = tc_result.total_time_us

    matched = [
        m for m in matches if m.match_status not in ("missing_in_tc", "missing_in_vllm")
    ]
    missing_in_tc = [m for m in matches if m.match_status == "missing_in_tc"]
    missing_in_vllm = [m for m in matches if m.match_status == "missing_in_vllm"]

    matched_vllm_time = sum(m.vllm_duration_us for m in matched)
    coverage_pct = (matched_vllm_time / vllm_total * 100) if vllm_total > 0 else 0

    valid_diffs = [m for m in matched if m.difference_pct != float("inf")]
    avg_abs_diff = (
        sum(abs(m.difference_pct) for m in valid_diffs) / len(valid_diffs)
        if valid_diffs
        else 0
    )

    overall_diff_us = tc_total - vllm_total
    overall_diff_pct = (overall_diff_us / vllm_total * 100) if vllm_total > 0 else 0

    summary = ComparisonSummary(
        vllm_total_time_us=vllm_total,
        tc_total_time_us=tc_total,
        overall_diff_us=overall_diff_us,
        overall_diff_pct=overall_diff_pct,
        num_matched=len(matched),
        num_missing_in_tc=len(missing_in_tc),
        num_missing_in_vllm=len(missing_in_vllm),
        coverage_pct=coverage_pct,
        avg_abs_diff_pct=avg_abs_diff,
    )

    # Build VLLM operations list for output
    vllm_operations = []
    for op_type, data in vllm_stats.items():
        vllm_operations.append(
            {
                "op_type": op_type,
                "count": data["count"],
                "total_duration_us": data["total_duration_us"],
                "avg_duration_us": data["avg_duration_us"],
                "core_types": data.get("core_types", []),
                "input_shapes": data.get("input_shapes", []),
            }
        )

    # Build TC operations list for output
    tc_operations = []
    for op in tc_result.operations:
        tc_operations.append(
            {
                "op_name": op.op_type,
                "count": op.count,
                "total_time_us": op.total_time_us,
                "avg_time_us": op.avg_time_us,
                "bound_classification": op.bound_classification,
                "input_shapes": op.input_shapes,
            }
        )

    return ComparisonResult(
        config=config.tensorcast.to_dict(),
        vllm_operations=vllm_operations,
        tc_operations=tc_operations,
        matches=matches,
        summary=summary,
        metadata={
            "step_index": config.vllm.step_index,
            "profiling_dir": str(config.vllm.profiling_dir),
        },
    )


def cmd_compare(args: argparse.Namespace) -> int:
    """Run comparison command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    # Build configuration
    if args.profile:
        # Load from profile
        profile = load_profile(args.profile)
        print(f"Loaded profile: {profile.name}")
        print(f"  Description: {profile.description}")

        phase = PhaseType(args.phase) if args.phase else PhaseType.AUTO

        config = ComparisonConfig.from_profile(
            profile=profile,
            vllm_dir=args.vllm_dir,
            output_path=args.output,
            phase=phase,
            step_index=args.step_index,
            output_format=args.format,
            mode=args.mode,
        )

        # Override with CLI args if provided
        if args.num_queries:
            config.tensorcast.num_queries = args.num_queries
        if args.context_length:
            config.tensorcast.context_length = args.context_length
        if args.query_length:
            config.tensorcast.query_length = args.query_length

    else:
        # Build from CLI args
        if not args.model_id:
            print("Error: Either --profile or --model-id must be specified")
            return 1

        tc_config = TensorCastConfig(
            model_id=args.model_id,
            device=args.device,
            world_size=args.world_size,
            tp_size=args.tp_size,
            dp_size=args.dp_size,
            ep=args.ep,
            quantize_linear_action=args.quantize_linear_action,
            num_queries=args.num_queries or 1,
            query_length=args.query_length or 1,
            context_length=args.context_length or 0,
            word_embedding_tp=args.word_embedding_tp,
            lmhead_tp_size=args.lmhead_tp_size,
            enable_external_shared_experts=args.enable_external_shared_experts,
        )

        phase = PhaseType(args.phase) if args.phase else PhaseType.AUTO

        vllm_config = VLLMConfig(
            profiling_dir=args.vllm_dir,
            step_index=args.step_index,
            phase=phase,
        )

        output_path = args.output
        if output_path is None:
            output_path = Path(f"comparison.{args.format}")

        output_config = OutputConfig(
            output_path=output_path,
            format=args.format,
        )

        config = ComparisonConfig(
            tensorcast=tc_config,
            vllm=vllm_config,
            output=output_config,
            mode=args.mode,
            mapping_file=args.mapping,
        )

    # Load mapper
    mapping_name = config.mapping_file or args.mapping or "default"
    try:
        mapper = FusionAwareMapper.from_yaml(mapping_name)
        print(f"Using mapping: {mapping_name}")
    except FileNotFoundError:
        print(f"Mapping '{mapping_name}' not found, using default hardcoded mappings")
        mapper = FusionAwareMapper()

    # Run comparison
    result = run_comparison(config, mapper)

    # Output results
    output_path = config.output.output_path
    if config.output.format == "excel":
        formatter = ExcelFormatter()
        formatter.format(result, output_path)
    else:
        print(f"Unsupported output format: {config.output.format}")
        return 1

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"VLLM total time:     {result.summary.vllm_total_time_us:.2f} us")
    print(f"TensorCast total:    {result.summary.tc_total_time_us:.2f} us")
    print(
        f"Difference:          {result.summary.overall_diff_us:+.2f} us ({result.summary.overall_diff_pct:+.1f}%)"
    )
    print(f"Matched operations:  {result.summary.num_matched}")
    print(f"Coverage:            {result.summary.coverage_pct:.1f}%")
    print(f"Output saved to:     {output_path}")

    return 0


def cmd_list_profiles(args: argparse.Namespace) -> int:
    """List available profiles command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    profiles = list_profiles()

    if not profiles:
        print("No profiles found.")
        print(
            "Create profiles in: tensor_cast/scripts/profiling_comparison/config/profiles/"
        )
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
    """List available mappings command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    mappings = list_mappings()

    if not mappings:
        print("No mappings found.")
        print(
            "Create mappings in: tensor_cast/scripts/profiling_comparison/alignment/mappings/"
        )
        return 0

    print("Available mappings:")
    print("-" * 40)

    for name in mappings:
        try:
            config = load_mappings(name)
            num_vllm_fusions = len(config.vllm_fusions)
            num_tc_fusions = len(config.tc_fusions)
            num_direct = len(config.direct_mappings)
            print(f"  {name}")
            print(f"    Version: {config.version}")
            print(
                f"    VLLM fusions: {num_vllm_fusions}, TC fusions: {num_tc_fusions}, Direct: {num_direct}"
            )
            print()
        except Exception as e:
            print(f"  {name} (error loading: {e})")

    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Profiling Comparison Tool: Compare VLLM profiling with TensorCast simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Run profiling comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Profile-based configuration
    compare_parser.add_argument(
        "--profile",
        type=str,
        help="Model profile to use (e.g., qwen3-32b, deepseek-v3)",
    )

    # VLLM profiling input
    compare_parser.add_argument(
        "--vllm-dir",
        type=Path,
        required=True,
        help="Path to ASCEND_PROFILER_OUTPUT directory",
    )

    # Phase configuration
    compare_parser.add_argument(
        "--phase",
        choices=["auto", "prefill", "decode"],
        default="auto",
        help="Phase type (default: auto-detect)",
    )
    compare_parser.add_argument(
        "--step-index",
        type=int,
        default=100,
        help="Decode step index to extract (default: 100)",
    )

    # TensorCast configuration (used if no profile)
    compare_parser.add_argument(
        "--model-id",
        type=str,
        help="HuggingFace model ID",
    )
    compare_parser.add_argument(
        "--device",
        type=str,
        default="ATLAS_800_A3_752T_128G_DIE",
        help="Device profile name",
    )
    compare_parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Total number of devices",
    )
    compare_parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallelism size",
    )
    compare_parser.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help="Data parallelism size",
    )
    compare_parser.add_argument(
        "--ep",
        action="store_true",
        help="Enable expert parallelism",
    )
    compare_parser.add_argument(
        "--quantize-linear-action",
        type=str,
        default="DISABLED",
        help="Quantization scheme",
    )
    compare_parser.add_argument(
        "--num-queries",
        type=int,
        help="Number of queries / batch size",
    )
    compare_parser.add_argument(
        "--query-length",
        type=int,
        help="Query length",
    )
    compare_parser.add_argument(
        "--context-length",
        type=int,
        help="Context length for decode",
    )
    compare_parser.add_argument(
        "--word-embedding-tp",
        action="store_true",
        help="Enable word embedding TP",
    )
    compare_parser.add_argument(
        "--lmhead-tp-size",
        type=int,
        default=1,
        help="LM head TP size",
    )
    compare_parser.add_argument(
        "--enable-external-shared-experts",
        action="store_true",
        help="Enable external shared experts",
    )

    # Mapping configuration
    compare_parser.add_argument(
        "--mapping",
        type=str,
        help="Mapping file to use (default: from profile or 'default')",
    )

    # Output configuration
    compare_parser.add_argument(
        "--output",
        type=Path,
        help="Output file path",
    )
    compare_parser.add_argument(
        "--format",
        choices=["excel", "json"],
        default="excel",
        help="Output format (default: excel)",
    )
    compare_parser.add_argument(
        "--mode",
        choices=["single-step", "full"],
        default="single-step",
        help="Comparison mode (default: single-step)",
    )

    # List profiles command
    subparsers.add_parser(
        "list-profiles",
        help="List available model profiles",
    )

    # List mappings command
    subparsers.add_parser(
        "list-mappings",
        help="List available operator mappings",
    )

    return parser


def main() -> int:
    """Main entry point for CLI.

    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "compare":
        return cmd_compare(args)
    elif args.command == "list-profiles":
        return cmd_list_profiles(args)
    elif args.command == "list-mappings":
        return cmd_list_mappings(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
