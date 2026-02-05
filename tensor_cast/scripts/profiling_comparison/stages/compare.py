"""Stage 3: Compare VLLM profiling with TensorCast chrome trace.

This stage parses a single forward pass from VLLM profiling data and
a TensorCast chrome trace, then runs sequence-based matching to produce
a comparison result with per-operation differences.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..alignment.sequence_matcher import (
    DecompositionConfig,
    SequenceMatch,
    load_decomposition_config,
    match_by_sequence,
    merge_decomposition_configs,
)
from ..output.base import ComparisonResult, ComparisonSummary, OperationMatch
from ..parsers.chrome_trace_parser import TraceEvent, parse_chrome_trace
from ..parsers.kernel_details_parser import KernelDetailsParser


@dataclass
class CompareResult:
    """Result of the compare stage.

    Attributes:
        sequence_matches: Ordered list of sequence match records
        comparison_result: ComparisonResult for output formatting
        num_matched: Number of successfully matched operations
        num_unmatched_vllm: VLLM ops with no TC match
        num_unmatched_tc: TC ops with no VLLM match
        num_mismatch: Operations where TC op didn't match expected
    """

    sequence_matches: List[SequenceMatch]
    comparison_result: ComparisonResult
    num_matched: int
    num_unmatched_vllm: int
    num_unmatched_tc: int
    num_mismatch: int


def run_compare(
    vllm_dir: Path,
    tc_trace_path: Path,
    mapping_name: str = "default",
    model_mapping_name: Optional[str] = None,
    num_layers: Optional[int] = None,
    step_index: int = 0,
    config_dict: Optional[dict] = None,
) -> CompareResult:
    """Run Stage 3: Sequence-based comparison.

    Args:
        vllm_dir: Path to ASCEND_PROFILER_OUTPUT directory
        tc_trace_path: Path to TensorCast chrome trace JSON
        mapping_name: Base decomposition mapping name (default: "default")
        model_mapping_name: Optional model-specific mapping to merge
        num_layers: Number of decoder layers for step detection
        step_index: Which forward pass to extract from VLLM data
        config_dict: Optional config dict for output metadata

    Returns:
        CompareResult with all comparison data

    Raises:
        FileNotFoundError: If input files not found
    """
    # Load decomposition config
    print(f"\n=== Loading Mappings ===")
    decomp_config = load_decomposition_config(mapping_name)
    print(f"  Base mapping: {mapping_name} "
          f"({len(decomp_config.decompositions)} decompositions)")

    if model_mapping_name:
        try:
            model_config = load_decomposition_config(model_mapping_name)
            decomp_config = merge_decomposition_configs(decomp_config, model_config)
            print(f"  Model mapping: {model_mapping_name} (merged)")
            print(f"  Total decompositions: {len(decomp_config.decompositions)}")
        except FileNotFoundError:
            print(f"  Warning: Model mapping '{model_mapping_name}' not found, "
                  f"using base only")

    # Parse VLLM single forward pass
    print(f"\n=== Parsing VLLM Profiling ===")
    kernel_csv = vllm_dir / "kernel_details.csv"
    if not kernel_csv.exists():
        raise FileNotFoundError(f"kernel_details.csv not found at {kernel_csv}")

    parser = KernelDetailsParser(kernel_csv)
    step_data = parser.extract_single_step(step_index, num_layers)
    vllm_ops = step_data.operations
    print(f"  Step {step_index}: {len(vllm_ops)} operations, "
          f"{step_data.total_duration_us:.2f} us")

    # Parse TC chrome trace
    print(f"\n=== Parsing TensorCast Trace ===")
    tc_events = parse_chrome_trace(tc_trace_path)
    tc_total_us = sum(e.duration_us for e in tc_events)
    print(f"  {len(tc_events)} events, {tc_total_us:.2f} us total")

    # Run sequence matching
    print(f"\n=== Running Sequence Matching ===")
    seq_matches = match_by_sequence(
        vllm_ops=vllm_ops,
        tc_events=tc_events,
        decompositions=decomp_config.decompositions,
        ignored_vllm_ops=decomp_config.ignored_vllm_ops,
        ignored_tc_ops=decomp_config.ignored_tc_ops,
    )

    # Compute statistics
    matched = [m for m in seq_matches if m.match_status == "matched"]
    unmatched_vllm = [m for m in seq_matches if m.match_status == "unmatched_vllm"]
    unmatched_tc = [m for m in seq_matches if m.match_status == "unmatched_tc"]
    mismatched = [m for m in seq_matches if m.match_status == "mismatch"]

    print(f"  Matched: {len(matched)}")
    print(f"  Unmatched VLLM: {len(unmatched_vllm)}")
    print(f"  Unmatched TC: {len(unmatched_tc)}")
    print(f"  Mismatches: {len(mismatched)}")

    # Build ComparisonResult for output formatting
    comparison_result = _build_comparison_result(
        seq_matches=seq_matches,
        vllm_ops=vllm_ops,
        tc_events=tc_events,
        step_data_total_us=step_data.total_duration_us,
        tc_total_us=tc_total_us,
        config_dict=config_dict or {},
        step_index=step_index,
        vllm_dir=vllm_dir,
    )

    return CompareResult(
        sequence_matches=seq_matches,
        comparison_result=comparison_result,
        num_matched=len(matched),
        num_unmatched_vllm=len(unmatched_vllm),
        num_unmatched_tc=len(unmatched_tc),
        num_mismatch=len(mismatched),
    )


def _build_comparison_result(
    seq_matches: List[SequenceMatch],
    vllm_ops: list,
    tc_events: List[TraceEvent],
    step_data_total_us: float,
    tc_total_us: float,
    config_dict: dict,
    step_index: int,
    vllm_dir: Path,
) -> ComparisonResult:
    """Build a ComparisonResult from sequence matches for Excel output."""
    # Convert sequence matches to OperationMatch list
    op_matches = []
    for m in seq_matches:
        vllm_name = m.vllm_op.op_type if m.vllm_op else "[Unmatched TC]"
        tc_names = [e.name for e in m.tc_events]

        op_matches.append(
            OperationMatch(
                vllm_op=vllm_name,
                tc_ops=tc_names,
                vllm_duration_us=m.vllm_duration_us,
                tc_duration_us=m.tc_duration_us,
                difference_us=m.difference_us,
                difference_pct=m.difference_pct,
                match_status=m.match_status,
            )
        )

    # Build VLLM operations list (aggregated by type)
    vllm_agg: dict = {}
    for op in vllm_ops:
        if op.op_type not in vllm_agg:
            vllm_agg[op.op_type] = {
                "op_type": op.op_type,
                "count": 0,
                "total_duration_us": 0.0,
                "core_types": set(),
                "input_shapes": set(),
            }
        vllm_agg[op.op_type]["count"] += 1
        vllm_agg[op.op_type]["total_duration_us"] += op.duration_us
        if op.core_type:
            vllm_agg[op.op_type]["core_types"].add(op.core_type)
        if op.input_shapes:
            vllm_agg[op.op_type]["input_shapes"].add(op.input_shapes)

    vllm_operations = []
    for data in vllm_agg.values():
        data["avg_duration_us"] = (
            data["total_duration_us"] / data["count"] if data["count"] > 0 else 0
        )
        data["core_types"] = list(data["core_types"])
        data["input_shapes"] = list(data["input_shapes"])
        vllm_operations.append(data)

    # Build TC operations list (aggregated by name)
    tc_agg: dict = {}
    for event in tc_events:
        if event.name not in tc_agg:
            tc_agg[event.name] = {
                "op_name": event.name,
                "count": 0,
                "total_time_us": 0.0,
                "bound_classification": "",
                "input_shapes": [],
            }
        tc_agg[event.name]["count"] += 1
        tc_agg[event.name]["total_time_us"] += event.duration_us

    tc_operations = []
    for data in tc_agg.values():
        data["avg_time_us"] = (
            data["total_time_us"] / data["count"] if data["count"] > 0 else 0
        )
        tc_operations.append(data)

    # Summary statistics
    matched = [m for m in seq_matches if m.match_status == "matched"]
    missing_in_tc = [m for m in seq_matches if m.match_status == "unmatched_vllm"]
    missing_in_vllm = [m for m in seq_matches if m.match_status == "unmatched_tc"]

    matched_vllm_time = sum(m.vllm_duration_us for m in matched)
    vllm_total = step_data_total_us

    coverage_pct = (matched_vllm_time / vllm_total * 100) if vllm_total > 0 else 0

    valid_diffs = [m for m in matched if m.difference_pct != float("inf")]
    avg_abs_diff = (
        sum(abs(m.difference_pct) for m in valid_diffs) / len(valid_diffs)
        if valid_diffs
        else 0
    )

    overall_diff_us = tc_total_us - vllm_total
    overall_diff_pct = (overall_diff_us / vllm_total * 100) if vllm_total > 0 else 0

    summary = ComparisonSummary(
        vllm_total_time_us=vllm_total,
        tc_total_time_us=tc_total_us,
        overall_diff_us=overall_diff_us,
        overall_diff_pct=overall_diff_pct,
        num_matched=len(matched),
        num_missing_in_tc=len(missing_in_tc),
        num_missing_in_vllm=len(missing_in_vllm),
        coverage_pct=coverage_pct,
        avg_abs_diff_pct=avg_abs_diff,
    )

    return ComparisonResult(
        config=config_dict,
        vllm_operations=vllm_operations,
        tc_operations=tc_operations,
        matches=op_matches,
        summary=summary,
        metadata={
            "step_index": step_index,
            "profiling_dir": str(vllm_dir),
            "matching_mode": "sequence",
        },
    )
