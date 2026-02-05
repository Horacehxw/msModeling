"""Profiling Comparison Tool.

This package provides a 3-stage pipeline for comparing VLLM profiling results
with TensorCast simulation to validate performance predictions.

Stages:
    1. analyze  - Parse VLLM profiling, detect phase, print TC command
    2. simulate - Run TensorCast simulation (subprocess), produce chrome trace
    3. compare  - Sequence-match VLLM ops vs TC trace, produce Excel report

Main entry point:
    python -m tensor_cast.scripts.profiling_comparison.cli analyze --help
    python -m tensor_cast.scripts.profiling_comparison.cli simulate --help
    python -m tensor_cast.scripts.profiling_comparison.cli compare --help
    python -m tensor_cast.scripts.profiling_comparison.cli run-all --help
"""

from tensor_cast.scripts.profiling_comparison.alignment import (
    DecompositionConfig,
    load_decomposition_config,
    match_by_sequence,
    merge_decomposition_configs,
    SequenceMatch,
)
from tensor_cast.scripts.profiling_comparison.config import (
    list_profiles,
    load_profile,
    ModelProfile,
    PhaseType,
    ProfileDefaults,
    TensorCastConfig,
)
from tensor_cast.scripts.profiling_comparison.output import (
    ComparisonResult,
    ComparisonSummary,
    ExcelFormatter,
    OperationMatch,
)
from tensor_cast.scripts.profiling_comparison.parsers import (
    KernelDetailsParser,
    KernelOp,
    normalize_trace_name,
    parse_chrome_trace,
    PhaseDetector,
    PhaseInfo,
    SingleStepData,
    StepBoundary,
    TraceEvent,
)

__all__ = [
    # Config
    "ModelProfile",
    "PhaseType",
    "ProfileDefaults",
    "TensorCastConfig",
    "list_profiles",
    "load_profile",
    # Alignment - sequence matcher
    "DecompositionConfig",
    "SequenceMatch",
    "load_decomposition_config",
    "match_by_sequence",
    "merge_decomposition_configs",
    # Parsers
    "KernelDetailsParser",
    "KernelOp",
    "PhaseDetector",
    "PhaseInfo",
    "SingleStepData",
    "StepBoundary",
    "TraceEvent",
    "parse_chrome_trace",
    "normalize_trace_name",
    # Output
    "ComparisonResult",
    "ComparisonSummary",
    "ExcelFormatter",
    "OperationMatch",
]
