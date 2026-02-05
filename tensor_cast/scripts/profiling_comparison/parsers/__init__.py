"""Parsers for profiling comparison.

This module provides parsers for extracting operation data from:
- VLLM profiling output (kernel_details.csv)
- TensorCast simulation results (chrome trace JSON)
- Phase detection (auto-detect prefill/decode)
"""

from tensor_cast.scripts.profiling_comparison.parsers.chrome_trace_parser import (
    normalize_trace_name,
    parse_chrome_trace,
    TraceEvent,
)
from tensor_cast.scripts.profiling_comparison.parsers.kernel_details_parser import (
    extract_single_step,
    find_step_boundaries,
    KernelDetailsParser,
    KernelOp,
    parse_kernel_details,
    SingleStepData,
)
from tensor_cast.scripts.profiling_comparison.parsers.phase_detector import (
    PhaseDetector,
    PhaseInfo,
    StepBoundary,
)

__all__ = [
    # Kernel details parser (single step extraction)
    "KernelDetailsParser",
    "KernelOp",
    "SingleStepData",
    "parse_kernel_details",
    "find_step_boundaries",
    "extract_single_step",
    # Chrome trace parser
    "TraceEvent",
    "parse_chrome_trace",
    "normalize_trace_name",
    # Phase detector
    "PhaseDetector",
    "PhaseInfo",
    "StepBoundary",
]
