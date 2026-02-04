"""Parsers for profiling comparison.

This module provides parsers for extracting operation data from:
- VLLM profiling output (kernel_details.csv)
- TensorCast simulation results

Key components:
- KernelDetailsParser: Parses VLLM kernel_details.csv files
- TensorCastAdapter: Runs TensorCast simulation and extracts profiling data
- PhaseDetector: Auto-detects prefill/decode phases from profiling data
"""

from tensor_cast.scripts.profiling_comparison.parsers.kernel_details_parser import (
    KernelDetailsParser,
    KernelOp,
    SingleStepData,
    extract_single_step,
    find_step_boundaries,
    parse_kernel_details,
)
from tensor_cast.scripts.profiling_comparison.parsers.base import (
    BaseParser,
    OperationData,
    Parser,
    ProfilingResult,
)
from tensor_cast.scripts.profiling_comparison.parsers.tensorcast_adapter import (
    TensorCastAdapter,
    TensorCastSimulationResult,
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
    # Base parser protocol
    "BaseParser",
    "OperationData",
    "Parser",
    "ProfilingResult",
    # TensorCast adapter
    "TensorCastAdapter",
    "TensorCastSimulationResult",
    # Phase detector
    "PhaseDetector",
    "PhaseInfo",
    "StepBoundary",
]
