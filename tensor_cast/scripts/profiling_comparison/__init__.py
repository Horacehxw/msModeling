"""Profiling Comparison Tool.

This package provides utilities for comparing VLLM profiling results with
TensorCast simulation to validate performance predictions.

Main entry point:
    python -m tensor_cast.scripts.profiling_comparison.cli compare --help

Example usage:
    python -m tensor_cast.scripts.profiling_comparison.cli compare \\
        --profile qwen3_32b \\
        --vllm-dir /path/to/ASCEND_PROFILER_OUTPUT \\
        --output comparison.xlsx
"""

from tensor_cast.scripts.profiling_comparison.config import (
    ComparisonConfig,
    ModelProfile,
    OutputConfig,
    PhaseType,
    ProfileDefaults,
    TensorCastConfig,
    VLLMConfig,
    list_profiles,
    load_profile,
)
from tensor_cast.scripts.profiling_comparison.alignment import (
    FusionAwareMapper,
    FusionMapping,
    FUSION_MAPPINGS,
    MappingConfig,
    OpMapper,
    VLLM_TO_TENSORCAST_MAPPING,
    list_mappings,
    load_mappings,
)
from tensor_cast.scripts.profiling_comparison.parsers import (
    BaseParser,
    KernelDetailsParser,
    KernelOp,
    OperationData,
    Parser,
    PhaseDetector,
    PhaseInfo,
    ProfilingResult,
    SingleStepData,
    StepBoundary,
    TensorCastAdapter,
    TensorCastSimulationResult,
)
from tensor_cast.scripts.profiling_comparison.output import (
    ComparisonResult,
    ComparisonSummary,
    ExcelFormatter,
    OperationMatch,
)

__all__ = [
    # Config
    "ComparisonConfig",
    "ModelProfile",
    "OutputConfig",
    "PhaseType",
    "ProfileDefaults",
    "TensorCastConfig",
    "VLLMConfig",
    "list_profiles",
    "load_profile",
    # Alignment
    "FusionAwareMapper",
    "FusionMapping",
    "FUSION_MAPPINGS",
    "MappingConfig",
    "OpMapper",
    "VLLM_TO_TENSORCAST_MAPPING",
    "list_mappings",
    "load_mappings",
    # Parsers
    "BaseParser",
    "KernelDetailsParser",
    "KernelOp",
    "OperationData",
    "Parser",
    "PhaseDetector",
    "PhaseInfo",
    "ProfilingResult",
    "SingleStepData",
    "StepBoundary",
    "TensorCastAdapter",
    "TensorCastSimulationResult",
    # Output
    "ComparisonResult",
    "ComparisonSummary",
    "ExcelFormatter",
    "OperationMatch",
]
