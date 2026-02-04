"""Alignment and mapping utilities for profiling comparison."""

from tensor_cast.scripts.profiling_comparison.alignment.op_mapper import (
    FUSION_MAPPINGS,
    FusionAwareMapper,
    FusionMapping,
    OpMapper,
    VLLM_TO_TENSORCAST_MAPPING,
)
from tensor_cast.scripts.profiling_comparison.alignment.mapping_loader import (
    DirectMapping,
    FusionMapping as YamlFusionMapping,
    MappingConfig,
    get_mapping_path,
    list_mappings,
    load_mappings,
    load_mappings_from_path,
    merge_mappings,
)

__all__ = [
    # Legacy op_mapper exports
    "FusionAwareMapper",
    "FusionMapping",
    "FUSION_MAPPINGS",
    "OpMapper",
    "VLLM_TO_TENSORCAST_MAPPING",
    # New YAML-based mapping loader
    "DirectMapping",
    "YamlFusionMapping",
    "MappingConfig",
    "get_mapping_path",
    "list_mappings",
    "load_mappings",
    "load_mappings_from_path",
    "merge_mappings",
]
