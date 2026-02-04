"""YAML mapping loader for operator mappings.

This module provides functionality to load many-to-many operator mappings
from YAML files, supporting both VLLM fusions (one VLLM -> many TC) and
TensorCast fusions (many VLLM -> one TC).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

import yaml


# Directory containing mapping YAML files
MAPPINGS_DIR = Path(__file__).parent / "mappings"


@dataclass
class FusionMapping:
    """Defines how ops are fused between VLLM and TensorCast.

    Attributes:
        name: Human-readable name for this fusion
        vllm_ops: List of VLLM operation names
        tc_ops: List of TensorCast operation names
        aggregation: How to combine times ("sum" or "max")
        description: Human-readable description
    """

    name: str
    vllm_ops: List[str]
    tc_ops: List[str]
    aggregation: str = "sum"
    description: str = ""


@dataclass
class DirectMapping:
    """Direct 1:1 mapping between VLLM and TensorCast ops.

    Attributes:
        vllm_op: VLLM operation name
        tc_ops: List of possible TensorCast operation names
    """

    vllm_op: str
    tc_ops: List[str]


@dataclass
class MappingConfig:
    """Complete mapping configuration.

    Attributes:
        version: Mapping file version
        vllm_fusions: VLLM fused ops -> multiple TC ops
        tc_fusions: Multiple VLLM ops -> TC fused op
        direct_mappings: Direct 1:1 mappings
        ignored_vllm_ops: VLLM ops to ignore in comparison
        ignored_tc_ops: TensorCast ops to ignore in comparison
    """

    version: str = "1.0"
    vllm_fusions: List[FusionMapping] = field(default_factory=list)
    tc_fusions: List[FusionMapping] = field(default_factory=list)
    direct_mappings: Dict[str, List[str]] = field(default_factory=dict)
    ignored_vllm_ops: List[str] = field(default_factory=list)
    ignored_tc_ops: List[str] = field(default_factory=list)

    def get_all_vllm_fusion_ops(self) -> Set[str]:
        """Get all VLLM ops involved in any fusion."""
        ops = set()
        for fusion in self.vllm_fusions:
            ops.update(fusion.vllm_ops)
        for fusion in self.tc_fusions:
            ops.update(fusion.vllm_ops)
        return ops

    def get_all_tc_fusion_ops(self) -> Set[str]:
        """Get all TC ops involved in any fusion."""
        ops = set()
        for fusion in self.vllm_fusions:
            ops.update(fusion.tc_ops)
        for fusion in self.tc_fusions:
            ops.update(fusion.tc_ops)
        return ops


def get_mapping_path(mapping_name: str) -> Path:
    """Get the path to a mapping YAML file.

    Args:
        mapping_name: Name of the mapping (e.g., "default", "qwen3")

    Returns:
        Path to the mapping YAML file

    Raises:
        FileNotFoundError: If mapping file doesn't exist
    """
    # Try exact match first
    mapping_path = MAPPINGS_DIR / f"{mapping_name}.yaml"
    if mapping_path.exists():
        return mapping_path

    # Try yml extension
    mapping_path = MAPPINGS_DIR / f"{mapping_name}.yml"
    if mapping_path.exists():
        return mapping_path

    raise FileNotFoundError(
        f"Mapping '{mapping_name}' not found. Looked in: {MAPPINGS_DIR}"
    )


def list_mappings() -> List[str]:
    """List all available mapping names.

    Returns:
        List of mapping names (without extension)
    """
    if not MAPPINGS_DIR.exists():
        return []

    mappings = []
    for path in MAPPINGS_DIR.glob("*.yaml"):
        mappings.append(path.stem)
    for path in MAPPINGS_DIR.glob("*.yml"):
        if path.stem not in mappings:
            mappings.append(path.stem)

    return sorted(mappings)


def _parse_fusion_mapping(data: Dict) -> FusionMapping:
    """Parse a fusion mapping from YAML data.

    Args:
        data: Dictionary from YAML fusion entry

    Returns:
        FusionMapping instance
    """
    return FusionMapping(
        name=data.get("name", "unnamed"),
        vllm_ops=data.get("vllm_ops", []),
        tc_ops=data.get("tc_ops", []),
        aggregation=data.get("aggregation", "sum"),
        description=data.get("description", ""),
    )


def load_mappings(mapping_name: str = "default") -> MappingConfig:
    """Load operator mappings from YAML file.

    Args:
        mapping_name: Name of the mapping file (default: "default")

    Returns:
        MappingConfig instance

    Raises:
        FileNotFoundError: If mapping file doesn't exist
        ValueError: If mapping YAML is invalid
    """
    mapping_path = get_mapping_path(mapping_name)

    with open(mapping_path) as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty mapping file: {mapping_path}")

    config = MappingConfig(version=data.get("version", "1.0"))

    # Parse VLLM fusions
    if "vllm_fusions" in data:
        for fusion_data in data["vllm_fusions"]:
            config.vllm_fusions.append(_parse_fusion_mapping(fusion_data))

    # Parse TC fusions
    if "tc_fusions" in data:
        for fusion_data in data["tc_fusions"]:
            config.tc_fusions.append(_parse_fusion_mapping(fusion_data))

    # Parse direct mappings
    if "direct_mappings" in data:
        for vllm_op, tc_ops in data["direct_mappings"].items():
            if isinstance(tc_ops, str):
                tc_ops = [tc_ops]
            config.direct_mappings[vllm_op] = tc_ops

    # Parse ignored ops
    if "ignored_vllm_ops" in data:
        config.ignored_vllm_ops = data["ignored_vllm_ops"]

    if "ignored_tc_ops" in data:
        config.ignored_tc_ops = data["ignored_tc_ops"]

    return config


def load_mappings_from_path(yaml_path: Path) -> MappingConfig:
    """Load operator mappings from a specific YAML file path.

    Args:
        yaml_path: Path to the YAML file

    Returns:
        MappingConfig instance

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If YAML is invalid
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty mapping file: {yaml_path}")

    config = MappingConfig(version=data.get("version", "1.0"))

    # Parse sections
    if "vllm_fusions" in data:
        for fusion_data in data["vllm_fusions"]:
            config.vllm_fusions.append(_parse_fusion_mapping(fusion_data))

    if "tc_fusions" in data:
        for fusion_data in data["tc_fusions"]:
            config.tc_fusions.append(_parse_fusion_mapping(fusion_data))

    if "direct_mappings" in data:
        for vllm_op, tc_ops in data["direct_mappings"].items():
            if isinstance(tc_ops, str):
                tc_ops = [tc_ops]
            config.direct_mappings[vllm_op] = tc_ops

    if "ignored_vllm_ops" in data:
        config.ignored_vllm_ops = data["ignored_vllm_ops"]

    if "ignored_tc_ops" in data:
        config.ignored_tc_ops = data["ignored_tc_ops"]

    return config


def merge_mappings(*configs: MappingConfig) -> MappingConfig:
    """Merge multiple mapping configs, with later configs taking precedence.

    Args:
        *configs: MappingConfig instances to merge

    Returns:
        Merged MappingConfig
    """
    if not configs:
        return MappingConfig()

    merged = MappingConfig()

    for config in configs:
        # Append fusions (later ones take precedence for same name)
        existing_vllm_names = {f.name for f in merged.vllm_fusions}
        for fusion in config.vllm_fusions:
            if fusion.name in existing_vllm_names:
                # Replace existing
                merged.vllm_fusions = [
                    f for f in merged.vllm_fusions if f.name != fusion.name
                ]
            merged.vllm_fusions.append(fusion)

        existing_tc_names = {f.name for f in merged.tc_fusions}
        for fusion in config.tc_fusions:
            if fusion.name in existing_tc_names:
                merged.tc_fusions = [
                    f for f in merged.tc_fusions if f.name != fusion.name
                ]
            merged.tc_fusions.append(fusion)

        # Update direct mappings (later takes precedence)
        merged.direct_mappings.update(config.direct_mappings)

        # Extend ignored ops (deduplicate)
        merged.ignored_vllm_ops = list(
            set(merged.ignored_vllm_ops) | set(config.ignored_vllm_ops)
        )
        merged.ignored_tc_ops = list(
            set(merged.ignored_tc_ops) | set(config.ignored_tc_ops)
        )

    return merged
