"""YAML configuration loader for profiling comparison tool.

This module provides functions to load model profiles from YAML files.
"""

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .schema import ModelProfile, ProfileDefaults, TensorCastConfig

# Directory containing profile YAML files
PROFILES_DIR = Path(__file__).parent / "profiles"


def get_profile_path(profile_name: str) -> Path:
    """Get the path to a profile YAML file.

    Args:
        profile_name: Name of the profile (e.g., "qwen3-32b")

    Returns:
        Path to the profile YAML file

    Raises:
        FileNotFoundError: If profile file doesn't exist
    """
    # Try exact match first
    profile_path = PROFILES_DIR / f"{profile_name}.yaml"
    if profile_path.exists():
        return profile_path

    # Try with underscores converted to hyphens
    profile_path = PROFILES_DIR / f"{profile_name.replace('_', '-')}.yaml"
    if profile_path.exists():
        return profile_path

    # Try yml extension
    profile_path = PROFILES_DIR / f"{profile_name}.yml"
    if profile_path.exists():
        return profile_path

    raise FileNotFoundError(
        f"Profile '{profile_name}' not found. Looked in: {PROFILES_DIR}"
    )


def list_profiles() -> List[str]:
    """List all available profile names.

    Returns:
        List of profile names (without extension)
    """
    if not PROFILES_DIR.exists():
        return []

    profiles = []
    for path in PROFILES_DIR.glob("*.yaml"):
        profiles.append(path.stem)
    for path in PROFILES_DIR.glob("*.yml"):
        if path.stem not in profiles:
            profiles.append(path.stem)

    return sorted(profiles)


def _parse_tensorcast_config(data: Dict) -> TensorCastConfig:
    """Parse TensorCast configuration from YAML data.

    Args:
        data: Dictionary from YAML tensorcast section

    Returns:
        TensorCastConfig instance
    """
    return TensorCastConfig(
        model_id=data.get("model_id", ""),
        device=data.get("device", "TEST_DEVICE"),
        world_size=data.get("world_size", 1),
        tp_size=data.get("tp_size", 1),
        dp_size=data.get("dp_size", 1),
        ep=data.get("ep", False),
        quantize_linear_action=data.get("quantize_linear_action", "DISABLED"),
        word_embedding_tp=data.get("word_embedding_tp", False),
        lmhead_tp_size=data.get("lmhead_tp_size", 1),
        enable_external_shared_experts=data.get(
            "enable_external_shared_experts", False
        ),
    )


def _parse_defaults(data: Optional[Dict]) -> ProfileDefaults:
    """Parse phase defaults from YAML data.

    Args:
        data: Dictionary from YAML defaults section

    Returns:
        ProfileDefaults instance
    """
    if data is None:
        return ProfileDefaults()

    return ProfileDefaults(
        num_queries=data.get("num_queries", 1),
        query_length=data.get("query_length", 1),
        context_length=data.get("context_length", 0),
    )


def load_profile(profile_name: str) -> ModelProfile:
    """Load a model profile from YAML file.

    Args:
        profile_name: Name of the profile (e.g., "qwen3-32b")

    Returns:
        ModelProfile instance

    Raises:
        FileNotFoundError: If profile file doesn't exist
        ValueError: If profile YAML is invalid
    """
    profile_path = get_profile_path(profile_name)

    with open(profile_path) as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty profile file: {profile_path}")

    # Validate required fields
    if "tensorcast" not in data:
        raise ValueError(
            f"Profile '{profile_name}' missing required 'tensorcast' section"
        )

    tc_data = data["tensorcast"]
    if "model_id" not in tc_data:
        raise ValueError(
            f"Profile '{profile_name}' missing required 'tensorcast.model_id'"
        )

    return ModelProfile(
        name=data.get("name", profile_name),
        description=data.get("description", ""),
        tensorcast=_parse_tensorcast_config(tc_data),
        prefill_defaults=_parse_defaults(data.get("prefill_defaults")),
        decode_defaults=_parse_defaults(data.get("decode_defaults")),
        mapping_file=data.get("mapping_file"),
        num_layers=data.get("num_layers"),
    )


def load_profile_from_path(yaml_path: Path) -> ModelProfile:
    """Load a model profile from a specific YAML file path.

    Args:
        yaml_path: Path to the YAML file

    Returns:
        ModelProfile instance

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If YAML is invalid
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Profile file not found: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty profile file: {yaml_path}")

    if "tensorcast" not in data:
        raise ValueError("Profile missing required 'tensorcast' section")

    tc_data = data["tensorcast"]
    if "model_id" not in tc_data:
        raise ValueError("Profile missing required 'tensorcast.model_id'")

    return ModelProfile(
        name=data.get("name", yaml_path.stem),
        description=data.get("description", ""),
        tensorcast=_parse_tensorcast_config(tc_data),
        prefill_defaults=_parse_defaults(data.get("prefill_defaults")),
        decode_defaults=_parse_defaults(data.get("decode_defaults")),
        mapping_file=data.get("mapping_file"),
        num_layers=data.get("num_layers"),
    )
