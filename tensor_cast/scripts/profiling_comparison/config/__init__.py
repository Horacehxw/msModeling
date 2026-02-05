"""Configuration module for profiling comparison tool."""

from .loader import get_profile_path, list_profiles, load_profile
from .schema import (
    ModelProfile,
    PhaseType,
    ProfileDefaults,
    TensorCastConfig,
)

__all__ = [
    "ModelProfile",
    "PhaseType",
    "ProfileDefaults",
    "TensorCastConfig",
    "load_profile",
    "list_profiles",
    "get_profile_path",
]
