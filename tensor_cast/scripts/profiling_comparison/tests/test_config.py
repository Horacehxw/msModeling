"""Tests for configuration module."""

import tempfile
from pathlib import Path

import pytest
import yaml

from tensor_cast.scripts.profiling_comparison.config import (
    list_profiles,
    load_profile,
    ModelProfile,
    PhaseType,
    ProfileDefaults,
    TensorCastConfig,
)
from tensor_cast.scripts.profiling_comparison.config.loader import (
    _parse_defaults,
    _parse_tensorcast_config,
    load_profile_from_path,
)


class TestPhaseType:
    """Tests for PhaseType enum."""

    def test_phase_values(self):
        """Test PhaseType enum values."""
        assert PhaseType.AUTO.value == "auto"
        assert PhaseType.PREFILL.value == "prefill"
        assert PhaseType.DECODE.value == "decode"

    def test_phase_from_string(self):
        """Test creating PhaseType from string."""
        assert PhaseType("auto") == PhaseType.AUTO
        assert PhaseType("prefill") == PhaseType.PREFILL
        assert PhaseType("decode") == PhaseType.DECODE


class TestTensorCastConfig:
    """Tests for TensorCastConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = TensorCastConfig(model_id="test/model", device="TEST")
        assert config.world_size == 1
        assert config.tp_size == 1
        assert config.dp_size == 1
        assert config.ep is False
        assert config.quantize_linear_action == "DISABLED"

    def test_to_dict(self):
        """Test to_dict conversion."""
        config = TensorCastConfig(
            model_id="Qwen/Qwen3-32B",
            device="ATLAS_800_A3_752T_128G_DIE",
            world_size=16,
            tp_size=16,
        )
        d = config.to_dict()
        assert d["model_id"] == "Qwen/Qwen3-32B"
        assert d["world_size"] == 16
        assert d["tp_size"] == 16


class TestProfileDefaults:
    """Tests for ProfileDefaults dataclass."""

    def test_default_values(self):
        """Test default values."""
        defaults = ProfileDefaults()
        assert defaults.num_queries == 1
        assert defaults.query_length == 1
        assert defaults.context_length == 0

    def test_custom_values(self):
        """Test custom values."""
        defaults = ProfileDefaults(
            num_queries=136,
            query_length=4096,
            context_length=0,
        )
        assert defaults.num_queries == 136
        assert defaults.query_length == 4096


class TestModelProfile:
    """Tests for ModelProfile dataclass."""

    def test_basic_profile(self):
        """Test basic profile creation."""
        tc_config = TensorCastConfig(
            model_id="test/model",
            device="TEST_DEVICE",
        )
        profile = ModelProfile(
            name="test-profile",
            tensorcast=tc_config,
        )
        assert profile.name == "test-profile"
        assert profile.tensorcast.model_id == "test/model"


class TestProfileLoader:
    """Tests for profile loader functions."""

    def test_parse_tensorcast_config(self):
        """Test parsing TensorCast config from dict."""
        data = {
            "model_id": "test/model",
            "device": "TEST",
            "world_size": 8,
            "tp_size": 4,
            "ep": True,
        }
        config = _parse_tensorcast_config(data)
        assert config.model_id == "test/model"
        assert config.world_size == 8
        assert config.tp_size == 4
        assert config.ep is True

    def test_parse_defaults(self):
        """Test parsing defaults from dict."""
        data = {
            "num_queries": 64,
            "query_length": 2048,
            "context_length": 1024,
        }
        defaults = _parse_defaults(data)
        assert defaults.num_queries == 64
        assert defaults.query_length == 2048
        assert defaults.context_length == 1024

    def test_parse_defaults_none(self):
        """Test parsing None defaults."""
        defaults = _parse_defaults(None)
        assert defaults.num_queries == 1
        assert defaults.query_length == 1
        assert defaults.context_length == 0

    def test_load_profile_from_path(self):
        """Test loading profile from path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "name": "test-profile",
                    "tensorcast": {
                        "model_id": "test/model",
                        "device": "TEST",
                    },
                },
                f,
            )
            f.flush()

            profile = load_profile_from_path(Path(f.name))
            assert profile.name == "test-profile"
            assert profile.tensorcast.model_id == "test/model"

    def test_list_profiles(self):
        """Test listing available profiles."""
        profiles = list_profiles()
        # Should have at least the default profiles we created
        assert "qwen3_32b" in profiles or "qwen3-32b" in profiles
        assert "deepseek_v3" in profiles or "deepseek-v3" in profiles

    def test_load_profile_qwen3(self):
        """Test loading Qwen3-32B profile."""
        try:
            profile = load_profile("qwen3_32b")
            assert profile.tensorcast.model_id == "Qwen/Qwen3-32B"
            assert profile.tensorcast.tp_size == 16
            assert profile.decode_defaults.num_queries == 136
        except FileNotFoundError:
            pytest.skip("Profile not found")

    def test_load_profile_not_found(self):
        """Test loading non-existent profile."""
        with pytest.raises(FileNotFoundError):
            load_profile("nonexistent_profile_xyz")
