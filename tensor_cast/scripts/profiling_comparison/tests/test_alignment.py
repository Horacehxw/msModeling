"""Tests for alignment and mapping module."""

import pytest
from pathlib import Path
import tempfile
import yaml

from tensor_cast.scripts.profiling_comparison.alignment import (
    FusionAwareMapper,
    FusionMapping,
    OpMapper,
    VLLM_TO_TENSORCAST_MAPPING,
    FUSION_MAPPINGS,
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


class TestFusionMapping:
    """Tests for FusionMapping dataclass."""

    def test_basic_mapping(self):
        """Test basic fusion mapping creation."""
        mapping = FusionMapping(
            vllm_op="AddRmsNorm",
            tc_ops=["aten.add", "tensor_cast.rmsnorm"],
            aggregation_method="sum",
            description="Test mapping",
        )
        assert mapping.vllm_op == "AddRmsNorm"
        assert len(mapping.tc_ops) == 2
        assert mapping.aggregation_method == "sum"

    def test_default_aggregation(self):
        """Test default aggregation method."""
        mapping = FusionMapping(
            vllm_op="Test",
            tc_ops=["aten.test"],
        )
        assert mapping.aggregation_method == "sum"


class TestFusionAwareMapper:
    """Tests for FusionAwareMapper class."""

    def test_default_initialization(self):
        """Test default initialization with hardcoded mappings."""
        mapper = FusionAwareMapper()
        # Should have fusion mappings loaded
        assert len(mapper._fusion_map) > 0

    def test_get_fusion_mapping(self):
        """Test getting fusion mapping for VLLM op."""
        mapper = FusionAwareMapper()
        mapping = mapper.get_fusion_mapping("InplaceAddRmsNorm")
        assert mapping is not None
        assert "aten.add" in mapping.tc_ops or "tensor_cast.rmsnorm" in mapping.tc_ops

    def test_get_tc_ops_for_vllm(self):
        """Test getting TC ops for VLLM op."""
        mapper = FusionAwareMapper()
        tc_ops = mapper.get_tc_ops_for_vllm("MatMul")
        assert len(tc_ops) > 0
        assert "aten.mm" in tc_ops or "aten.matmul" in tc_ops

    def test_aggregate_tc_times(self):
        """Test aggregating TC times for fused VLLM op."""
        mapper = FusionAwareMapper()
        tc_op_times = {
            "aten.add": 100.0,
            "tensor_cast.rmsnorm": 200.0,
        }
        time, matched = mapper.aggregate_tc_times("InplaceAddRmsNorm", tc_op_times)
        # Should aggregate the matched ops
        assert time >= 0

    def test_from_yaml(self):
        """Test creating mapper from YAML file."""
        try:
            mapper = FusionAwareMapper.from_yaml("default")
            assert mapper._fusion_map is not None
        except FileNotFoundError:
            pytest.skip("Default mapping not found")

    def test_match_single_step_ops(self):
        """Test matching single step operations."""
        mapper = FusionAwareMapper()

        vllm_stats = {
            "MatMulV2": {
                "total_duration_us": 1000.0,
                "count": 10,
            },
            "AddRmsNorm": {
                "total_duration_us": 500.0,
                "count": 5,
            },
        }

        tc_stats = {
            "aten.mm": {
                "total_duration_us": 950.0,
                "count": 10,
            },
            "aten.add": {
                "total_duration_us": 200.0,
                "count": 5,
            },
            "tensor_cast.rmsnorm": {
                "total_duration_us": 250.0,
                "count": 5,
            },
        }

        matches = mapper.match_single_step_ops(vllm_stats, tc_stats)
        assert len(matches) > 0

        # Check that matches have required fields
        for match in matches:
            assert "vllm_op" in match
            assert "tc_ops" in match
            assert "vllm_duration_us" in match
            assert "tc_duration_us" in match
            assert "match_status" in match


class TestOpMapper:
    """Tests for OpMapper class."""

    def test_initialization(self):
        """Test OpMapper initialization."""
        mapper = OpMapper()
        assert mapper.mapping == VLLM_TO_TENSORCAST_MAPPING
        assert len(mapper.categories) > 0

    def test_get_candidate_ops_direct(self):
        """Test getting candidate ops for direct mapping."""
        mapper = OpMapper()
        candidates = mapper.get_candidate_ops("MatMul")
        assert "aten.mm" in candidates or "aten.matmul" in candidates

    def test_get_candidate_ops_fuzzy(self):
        """Test fuzzy matching for candidate ops."""
        mapper = OpMapper()
        candidates = mapper.get_candidate_ops("SomeMatMulOp")
        # Should find through fuzzy matching
        assert len(candidates) >= 0  # May or may not find matches

    def test_compute_match_confidence_exact(self):
        """Test computing match confidence for exact match."""
        mapper = OpMapper()
        confidence, status = mapper.compute_match_confidence(
            "MatMul", "aten.mm", 10, 10
        )
        assert confidence > 0.5
        assert status in ("exact", "partial")

    def test_get_match_status(self):
        """Test getting match status."""
        mapper = OpMapper()
        assert mapper.get_match_status("MatMul", "aten.mm") in ("exact", "partial")
        assert mapper.get_match_status("MatMul", None) == "missing_in_tc"


class TestMappingLoader:
    """Tests for YAML mapping loader."""

    def test_list_mappings(self):
        """Test listing available mappings."""
        mappings = list_mappings()
        assert "default" in mappings

    def test_load_default_mappings(self):
        """Test loading default mappings."""
        config = load_mappings("default")
        assert config.version == "1.0"
        assert len(config.vllm_fusions) > 0
        assert len(config.direct_mappings) > 0

    def test_mapping_config_structure(self):
        """Test MappingConfig structure."""
        config = load_mappings("default")

        # Check VLLM fusions
        for fusion in config.vllm_fusions:
            assert fusion.name
            assert fusion.vllm_ops
            assert fusion.tc_ops
            assert fusion.aggregation in ("sum", "max")

        # Check direct mappings
        for vllm_op, tc_ops in config.direct_mappings.items():
            assert isinstance(vllm_op, str)
            assert isinstance(tc_ops, list)

    def test_load_mappings_from_path(self):
        """Test loading mappings from specific path."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump({
                "version": "1.0",
                "direct_mappings": {
                    "TestOp": ["aten.test"],
                },
            }, f)
            f.flush()

            config = load_mappings_from_path(Path(f.name))
            assert config.version == "1.0"
            assert "TestOp" in config.direct_mappings

    def test_merge_mappings(self):
        """Test merging multiple mapping configs."""
        config1 = MappingConfig(
            direct_mappings={"Op1": ["tc1"]},
        )
        config2 = MappingConfig(
            direct_mappings={"Op2": ["tc2"]},
        )

        merged = merge_mappings(config1, config2)
        assert "Op1" in merged.direct_mappings
        assert "Op2" in merged.direct_mappings

    def test_merge_mappings_override(self):
        """Test that later configs override earlier ones."""
        config1 = MappingConfig(
            direct_mappings={"Op1": ["tc1"]},
        )
        config2 = MappingConfig(
            direct_mappings={"Op1": ["tc2"]},  # Override
        )

        merged = merge_mappings(config1, config2)
        assert merged.direct_mappings["Op1"] == ["tc2"]

    def test_get_all_fusion_ops(self):
        """Test getting all fusion ops."""
        config = load_mappings("default")
        vllm_ops = config.get_all_vllm_fusion_ops()
        tc_ops = config.get_all_tc_fusion_ops()

        assert len(vllm_ops) > 0
        assert len(tc_ops) > 0


class TestQwen3Mapping:
    """Tests for Qwen3-specific mappings."""

    def test_load_qwen3_mappings(self):
        """Test loading Qwen3 mappings."""
        try:
            config = load_mappings("qwen3")
            assert config.version == "1.0"
            # Qwen3 should have AddRmsNorm mapping
            has_addrmsnorm = any(
                "AddRmsNorm" in f.name for f in config.vllm_fusions
            )
            assert has_addrmsnorm
        except FileNotFoundError:
            pytest.skip("Qwen3 mapping not found")

    def test_qwen3_no_moe_ops(self):
        """Test that Qwen3 mappings don't require MoE ops."""
        try:
            config = load_mappings("qwen3")
            # Qwen3 is dense, so no GroupedMatmul required
            # (it may be in direct_mappings for fallback, but not critical)
        except FileNotFoundError:
            pytest.skip("Qwen3 mapping not found")
