"""Tests for alignment module imports."""

from tensor_cast.scripts.profiling_comparison.alignment import (
    DecompositionConfig,
    load_decomposition_config,
    load_decomposition_config_from_path,
    match_by_sequence,
    merge_decomposition_configs,
    SequenceMatch,
)


class TestAlignmentExports:
    """Tests that alignment module exports are accessible."""

    def test_exports_available(self):
        """Test all expected exports are importable."""
        assert DecompositionConfig is not None
        assert SequenceMatch is not None
        assert load_decomposition_config is not None
        assert load_decomposition_config_from_path is not None
        assert match_by_sequence is not None
        assert merge_decomposition_configs is not None
