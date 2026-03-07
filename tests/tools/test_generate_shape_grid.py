"""Tests for generate_shape_grid.py."""

import json

import pytest

from tools.perf_data_collection.generate_shape_grid import (
    generate_shape_grid,
    ModelDims,
)


@pytest.fixture
def model_dims():
    """Qwen3-32B dimensions."""
    return ModelDims(
        model_id="Qwen/Qwen3-32B",
        hidden_size=5120,
        intermediate_size=25600,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=151936,
    )


@pytest.fixture
def dsv3_dims():
    """DSV3 dimensions."""
    return ModelDims(
        model_id="deepseek-ai/DeepSeek-V3",
        hidden_size=7168,
        intermediate_size=18432,
        num_attention_heads=128,
        num_key_value_heads=128,
        head_dim=128,
        vocab_size=129280,
    )


# --- Tests ---


def test_gemm_shapes(model_dims):
    """GEMM grids should have model N/K with M as powers-of-2."""
    grid = generate_shape_grid(model_dims)
    gemm = grid["gemm"]
    assert len(gemm) > 0

    # Should contain model-specific N/K dimensions
    nk_pairs = {(s["N"], s["K"]) for s in gemm}
    # Qwen3: hidden→intermediate (5120→25600) and back (25600→5120)
    assert (5120, 25600) in nk_pairs or (25600, 5120) in nk_pairs

    # M values should include powers of 2
    m_values = {s["M"] for s in gemm}
    assert 1 in m_values  # decode
    assert 128 in m_values
    assert 1024 in m_values


def test_gemm_includes_kv_proj(model_dims):
    """GEMM grid should include KV projection dimensions."""
    grid = generate_shape_grid(model_dims)
    gemm = grid["gemm"]

    nk_pairs = {(s["N"], s["K"]) for s in gemm}
    kv_size = model_dims.num_key_value_heads * model_dims.head_dim  # 8*128=1024
    # hidden→kv_size projection
    assert (model_dims.hidden_size, kv_size) in nk_pairs or (
        kv_size,
        model_dims.hidden_size,
    ) in nk_pairs


def test_attention_shapes(model_dims):
    """Attention grid should have model heads/head_dim with batch×seq grid."""
    grid = generate_shape_grid(model_dims)
    attn = grid["attention"]
    assert len(attn) > 0

    # All entries should have correct num_heads and head_dim
    for s in attn:
        assert s["num_heads"] == 64
        assert s["head_dim"] == 128

    # Should have both small (decode) and large (prefill) seq_len
    seq_lens = {s["avg_seq_len"] for s in attn}
    assert 1 in seq_lens or 2 in seq_lens  # decode-like
    assert any(s >= 1024 for s in seq_lens)  # prefill-like


def test_elementwise_shapes(model_dims):
    """Elementwise grid should use model hidden_size with token count grid."""
    grid = generate_shape_grid(model_dims)
    elem = grid["elementwise"]
    assert len(elem) > 0

    # Should include model hidden_size
    hidden_sizes = {s["hidden_size"] for s in elem}
    assert 5120 in hidden_sizes

    # Token counts should include powers of 2
    token_counts = {s["num_tokens"] for s in elem}
    assert 1 in token_counts
    assert 128 in token_counts


def test_comm_shapes(model_dims):
    """Communication grid should have message_bytes as powers of 2."""
    grid = generate_shape_grid(model_dims)
    comm = grid["communication"]
    assert len(comm) > 0

    msg_bytes = {s["message_bytes"] for s in comm}
    # Should include various message sizes
    assert any(b >= 1024 for b in msg_bytes)
    assert any(b >= 1048576 for b in msg_bytes)  # 1MB+


def test_output_json(model_dims, tmp_path):
    """Grid should be JSON-serializable."""
    grid = generate_shape_grid(model_dims)
    out = tmp_path / "grid.json"
    with out.open("w") as f:
        json.dump(grid, f, indent=2)

    with out.open() as f:
        loaded = json.load(f)
    assert "gemm" in loaded
    assert "attention" in loaded
    assert "elementwise" in loaded
    assert "communication" in loaded


def test_dsv3_includes_moe_dims(dsv3_dims):
    """DSV3 GEMM grid should include MoE-related dimensions."""
    grid = generate_shape_grid(dsv3_dims)
    gemm = grid["gemm"]

    nk_pairs = {(s["N"], s["K"]) for s in gemm}
    # DSV3 MoE: intermediate_size (18432) is per-expert
    assert any(18432 in pair for pair in nk_pairs)


def test_total_shape_count(model_dims):
    """Total shapes should be manageable (design doc says ~5000 per version)."""
    grid = generate_shape_grid(model_dims)
    total = sum(len(v) for v in grid.values())
    assert total > 50, "Too few shapes"
    assert total < 10000, "Too many shapes"


def test_no_duplicate_shapes(model_dims):
    """Each category should have no duplicate shape entries."""
    grid = generate_shape_grid(model_dims)
    for category, shapes in grid.items():
        seen = set()
        for s in shapes:
            key = tuple(sorted(s.items()))
            assert key not in seen, f"Duplicate in {category}: {s}"
            seen.add(key)
