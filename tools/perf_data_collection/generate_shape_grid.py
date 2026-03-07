"""Generate shape traversal grids for microbenchmarking.

Reads model dimensions (from HuggingFace config or CLI args) and generates
shape combinations organized by kernel category:
- GEMM: model N/K + M as powers-of-2 grid
- Attention: model heads/head_dim + batch x seq_len grid
- Elementwise: model hidden_size + num_tokens grid
- Communication: message_bytes powers-of-2 grid

Design doc reference: S6.2 + S7.3 (Shape Grid Strategy)
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class ModelDims:
    """Model dimension parameters for shape grid generation."""

    model_id: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int


# Powers-of-2 M/token grids
_M_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
_SEQ_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
_BATCH_GRID = [1, 2, 4, 8, 16, 32, 64]
_COMM_BYTES_GRID = [
    1024,       # 1KB
    4096,       # 4KB
    16384,      # 16KB
    65536,      # 64KB
    262144,     # 256KB
    1048576,    # 1MB
    4194304,    # 4MB
    16777216,   # 16MB
    67108864,   # 64MB
    268435456,  # 256MB
]


def _unique_nk_pairs(dims: ModelDims) -> List[tuple]:
    """Generate unique (N, K) dimension pairs from model config."""
    h = dims.hidden_size
    inter = dims.intermediate_size
    kv_size = dims.num_key_value_heads * dims.head_dim

    pairs = set()

    # Q projection: hidden -> num_heads * head_dim
    q_size = dims.num_attention_heads * dims.head_dim
    pairs.add((h, q_size))

    # KV projection: hidden -> kv_size
    pairs.add((h, kv_size))

    # Output projection: num_heads * head_dim -> hidden
    pairs.add((q_size, h))

    # Gate/Up projection: hidden -> intermediate
    pairs.add((h, inter))

    # Down projection: intermediate -> hidden
    pairs.add((inter, h))

    # Embedding: vocab -> hidden
    pairs.add((dims.vocab_size, h))

    return sorted(pairs)


def _generate_gemm(dims: ModelDims) -> List[Dict]:
    """Generate GEMM shape grid: model N/K with M as powers-of-2."""
    shapes = []
    seen = set()

    for n, k in _unique_nk_pairs(dims):
        for m in _M_GRID:
            key = (m, n, k)
            if key not in seen:
                seen.add(key)
                shapes.append({"M": m, "N": n, "K": k})

    return shapes


def _generate_attention(dims: ModelDims) -> List[Dict]:
    """Generate attention shape grid: model heads/head_dim with batch x seq grid."""
    shapes = []
    seen = set()

    for batch in _BATCH_GRID:
        for seq in _SEQ_GRID:
            key = (batch, seq, dims.num_attention_heads, dims.head_dim)
            if key not in seen:
                seen.add(key)
                shapes.append({
                    "batch_size": batch,
                    "avg_seq_len": seq,
                    "num_heads": dims.num_attention_heads,
                    "head_dim": dims.head_dim,
                })

    return shapes


def _generate_elementwise(dims: ModelDims) -> List[Dict]:
    """Generate elementwise shape grid: model hidden/intermediate with token grid."""
    shapes = []
    seen = set()

    hidden_sizes = sorted({dims.hidden_size, dims.intermediate_size})

    for h in hidden_sizes:
        for tokens in _M_GRID:
            key = (tokens, h)
            if key not in seen:
                seen.add(key)
                shapes.append({"num_tokens": tokens, "hidden_size": h})

    return shapes


def _generate_comm() -> List[Dict]:
    """Generate communication shape grid: message_bytes powers-of-2."""
    shapes = []
    for msg_bytes in _COMM_BYTES_GRID:
        shapes.append({"message_bytes": msg_bytes})
    return shapes


def generate_shape_grid(dims: ModelDims) -> Dict[str, List[Dict]]:
    """Generate complete shape grid for all kernel categories.

    Args:
        dims: Model dimension parameters

    Returns:
        Dict with keys: gemm, attention, elementwise, communication
        Each value is a list of shape dicts.
    """
    return {
        "gemm": _generate_gemm(dims),
        "attention": _generate_attention(dims),
        "elementwise": _generate_elementwise(dims),
        "communication": _generate_comm(),
    }


def _dims_from_hf(model_id: str) -> ModelDims:
    """Load model dimensions from HuggingFace config."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads

    return ModelDims(
        model_id=model_id,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=getattr(
            config, "num_key_value_heads", config.num_attention_heads
        ),
        head_dim=head_dim,
        vocab_size=config.vocab_size,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate shape traversal grids for microbenchmarking."
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="HuggingFace model ID (e.g., Qwen/Qwen3-32B)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: stdout)",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    dims = _dims_from_hf(args.model_id)
    grid = generate_shape_grid(dims)

    total = sum(len(v) for v in grid.values())
    summary = {cat: len(shapes) for cat, shapes in grid.items()}

    output = {
        "model_id": dims.model_id,
        "model_dims": {
            "hidden_size": dims.hidden_size,
            "intermediate_size": dims.intermediate_size,
            "num_attention_heads": dims.num_attention_heads,
            "num_key_value_heads": dims.num_key_value_heads,
            "head_dim": dims.head_dim,
            "vocab_size": dims.vocab_size,
        },
        "summary": {**summary, "total": total},
        "grids": grid,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(output, f, indent=2)
        print(f"Generated {total} shapes -> {out_path}", file=sys.stderr)
    else:
        json.dump(output, sys.stdout, indent=2)
        print(file=sys.stdout)


if __name__ == "__main__":
    main()
