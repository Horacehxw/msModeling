"""ProfilingDataSource: CSV-backed data source with op_mapping + FRACTAL_NZ.

Design doc reference: S4.2 (ProfilingDataSource), S4.9 (FRACTAL_NZ)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from ...device import DeviceProfile
from .data_source import DataSource, QueryResult, QuerySource


if TYPE_CHECKING:
    from ...device import CommGrid
    from ..op_invoke_info import OpInvokeInfo

logger = logging.getLogger(__name__)

# torch dtype -> Profiling dtype string (design doc S4.2)
DTYPE_MAP = {
    torch.bfloat16: "DT_BF16",
    torch.float16: "DT_BF16",  # FP16 treated as BF16 on Ascend
    torch.int8: "INT8",
    torch.int32: "INT32",
    torch.int64: "INT64",
    torch.float32: "FLOAT",
    torch.bool: "BOOL",
}


def fractal_nz_to_nd(nz_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Restore FRACTAL_NZ tiled shape to ND shape.
    [..., H, W, block_h, block_w] -> [..., H*block_w, W*block_h]

    Design doc S4.9, Appendix B:
    - BF16: [K/16, N/16, 16, 16] -> (K, N)
    - INT8: [N/32, K/16, 16, 32] -> (K, N) after H*block_w, W*block_h
    - Batched: [E, N/32, K/16, 16, 32] -> (E, K, N)
    """
    *batch, H, W, block_h, block_w = nz_shape
    return (*batch, H * block_w, W * block_h)


def _normalize_func_name(func) -> str:
    """Convert torch op to string matching op_mapping.yaml keys.
    e.g. torch.ops.aten.mm.default -> 'aten.mm.default'
         torch.ops.tensor_cast.attention.default -> 'tensor_cast.attention.default'
    """
    s = str(func)
    return s.removeprefix("torch.ops.")


def _parse_shape_str(s: str) -> List[Tuple[int, ...]]:
    """Parse CSV shape string -> list of tuples.
    e.g. '"136,5120;320,48,16,16"' -> [(136,5120), (320,48,16,16)]
    """
    s = s.strip().strip('"')
    shapes = []
    for part in s.split(";"):
        part = part.strip()
        if part:
            shapes.append(tuple(int(x) for x in part.split(",")))
    return shapes


def _parse_str_list(s: str) -> List[str]:
    """Parse 'A;B;C' -> ['A', 'B', 'C']"""
    s = s.strip().strip('"')
    return [x.strip() for x in s.split(";") if x.strip()]


def _parse_fia_q_shape(input_shapes_str: str) -> Optional[Tuple[int, ...]]:
    """Parse slot 0 (Q shape) from FIA CSV Input Shapes string."""
    if not input_shapes_str or not input_shapes_str.strip():
        return None
    parts = input_shapes_str.split(";")
    if not parts or not parts[0].strip():
        return None
    try:
        return tuple(int(x) for x in parts[0].strip().split(","))
    except ValueError:
        return None


def _normalize_fia_q_shape(
    q_shape: Tuple[int, ...], head_dim: int = 0
) -> Optional[Tuple[int, ...]]:
    """Normalize FIA Q shape to 3D (T, N, D).

    3D (T, N, D) → identity; 4D (B, N, 1, D) → squeeze; 2D (T, H) → reshape.
    """
    ndim = len(q_shape)
    if ndim == 3:
        return q_shape
    if ndim == 4 and q_shape[2] == 1:
        return (q_shape[0], q_shape[1], q_shape[3])
    if ndim == 2 and head_dim > 0 and q_shape[1] % head_dim == 0:
        return (q_shape[0], q_shape[1] // head_dim, head_dim)
    return None


def _infer_sparse_mode(query_lens) -> int:
    """Infer FIA sparse_mode from query_lens.

    TC attention op does not pass sparse_mode explicitly.
    Prefill (query_lens has values > 1) uses causal mask -> sparse_mode=3.
    Decode (all query_lens == 1) uses no mask -> sparse_mode=0.
    """
    if query_lens is not None and isinstance(query_lens, torch.Tensor):
        try:
            if query_lens.numel() > 0 and query_lens.max().item() > 1:
                return 3  # right_down_causal
        except Exception:
            pass
    return 0  # no_mask




# while TC's aten.mm receives (K,N) after F.linear transpose.
# FRACTAL_NZ weights restore to (K,N) directly — no transpose needed.
_MATMUL_KERNELS = frozenset(
    {
        "MatMulV2",
        "MatMulV3",
        "MatMulCommon",
        "MatMul",
        "QuantBatchMatmulV3",
        "BatchMatMulV2",
        "TransposeBatchMatMul",
    }
)

_RELAXED_DTYPE_MATMUL_KERNELS = frozenset(
    {
        "MatMulV2",
    }
)

# SwiGlu kernel types: TC dispatches 2 inputs (gate, up) as separate tensors,
# but profiling CSVs store 1 concatenated input along last dim.
_SWIGLU_KERNELS = frozenset({"SwiGlu"})

# RoPE kernel types: TC dispatches (B,H,S,D) layout with [Q, K, cos, sin],
# but profiling CSVs store (B,S,H,D) layout with [K, Q, cos, sin] and
# cos/sin have an extra head dim (1).
_ROPE_KERNELS = frozenset(
    {"ApplyRotaryPosEmb", "_triton_rope", "split_qkv_rmsnorm_rope_kernel"}
)

# Dtype groups that are considered equivalent for matching purposes.
# NPU _triton_rope profiling records K as FLOAT (FP32) while TC dispatches
# BF16 — the kernel internally up-casts, but performance is the same.
_DTYPE_COMPAT = {"DT_BF16": "FLOAT_GROUP", "FLOAT": "FLOAT_GROUP"}

# Kernel types that allow relaxed dtype matching via _DTYPE_COMPAT.
# For non-quant matmul kernels, some model code paths upcast inputs/weights to
# FP32 in eager code for numerical stability, while Ascend profiling records
# the realized kernel as BF16. Allow FLOAT <-> DT_BF16 compatibility so shape
# matching can still reuse the measured kernel entry. Quant matmul kernels keep
# strict dtype matching because their dtype semantics differ from plain matmul.
_DTYPE_RELAXED_KERNELS = _ROPE_KERNELS | _RELAXED_DTYPE_MATMUL_KERNELS

# Kernel types where TC may produce 3D (B, M, D) shapes that should
# match CSV's 2D (B*M, D) shapes by flattening the leading two dims.
# This happens when TC keeps an explicit batch dimension that profiling
# absorbs into the token/sequence dimension.
_FLATTEN_BATCH_KERNELS = frozenset(
    {
        "AscendQuantV2",
        "DynamicQuant",
        "RmsNorm",
        "AddRmsNormBias",
        "AddRmsNorm",
        "DispatchFFNCombine",
    }
)

# Kernel types where TC produces 3D (T, H, D) per-head shapes that should
# match CSV's 2D (T, H*D) shapes by merging the last two dims.
# This is specific to MLA quantize where NPU reshapes to hidden_dim before quantize.
_MERGE_LAST_DIMS_KERNELS = frozenset({"AscendQuantV2", "DynamicQuant"})

# Common NPU tile alignment sizes (Da Vinci Cube unit)
# BF16: 16x16, INT8: 16x32
_BLOCK_SIZES = (16, 32, 64)

# Byte sizes for profiling dtype strings (for elementwise byte-ratio scaling)
_DTYPE_BYTE_SIZES = {
    "DT_BF16": 2,
    "DT_FLOAT16": 2,
    "FLOAT": 4,
    "DT_FLOAT": 4,
    "INT8": 1,
    "DT_INT8": 1,
    "INT16": 2,
    "INT32": 4,
    "DT_INT32": 4,
    "INT64": 8,
    "DT_INT64": 8,
}


def _dtype_byte_size(dtype_str: str) -> int:
    """Return byte size for a profiling dtype string. Returns 0 for unknown."""
    return _DTYPE_BYTE_SIZES.get(dtype_str, 0)


def _normalize_rope_inputs(
    tc_inputs: List[Tuple[Tuple[int, ...], torch.dtype]],
) -> List[Tuple[Tuple[int, ...], torch.dtype]]:
    """Normalize RoPE inputs from TC layout to profiling CSV layout.

    Full (4 inputs):
      TC:  [Q(B,Hq,S,D), K(B,Hk,S,D), cos(1,S,D), sin(1,S,D)]
      CSV: [K(B,S,Hk,D), Q(B,S,Hq,D), cos(B,S,1,D), sin(B,S,1,D)]

    Truncated (2 inputs, tc_input_count=2):
      TC:  [Q(B,Hq,S,D), K(B,Hk,S,D)]
      CSV: [K(B,S,Hk,D), Q(B,S,Hq,D)]

    Transformations:
    1. Swap Q and K (TC: [Q,K,...] → CSV: [K,Q,...])
    2. Transpose H,S dims in Q and K: (B,H,S,D) → (B,S,H,D)
    3. (Full only) Insert head dim=1 for cos/sin: (1,S,D) → (1,S,1,D)
    """
    q_shape, q_dtype = tc_inputs[0]
    k_shape, k_dtype = tc_inputs[1]

    # Transpose Q and K: (B,H,S,D) → (B,S,H,D)
    if len(q_shape) == 4:
        q_shape = (q_shape[0], q_shape[2], q_shape[1], q_shape[3])
    if len(k_shape) == 4:
        k_shape = (k_shape[0], k_shape[2], k_shape[1], k_shape[3])

    # Reorder: [Q, K] → [K, Q]
    result = [
        (k_shape, k_dtype),
        (q_shape, q_dtype),
    ]

    # Process cos/sin if present (full 4-input case)
    if len(tc_inputs) >= 4:
        cos_shape, cos_dtype = tc_inputs[2]
        sin_shape, sin_dtype = tc_inputs[3]
        if len(cos_shape) == 3:
            cos_shape = (cos_shape[0], cos_shape[1], 1, cos_shape[2])
        if len(sin_shape) == 3:
            sin_shape = (sin_shape[0], sin_shape[1], 1, sin_shape[2])
        result.append((cos_shape, cos_dtype))
        result.append((sin_shape, sin_dtype))

    return result


def _strip_batch_dim(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Strip leading batch dim=1 from TC shapes.
    TC keeps explicit batch: (1, seq, dim). Profiling flattens: (seq, dim).
    Only strip if leading dim is exactly 1.
    """
    if len(shape) > 1 and shape[0] == 1:
        return shape[1:]
    return shape


def _is_block_padded(tc_dim: int, csv_dim: int) -> bool:
    """Check if tc_dim is a block-padded version of csv_dim.
    TC pads sequence dims to NPU tile alignment; profiling stores unpadded shapes.
    """
    if tc_dim <= csv_dim:
        return False
    for bs in _BLOCK_SIZES:
        padded = ((csv_dim + bs - 1) // bs) * bs
        if tc_dim == padded:
            return True
    return False


def get_topology_tier(comm_grid: "CommGrid", group: List[int]) -> int:
    """Determine topology tier index for a communication group.

    Finds the outermost grid dimension where ranks differ, then returns the
    most specific (fastest) topology that covers that span.

    Mirrors CommAnalyticModel._get_topology_idx_for_group logic, but operates
    directly on CommGrid to avoid importing the model layer.

    Args:
        comm_grid: CommGrid with .grid (torch.Tensor) and .topologies (dict).
        group: list of rank IDs in the communication group.

    Returns:
        topology tier index (key into comm_grid.topologies).
    """

    def _rank_to_coord(rank: int) -> List[int]:
        coord = []
        temp = rank
        for dim_size in reversed(comm_grid.grid.shape):
            coord.insert(0, temp % dim_size)
            temp //= dim_size
        return coord

    coords = [_rank_to_coord(r) for r in group]

    diff_dim = -1
    for dim_idx in range(comm_grid.grid.dim()):
        first = coords[0][dim_idx]
        if any(c[dim_idx] != first for c in coords[1:]):
            diff_dim = dim_idx
            break

    if diff_dim == -1:
        # All ranks identical (shouldn't happen for group > 1); use fastest tier.
        return max(comm_grid.topologies.keys())

    for start_dim in sorted(comm_grid.topologies.keys(), reverse=True):
        if start_dim <= diff_dim:
            return start_dim

    raise ValueError(f"No topology found for group spanning grid dimension {diff_dim}")


# ---- MLA / MLAPO composite decomposition (design doc §4.2, G2) ----


@dataclass
class SubKernelSpec:
    """Specification for a sub-kernel in composite decomposition."""

    kernel_type: str
    input_shapes: List[Tuple[int, ...]]
    dtype: str  # Profiling dtype string, e.g. "DT_BF16"
    query_mode: str = "compute"  # "compute" | "attention"
    attention_params: Optional[Dict[str, Any]] = field(default=None)


def _is_decode_mla(args: tuple) -> bool:
    """Determine if MLA op is in decode mode.

    query_lens (args[5]) is None or all 1s → decode.
    """
    query_lens = args[5]
    if query_lens is None:
        return True
    if isinstance(query_lens, torch.Tensor):
        try:
            return bool(query_lens.max().item() <= 1)
        except Exception:
            return True
    return True


def _decompose_mla_common(
    op_invoke_info: "OpInvokeInfo",
    mapping: dict,
    first_kernel_type: str,
) -> Optional[List[SubKernelSpec]]:
    """Shared MLA decomposition for BF16 and quantized variants.

    Decode: first_kernel_type(q@W_UK_T) + FIA + TransposeBatchMatMul(out@W_UV)
    Prefill: MatMulV2(kv_c@kv_b_proj) + RINGMLAPrefillBF16Kernel (compute match)

    Args:
        first_kernel_type: "TransposeBatchMatMul" for BF16, "QuantBatchMatmulV3" for quant.
    """
    args = op_invoke_info.args
    if len(args) < 10:
        return None
    q = args[0]  # (num_tokens, num_heads, qk_head_dim)
    seq_lens = args[4]  # (batch_size,)
    dtype_str = DTYPE_MAP.get(q.dtype)
    if dtype_str is None:
        return None

    if not isinstance(seq_lens, torch.Tensor):
        return None
    batch_size = seq_lens.shape[0]
    num_heads = q.shape[1]
    kv_cache = args[1]  # (total_blocks, block_size, kv_lora_rank + qk_rope_head_dim)
    head_dim = kv_cache.shape[-1]
    num_tokens = q.shape[0]

    if _is_decode_mla(args):
        W_UK_T = args[6]  # (num_heads, qk_nope_head_dim, kv_lora_rank)
        W_UV = args[7]  # (num_heads, kv_lora_rank, v_head_dim)
        if W_UK_T is None or W_UV is None:
            return None

        qk_nope_head_dim = W_UK_T.shape[1]
        kv_lora_rank = W_UK_T.shape[2]
        v_head_dim_val = W_UV.shape[2]

        try:
            avg_seq_len = int(seq_lens.float().mean().item())
        except (RuntimeError, ValueError):
            avg_seq_len = 0

        fia_q_raw = (batch_size, num_heads, 1, head_dim)
        fia_q_normalized = _normalize_fia_q_shape(fia_q_raw, head_dim)

        fia_spec = SubKernelSpec(
            kernel_type="FusedInferAttentionScore",
            input_shapes=[],
            dtype=dtype_str,
            query_mode="attention",
            attention_params={
                "q_shape_3d": fia_q_normalized or (batch_size, num_heads, head_dim),
                "avg_seq_len": avg_seq_len,
                "sparse_mode": 0,  # decode uses no_mask
                "num_kv_heads": 1,  # MLA compressed attention: single KV head
            },
        )

        return [
            SubKernelSpec(
                kernel_type=first_kernel_type,
                input_shapes=[
                    (num_tokens, num_heads, qk_nope_head_dim),
                    (num_heads, qk_nope_head_dim, kv_lora_rank),
                ],
                dtype=dtype_str,
            ),
            fia_spec,
            SubKernelSpec(
                kernel_type="TransposeBatchMatMul",
                input_shapes=[
                    (num_tokens, num_heads, kv_lora_rank),
                    (num_heads, kv_lora_rank, v_head_dim_val),
                ],
                dtype=dtype_str,
            ),
        ]
    else:
        # Prefill: MatMulV2(kv_c@kv_b_proj) + FusedInferAttentionScore
        # vllm-ascend v0.18.0: MLA prefill uses FIA (unified, RING kernel removed)
        kv_b_proj = args[8]  # (kv_lora_rank, num_heads*(qk_nope_head_dim+v_head_dim))
        if kv_b_proj is None:
            logger.debug("MLA prefill: kv_b_proj is None, fallback to analytic")
            return None

        kv_lora_rank = kv_b_proj.shape[0]
        try:
            avg_seq_len = int(seq_lens.float().mean().item())
        except (RuntimeError, ValueError):
            avg_seq_len = 0
        fia_q_raw = (batch_size, num_heads, 1, head_dim)
        fia_q_normalized = _normalize_fia_q_shape(fia_q_raw, head_dim)

        return [
            SubKernelSpec(
                kernel_type="MatMulV2",
                input_shapes=[
                    (num_tokens, kv_lora_rank),
                    tuple(kv_b_proj.shape),
                ],
                dtype=dtype_str,
            ),
            SubKernelSpec(
                kernel_type="FusedInferAttentionScore",
                input_shapes=[],
                dtype=dtype_str,
                query_mode="attention",
                attention_params={
                    "q_shape_3d": fia_q_normalized
                    or (batch_size, num_heads, head_dim),
                    "avg_seq_len": avg_seq_len,
                    "sparse_mode": 0,
                    "num_kv_heads": 1,
                },
            ),
        ]


def _decompose_mla(
    op_invoke_info: "OpInvokeInfo", mapping: dict
) -> Optional[List[SubKernelSpec]]:
    """Decompose multihead_latent_attention (BF16)."""
    return _decompose_mla_common(op_invoke_info, mapping, "TransposeBatchMatMul")


def _decompose_mla_quant(
    op_invoke_info: "OpInvokeInfo", mapping: dict
) -> Optional[List[SubKernelSpec]]:
    """Decompose multihead_latent_attention_quant."""
    return _decompose_mla_common(op_invoke_info, mapping, "QuantBatchMatmulV3")


def _decompose_mlapo(
    op_invoke_info: "OpInvokeInfo", mapping: dict
) -> Optional[List[SubKernelSpec]]:
    """Decompose mlapo into projection MatMuls + KvRmsNormRopeCache.

    TC mlapo fuses: q_a_proj + q_a_norm + q_b_proj + kv_a_proj + kv_a_norm + rope.
    NPU runs these as separate kernels. Decompose to match profiling data.

    Args layout (tensor_cast/ops/mla.py):
        args[0]: hidden_states (num_tokens, hidden_size)
        args[3]: q_a_proj_weight (q_lora_rank, hidden_size) — Optional
            F.linear convention: (out_features, in_features)
        args[5]: q_b_proj_weight (num_heads*qk_head_dim, q_lora_rank) — Optional
        args[6]: kv_a_proj_weight (kv_lora_rank+rope_dim, hidden_size) — Optional
        args[7]: kv_a_layernorm_weight (kv_lora_rank,)
        args[12]: kv_lora_rank (int)
    """
    args = op_invoke_info.args
    if len(args) < 14:
        return None

    hidden_states = args[0]  # (num_tokens, hidden_size)
    q_a_proj = args[3]  # (q_lora_rank, hidden_size)
    q_b_proj = args[5]  # (num_heads*qk_head_dim, q_lora_rank)
    kv_a_proj = args[6]  # (kv_lora_rank+rope_dim, hidden_size)

    if hidden_states is None or q_a_proj is None or q_b_proj is None or kv_a_proj is None:
        return None

    dtype_str = DTYPE_MAP.get(hidden_states.dtype)
    if dtype_str is None:
        return None

    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    q_lora_rank = q_a_proj.shape[0]  # out_features of q_a_proj
    kv_proj_dim = kv_a_proj.shape[0]  # out_features: kv_lora_rank + rope_dim

    return [
        # Op1: hidden @ q_a_proj
        SubKernelSpec(
            kernel_type="MatMulV2",
            input_shapes=[(num_tokens, hidden_size), tuple(q_a_proj.shape)],
            dtype=dtype_str,
        ),
        # Op3: q_compressed @ q_b_proj
        SubKernelSpec(
            kernel_type="MatMulV2",
            input_shapes=[(num_tokens, q_lora_rank), tuple(q_b_proj.shape)],
            dtype=dtype_str,
        ),
        # Op5: hidden @ kv_a_proj
        SubKernelSpec(
            kernel_type="MatMulV2",
            input_shapes=[(num_tokens, hidden_size), tuple(kv_a_proj.shape)],
            dtype=dtype_str,
        ),
        # Op6+7: kv norm + rope (post-projection)
        SubKernelSpec(
            kernel_type="KvRmsNormRopeCache",
            input_shapes=[(num_tokens, kv_proj_dim)],
            dtype=dtype_str,
        ),
    ]


def _decompose_mlapo_quant(
    op_invoke_info: "OpInvokeInfo", mapping: dict
) -> Optional[List[SubKernelSpec]]:
    """Decompose mlapo_quant — same structure, QuantBatchMatmulV3 for projections.

    Weight shapes follow F.linear convention: (out_features, in_features).
    """
    args = op_invoke_info.args
    if len(args) < 20:
        return None

    hidden_states = args[0]  # (num_tokens, hidden_size)
    q_a_proj = args[3]  # (q_lora_rank, hidden_size)
    q_b_proj = args[5]  # (num_heads*qk_head_dim, q_lora_rank)
    kv_a_proj = args[6]  # (kv_lora_rank+rope_dim, hidden_size)

    if hidden_states is None or q_a_proj is None or q_b_proj is None or kv_a_proj is None:
        return None

    dtype_str = DTYPE_MAP.get(hidden_states.dtype)
    if dtype_str is None:
        return None

    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    q_lora_rank = q_a_proj.shape[0]  # out_features of q_a_proj
    kv_proj_dim = kv_a_proj.shape[0]  # out_features: kv_lora_rank + rope_dim

    return [
        SubKernelSpec(
            kernel_type="QuantBatchMatmulV3",
            input_shapes=[(num_tokens, hidden_size), tuple(q_a_proj.shape)],
            dtype=dtype_str,
        ),
        SubKernelSpec(
            kernel_type="QuantBatchMatmulV3",
            input_shapes=[(num_tokens, q_lora_rank), tuple(q_b_proj.shape)],
            dtype=dtype_str,
        ),
        SubKernelSpec(
            kernel_type="QuantBatchMatmulV3",
            input_shapes=[(num_tokens, hidden_size), tuple(kv_a_proj.shape)],
            dtype=dtype_str,
        ),
        SubKernelSpec(
            kernel_type="KvRmsNormRopeCache",
            input_shapes=[(num_tokens, kv_proj_dim)],
            dtype=dtype_str,
        ),
    ]


COMPOSITE_DECOMPOSERS: Dict[
    str,
    Callable[["OpInvokeInfo", dict], Optional[List[SubKernelSpec]]],
] = {
    "tensor_cast.multihead_latent_attention.default": _decompose_mla,
    "tensor_cast.multihead_latent_attention_quant.default": _decompose_mla_quant,
    "tensor_cast.mlapo.default": _decompose_mlapo,
    "tensor_cast.mlapo_quant.default": _decompose_mlapo_quant,
}


class ProfilingDataSource(DataSource):
    """CSV-backed data source with op_mapping.yaml + FRACTAL_NZ.

    Design doc S4.2: internally handles all mapping, shape extraction,
    format conversion. The caller (EmpiricalPerformanceModel) only calls
    lookup(OpInvokeInfo).

    Init args:
        data_dir: path containing op_mapping.yaml + {KernelType}.csv files
        device_profile: DeviceProfile for comm_grid topology_tier resolution.
            Optional — when omitted, communication lookups skip tier filtering.
    """

    def __init__(
        self, data_dir: str | Path, device_profile: Optional[DeviceProfile] = None
    ):
        self.data_dir = Path(data_dir)
        self.comm_grid = device_profile.comm_grid if device_profile else None
        self._op_mapping = self._load_op_mapping()
        self._csv_cache: Dict[str, Optional[pd.DataFrame]] = {}
        # Resolve communication data directory from op_mapping communication_data_ref.
        # Falls back to data_dir when the field is absent (legacy layout).
        # NOTE: when _comm_data_dir == data_dir, the fallback in _load_csv is
        # redundant but harmless — kept for clarity over micro-optimization.
        comm_ref = self._op_mapping.get("communication_data_ref")
        if comm_ref:
            self._comm_data_dir = (self.data_dir / comm_ref).resolve()
        else:
            self._comm_data_dir = self.data_dir
        # Set after each lookup() miss to explain why
        self.last_miss_reason: str = ""

    def _load_op_mapping(self) -> dict:
        yaml_path = self.data_dir / "op_mapping.yaml"
        if not yaml_path.exists():
            logger.warning("op_mapping.yaml not found at %s", yaml_path)
            return {}
        with open(yaml_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_csv(self, kernel_type: str) -> Optional[pd.DataFrame]:
        # Convention: comm kernel_types use lowercase hcom_ prefix (e.g. hcom_allReduce_).
        # CamelCase variants (HcomAllReduce) are graph-compiled names and should be
        # listed in alternate_kernel_types, not as primary kernel_type.
        if kernel_type in self._csv_cache:
            return self._csv_cache[kernel_type]
        csv_path = self.data_dir / f"{kernel_type}.csv"
        if (
            not csv_path.exists()
            and self._comm_data_dir
            and kernel_type.startswith("hcom_")
        ):
            alt_path = self._comm_data_dir / f"{kernel_type}.csv"
            if alt_path.exists():
                csv_path = alt_path
        if not csv_path.exists():
            logger.debug("CSV not found: %s", csv_path)
            self._csv_cache[kernel_type] = None
            return None
        df = pd.read_csv(csv_path)
        self._csv_cache[kernel_type] = df
        return df

    @staticmethod
    def _latency_col(df: pd.DataFrame) -> str:
        """Return the latency column name present in *df*.

        Priority: MicroBench (op-replay) > Profiling Average (enriched CSV)
        > Average (legacy parse_kernel_details) > Duration (comm / earliest).
        """
        for col in (
            "MicroBench Duration(us)",
            "Profiling Average Duration(us)",
            "Average Duration(us)",
            "Duration(us)",
        ):
            if col in df.columns:
                return col
        return "Duration(us)"

    def _query_comm_csv(
        self,
        kernel_type: str,
        message_bytes: int,
        num_devices: int,
        topology_tier: Optional[int],
    ) -> Optional[Tuple[float, bool]]:
        """Shared comm CSV query with interpolation fallback.

        Tries exact match first. On miss, interpolates linearly on message_bytes
        (num_devices + topology_tier remain exact). Interpolation is default
        behavior because message_bytes is continuous and exact match rarely works.

        Returns (latency_us, is_interpolated) or None on miss.
        Sets self.last_miss_reason on failure.
        """
        df = self._load_csv(kernel_type)
        if df is None:
            self.last_miss_reason = "csv_not_found"
            return None

        required_cols = {"message_bytes", "num_devices"}
        if not required_cols.issubset(df.columns):
            logger.debug(
                "MISS (comm) %s: CSV missing columns %s, need microbenchmark format",
                kernel_type,
                required_cols - set(df.columns),
            )
            self.last_miss_reason = "csv_format_raw"
            return None

        lat_col = self._latency_col(df)

        # --- Exact match ---
        mask = (df["message_bytes"] == message_bytes) & (
            df["num_devices"] == num_devices
        )
        if topology_tier is not None and "topology_tier" in df.columns:
            mask = mask & (df["topology_tier"] == topology_tier)

        matched = df[mask]
        if not matched.empty:
            return (float(matched.iloc[0][lat_col]), False)

        # --- Interpolation fallback: bracket message_bytes ---
        device_mask = df["num_devices"] == num_devices
        if topology_tier is not None and "topology_tier" in df.columns:
            device_mask = device_mask & (df["topology_tier"] == topology_tier)
        candidates = df[device_mask]

        if candidates.empty:
            logger.debug(
                "MISS (comm) %s: no rows for num_devices=%d, topology_tier=%s",
                kernel_type,
                num_devices,
                topology_tier,
            )
            self.last_miss_reason = "shape_mismatch"
            return None

        mb_values = candidates["message_bytes"].values
        below = mb_values[mb_values <= message_bytes]
        above = mb_values[mb_values >= message_bytes]

        if len(below) == 0 or len(above) == 0:
            logger.debug(
                "MISS (comm) %s: message_bytes=%d outside range [%d, %d]",
                kernel_type,
                message_bytes,
                int(mb_values.min()),
                int(mb_values.max()),
            )
            self.last_miss_reason = "shape_mismatch"
            return None

        mb_lo, mb_hi = int(below.max()), int(above.min())
        lat_lo = float(
            candidates.loc[candidates["message_bytes"] == mb_lo, lat_col].iloc[0]
        )
        if mb_lo == mb_hi:
            return (lat_lo, False)  # degenerate bracket = exact

        lat_hi = float(
            candidates.loc[candidates["message_bytes"] == mb_hi, lat_col].iloc[0]
        )

        # Alpha-beta interpolation: comm latency = alpha + message_bytes / bandwidth
        # Fit from ALL candidate data points (least-squares) rather than just the
        # bracket endpoints. This gives a global alpha-beta model for this
        # (num_devices, topology_tier) group, which handles the latency-dominated →
        # bandwidth-dominated transition more accurately than piecewise linear.
        all_mb = candidates["message_bytes"].values.astype(np.float64)
        all_lat = candidates[lat_col].values.astype(np.float64)

        if len(all_mb) >= 2:
            A = np.column_stack([np.ones_like(all_mb), all_mb])
            params, _, _, _ = np.linalg.lstsq(A, all_lat, rcond=None)
            interpolated = float(params[0] + params[1] * message_bytes)
        else:
            # Fallback: single-point, use that value
            interpolated = float(all_lat[0])

        # Clamp to bracket bounds (safety: don't go below lower or above upper)
        interpolated = max(min(lat_lo, lat_hi), min(interpolated, max(lat_lo, lat_hi)))

        logger.debug(
            "HIT (comm interpolated) %s: message_bytes=%d between "
            "[%d (%.1fus), %d (%.1fus)] → %.1fus (alpha-beta fit from %d points)",
            kernel_type,
            message_bytes,
            mb_lo,
            lat_lo,
            mb_hi,
            lat_hi,
            interpolated,
            len(all_mb),
        )
        return (interpolated, True)

    # ---- Main lookup (design doc S4.2 dispatch logic) ----

    def lookup(self, op_invoke_info: "OpInvokeInfo") -> Optional[QueryResult]:
        """Query perf data for an op.

        Dispatch logic (design doc S4.2):
          func_name -> op_mapping.yaml
            - not found -> return None
            - composite == true -> _lookup_composite()
            - category == "communication" -> _lookup_comm()
            - query_mode == "attention_special" -> _lookup_attention()
            - query_mode == "elementwise" -> _lookup_elementwise()
            - zero_cost == true -> return QueryResult(0.0)
            - default -> _lookup_compute()
        """
        func_str = _normalize_func_name(op_invoke_info.func)
        mappings = self._op_mapping.get("operator_mappings", {})
        mapping = mappings.get(func_str)
        if mapping is None:
            self.last_miss_reason = "unmapped"
            return None

        # Composite ops: try decomposition via sub_kernels, else skip
        if mapping.get("composite"):
            return self._lookup_composite(op_invoke_info, mapping)
        if mapping.get("category") == "communication":
            return self._lookup_comm(op_invoke_info, mapping)
        if mapping.get("query_mode") == "attention_special":
            return self._lookup_attention(op_invoke_info, mapping)
        if mapping.get("query_mode") == "elementwise":
            return self._lookup_elementwise(op_invoke_info, mapping)

        # Zero-cost ops: shape-only operations with no kernel execution
        if mapping.get("zero_cost"):
            return QueryResult(
                latency_us=0.0,
                confidence=1.0,
                source=QuerySource.MEASURED,
                details={"kernel_type": "zero_cost", "zero_cost": True},
            )

        return self._lookup_compute(op_invoke_info, mapping)

    # ---- Composite op lookup ----

    def _lookup_composite(
        self, op_invoke_info: "OpInvokeInfo", mapping: dict
    ) -> Optional[QueryResult]:
        """Decompose composite ops and sum sub-kernel latencies.

        For MLA/MLAPO: uses registered decomposer to derive sub-kernel shapes,
        then queries each sub-kernel individually.
        For MC2 (matmul+comm): queries both compute and comm sub-kernels.
        Returns None if any required sub-kernel misses.
        """
        # Check for registered decomposer (MLA/MLAPO)
        func_str = _normalize_func_name(op_invoke_info.func)
        decomposer = COMPOSITE_DECOMPOSERS.get(func_str)
        if decomposer is not None:
            return self._lookup_composite_decomposed(
                op_invoke_info, mapping, decomposer
            )

        # Generic composite path (MC2 etc.)
        sub_kernels = mapping.get("sub_kernels", [])
        if not sub_kernels:
            self.last_miss_reason = "no_sub_kernels"
            return None

        tc_inputs = self._extract_tensor_inputs(op_invoke_info)

        # tc_input_count truncation (same as _lookup_compute):
        # quant MC2 ops have 6 tensor args but CSV only needs x + w
        tc_input_count = mapping.get("tc_input_count")
        if tc_input_count is not None:
            tc_inputs = tc_inputs[:tc_input_count]

        # --- Compute sub-kernels ---
        compute_latency = None
        compute_kernel_hit = None
        any_compute_csv = False

        for kernel_type in sub_kernels:
            if kernel_type.startswith("hcom_"):
                continue
            df = self._load_csv(kernel_type)
            if df is None:
                continue
            any_compute_csv = True
            for _, row in df.iterrows():
                if self._inputs_match(
                    tc_inputs,
                    row,
                    kernel_type=kernel_type,
                    tc_input_count=tc_input_count,
                ):
                    compute_latency = float(row[self._latency_col(df)])
                    compute_kernel_hit = kernel_type
                    logger.debug(
                        "HIT (composite compute) %s: %.1f us",
                        kernel_type,
                        compute_latency,
                    )
                    break
            if compute_latency is not None:
                break

        if compute_latency is None:
            self.last_miss_reason = (
                "csv_not_found" if not any_compute_csv else "shape_mismatch"
            )
            return None

        # --- Communication sub-kernels ---
        # Convention: comm sub_kernels must use hcom_ prefix (lowercase).
        # CamelCase names (HcomAllReduce) are graph-compiled variants and
        # should only appear in alternate_kernel_types.
        # NOTE: _lookup_comm_for_composite assumes matmul+comm arg layout:
        #   args[0]=mat1, args[1]=mat2, args[-1]=rank_group.
        # This holds for all current MC2 variants (matmul_all_reduce,
        # static_quant_linear_all_reduce, fp8_linear_all_reduce, etc.).
        # If a future composite op has a different arg layout, this will
        # need per-op dispatch or a mapping-driven arg index scheme.
        comm_latency = 0.0
        has_comm = False
        for kernel_type in sub_kernels:
            if not kernel_type.startswith("hcom_"):
                continue
            has_comm = True
            lat = self._lookup_comm_for_composite(op_invoke_info, kernel_type)
            if lat is None:
                self.last_miss_reason = "comm_sub_kernel_miss"
                return None
            comm_latency += lat
            logger.debug("HIT (composite comm) %s: %.1f us", kernel_type, lat)

        return QueryResult(
            latency_us=compute_latency + comm_latency,
            confidence=0.9 if has_comm else 0.8,
            source=QuerySource.MEASURED,
            details={
                "kernel_type": compute_kernel_hit,
                "composite": True,
                "note": (
                    "compute + comm sub-kernels"
                    if has_comm
                    else "compute sub-kernel only"
                ),
            },
        )

    def _lookup_composite_decomposed(
        self,
        op_invoke_info: "OpInvokeInfo",
        mapping: dict,
        decomposer: Callable,
    ) -> Optional[QueryResult]:
        """Query composite op using registered decomposer (MLA/MLAPO).

        Calls the decomposer to get SubKernelSpec list, then queries each
        sub-kernel via _lookup_compute_by_shapes or _lookup_attention_by_params.
        """
        specs = decomposer(op_invoke_info, mapping)
        if not specs:
            self.last_miss_reason = "decompose_failed"
            return None

        total_latency = 0.0
        hit_kernels = []

        for spec in specs:
            if spec.query_mode == "attention" and spec.attention_params:
                lat = self._lookup_attention_by_params(
                    spec.kernel_type, spec.attention_params, spec.dtype
                )
            else:
                lat = self._lookup_compute_by_shapes(
                    spec.kernel_type, spec.input_shapes, spec.dtype
                )

            if lat is None:
                self.last_miss_reason = f"sub_kernel_miss:{spec.kernel_type}"
                return None

            total_latency += lat
            hit_kernels.append(spec.kernel_type)

        logger.debug(
            "HIT (composite decomposed) %s: sub_kernels=%s, total=%.1f us",
            _normalize_func_name(op_invoke_info.func),
            hit_kernels,
            total_latency,
        )
        return QueryResult(
            latency_us=total_latency,
            confidence=0.8,
            source=QuerySource.MEASURED,
            details={
                "kernel_type": hit_kernels,
                "composite": True,
                "note": "decomposed sub-kernels",
            },
        )

    def _lookup_compute_by_shapes(
        self,
        kernel_type: str,
        input_shapes: List[Tuple[int, ...]],
        dtype_str: str,
    ) -> Optional[float]:
        """Query compute CSV by explicit shapes + dtype (for composite decomposition)."""
        df = self._load_csv(kernel_type)
        if df is None:
            return None

        # Build synthetic tc_inputs for _inputs_match
        torch_dtype = None
        for k, v in DTYPE_MAP.items():
            if v == dtype_str:
                torch_dtype = k
                break
        if torch_dtype is None:
            return None

        tc_inputs = [(shape, torch_dtype) for shape in input_shapes]

        for _, row in df.iterrows():
            if self._inputs_match(tc_inputs, row, kernel_type=kernel_type):
                lat = float(row[self._latency_col(df)])
                logger.debug(
                    "HIT (by_shapes) %s: shapes=%s -> %.1f us",
                    kernel_type,
                    input_shapes,
                    lat,
                )
                return lat
        return None

    def _lookup_attention_by_params(
        self,
        kernel_type: str,
        params: Dict[str, Any],
        dtype_str: str,
    ) -> Optional[float]:
        """Query FIA enriched CSV by explicit params (for composite decomposition).

        params must contain: q_shape_3d (tuple), avg_seq_len (int).
        Optional: sparse_mode (int), num_kv_heads (int).
        """
        df = self._load_csv(kernel_type)
        if df is None:
            return None

        # Unified column naming: prefer "Runtime avg_seq_len", fall back to "avg_seq_len"
        avg_seq_col = None
        if "Runtime avg_seq_len" in df.columns:
            avg_seq_col = "Runtime avg_seq_len"
        elif "avg_seq_len" in df.columns:
            avg_seq_col = "avg_seq_len"
        else:
            return None

        if "Input Shapes" not in df.columns:
            return None

        q_shape_3d = params.get("q_shape_3d")
        target_avg_seq = params.get("avg_seq_len")
        if q_shape_3d is None or target_avg_seq is None:
            return None

        target_sparse = params.get("sparse_mode")
        target_kv_heads = params.get("num_kv_heads")

        has_sparse_col = "Runtime sparse_mode" in df.columns
        has_kv_heads_col = "Runtime num_key_value_heads" in df.columns

        tc_N, tc_D = q_shape_3d[1], q_shape_3d[2]
        head_dim = tc_D
        latency_col = self._latency_col(df)

        for _, row in df.iterrows():
            csv_avg_seq = int(row[avg_seq_col])
            if csv_avg_seq < 0:
                continue

            shapes_str = str(row.get("Input Shapes", "")).strip('"')
            csv_q_raw = _parse_fia_q_shape(shapes_str)
            if csv_q_raw is None:
                continue
            csv_q_3d = _normalize_fia_q_shape(csv_q_raw, head_dim)
            if csv_q_3d is None:
                continue

            csv_N, csv_D = csv_q_3d[1], csv_q_3d[2]

            csv_dtypes_str = str(row.get("Input Data Types", ""))
            csv_first_dtype = csv_dtypes_str.split(";")[0].strip() if csv_dtypes_str else ""
            if dtype_str != csv_first_dtype:
                continue

            if tc_N != csv_N or tc_D != csv_D:
                continue

            # sparse_mode exact match (skip if CSV lacks column or param)
            if has_sparse_col and target_sparse is not None:
                if int(row["Runtime sparse_mode"]) != target_sparse:
                    continue

            # num_kv_heads exact match (skip if CSV lacks column or param)
            if has_kv_heads_col and target_kv_heads is not None:
                if int(row["Runtime num_key_value_heads"]) != target_kv_heads:
                    continue

            if target_avg_seq != csv_avg_seq:
                continue

            # T dim: block-padding tolerance
            tc_T = q_shape_3d[0]
            csv_T = csv_q_3d[0]
            if tc_T != csv_T:
                if not _is_block_padded(tc_T, csv_T) and not _is_block_padded(csv_T, tc_T):
                    continue

            lat = float(row[latency_col])
            logger.debug(
                "HIT (attention by_params) %s: params=%s -> %.1f us",
                kernel_type,
                params,
                lat,
            )
            return lat
        return None

    def _lookup_comm_for_composite(
        self, op_invoke_info: "OpInvokeInfo", kernel_type: str
    ) -> Optional[float]:
        """Look up comm sub-kernel latency for composite ops (e.g., MC2).

        Computes message_bytes from the matmul output shape:
          output = (mat1.shape[0], mat2.shape[-1])
          message_bytes = output_elements * element_size

        Args layout for matmul composites:
          args[0]: mat1, args[1]: mat2, args[-1]: rank_group
        """
        args = op_invoke_info.args
        rank_group = args[-1]
        if not isinstance(rank_group, (list, tuple)):
            return None
        num_devices = len(rank_group)

        mat1 = args[0]
        mat2 = args[1]
        if not isinstance(mat1, torch.Tensor) or not isinstance(mat2, torch.Tensor):
            return None
        # Determine output element size for message_bytes calculation.
        # Quant MC2 ops (INT8/FP8/MXFP4 inputs) always accumulate and
        # all_reduce in BF16. Non-quant MC2 (BF16 inputs) keeps the same dtype.
        input_dtype = mat1.dtype
        if input_dtype in (
            torch.int8,
            torch.uint8,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ):
            output_elem_size = 2  # BF16
        else:
            output_elem_size = mat1.element_size()
        message_bytes = mat1.shape[0] * mat2.shape[-1] * output_elem_size

        topology_tier = self._resolve_topology_tier(list(rank_group))

        result = self._query_comm_csv(
            kernel_type, message_bytes, num_devices, topology_tier
        )
        if result is None:
            return None
        return result[0]  # latency only, caller doesn't need is_interpolated

    # ---- Communication op lookup (design doc §4.7) ----

    def _resolve_topology_tier(self, group: list) -> Optional[int]:
        """Resolve topology_tier from group using CommGrid.

        Returns topology_tier or None if comm_grid is not set.
        """
        if self.comm_grid is None:
            return None
        try:
            return get_topology_tier(self.comm_grid, group)
        except ValueError:
            logger.debug("Could not resolve topology_tier for group %s", group)
            return None

    def _lookup_comm(
        self, op_invoke_info: "OpInvokeInfo", mapping: dict
    ) -> Optional[QueryResult]:
        """Look up communication op latency by message_bytes + num_devices + topology_tier.

        All TC comm ops have rank_group as the last arg:
          all_reduce(x, rank, rank_group)
          all_gather(x, dim, rank, rank_group)
          reduce_scatter(x, dim, rank, rank_group)
          all_to_all(x, out_splits, in_splits, rank, rank_group)

        Args are expected as (tensor, ..., rank, rank_group) where rank is
        second-to-last and rank_group (list of device ranks) is always last.
        topology_tier is resolved from rank + rank_group via CommGrid when
        comm_grid is set; otherwise the CSV is queried without tier filtering.
        """
        kernel_type = mapping.get("kernel_type")
        if not kernel_type:
            self.last_miss_reason = "unmapped"
            return None

        # Extract the first tensor arg for message_bytes
        tensor = op_invoke_info.args[0]
        if not isinstance(tensor, torch.Tensor):
            self.last_miss_reason = "invalid_args"
            return None
        message_bytes = tensor.nelement() * tensor.element_size()

        # Extract rank (second-to-last) and rank_group (last)
        rank_group = op_invoke_info.args[-1]
        rank = op_invoke_info.args[-2]
        if not isinstance(rank_group, (list, tuple)):
            self.last_miss_reason = "invalid_args"
            return None
        num_devices = len(rank_group)

        # reduce_scatter: TC args[0] is the full input tensor (sendBuf), but
        # bench CSV message_bytes follows HCCL API convention where recvCount
        # is the per-rank output size.  Divide by num_devices to align.
        func_str = _normalize_func_name(op_invoke_info.func)
        if func_str == "tensor_cast.reduce_scatter.default" and num_devices > 1:
            message_bytes = message_bytes // num_devices

        # Resolve topology_tier from group via CommGrid
        topology_tier = self._resolve_topology_tier(list(rank_group))

        result = self._query_comm_csv(
            kernel_type, message_bytes, num_devices, topology_tier
        )
        if result is None:
            return None

        latency, is_interpolated = result
        source = QuerySource.INTERPOLATED if is_interpolated else QuerySource.MEASURED
        logger.debug(
            "HIT (comm%s) %s: message_bytes=%d, num_devices=%d, topology_tier=%s -> %.2f us",
            " interpolated" if is_interpolated else "",
            kernel_type,
            message_bytes,
            num_devices,
            topology_tier,
            latency,
        )
        return QueryResult(
            latency_us=latency,
            confidence=0.8 if is_interpolated else 0.9,
            source=source,
            details={"kernel_type": kernel_type, "topology_tier": topology_tier},
        )

    # ---- Attention special lookup (design doc §4.8) ----

    def _lookup_attention(
        self, op_invoke_info: "OpInvokeInfo", mapping: dict
    ) -> Optional[QueryResult]:
        """Query FIA enriched CSV: slot 0 Q shape + avg_seq_len column.

        Matching: (N, D, dtype, sparse_mode, num_kv_heads) exact
                  + avg_seq_len exact.
        Q shape normalized to 3D via _normalize_fia_q_shape().
        Returns None if avg_seq_len column is missing or -1.
        sparse_mode / num_kv_heads are optional (backward compat with old CSV).
        """
        kernel_type = mapping.get("kernel_type")
        if not kernel_type:
            self.last_miss_reason = "unmapped"
            return None

        df = self._load_csv(kernel_type)
        if df is None:
            self.last_miss_reason = "csv_not_found"
            return None

        # Require enriched CSV with avg_seq_len column
        # Unified column naming: prefer "Runtime avg_seq_len" (TCX enriched format),
        # fall back to "avg_seq_len" for backward compatibility
        avg_seq_col = None
        if "Runtime avg_seq_len" in df.columns:
            avg_seq_col = "Runtime avg_seq_len"
        elif "avg_seq_len" in df.columns:
            avg_seq_col = "avg_seq_len"
        else:
            self.last_miss_reason = "csv_missing_avg_seq_len"
            return None

        # Require raw profiling format (Input Shapes column)
        if "Input Shapes" not in df.columns:
            self.last_miss_reason = "csv_format_not_enriched"
            return None

        args = op_invoke_info.args
        if len(args) < 7:
            self.last_miss_reason = "insufficient_args"
            return None

        query = args[0]
        key = args[1]
        seq_lens = args[6] if len(args) > 6 else None
        query_lens = args[7] if len(args) > 7 else None

        if not isinstance(query, torch.Tensor):
            self.last_miss_reason = "query_not_tensor"
            return None

        tc_dtype_str = DTYPE_MAP.get(query.dtype)
        if tc_dtype_str is None:
            self.last_miss_reason = "dtype_unmapped"
            return None

        # Get head_dim from key tensor
        head_dim = key.shape[-1] if isinstance(key, torch.Tensor) and key.ndim >= 1 else 0

        # Normalize TC query to 3D
        tc_q_3d = _normalize_fia_q_shape(tuple(query.shape), head_dim)
        if tc_q_3d is None:
            self.last_miss_reason = "q_shape_normalize_failed"
            return None
        tc_N, tc_D = tc_q_3d[1], tc_q_3d[2]

        # Compute avg_seq_len from seq_lens
        if seq_lens is not None and isinstance(seq_lens, torch.Tensor):
            try:
                tc_avg_seq_len = int(seq_lens.float().mean().item())
            except Exception:
                self.last_miss_reason = "invalid_seq_lens"
                return None
        else:
            self.last_miss_reason = "missing_seq_lens"
            return None

        # Infer sparse_mode from query_lens (TC does not pass it explicitly)
        tc_sparse_mode = _infer_sparse_mode(query_lens)

        # Extract num_kv_heads from key tensor: shape[-2] is kv_head_num
        tc_num_kv_heads = (
            key.shape[-2]
            if isinstance(key, torch.Tensor) and key.ndim >= 2
            else None
        )

        # Check if CSV has optional Runtime columns
        has_sparse_mode_col = "Runtime sparse_mode" in df.columns
        has_kv_heads_col = "Runtime num_key_value_heads" in df.columns

        latency_col = self._latency_col(df)

        for _, row in df.iterrows():
            # Skip profiling-only rows (avg_seq_len == -1)
            csv_avg_seq = int(row[avg_seq_col])
            if csv_avg_seq < 0:
                continue

            # Parse and normalize CSV Q shape from slot 0
            shapes_str = str(row.get("Input Shapes", "")).strip('"')
            csv_q_raw = _parse_fia_q_shape(shapes_str)
            if csv_q_raw is None:
                continue
            csv_q_3d = _normalize_fia_q_shape(csv_q_raw, head_dim)
            if csv_q_3d is None:
                continue

            csv_N, csv_D = csv_q_3d[1], csv_q_3d[2]

            # Check dtype
            csv_dtypes_str = str(row.get("Input Data Types", ""))
            csv_first_dtype = csv_dtypes_str.split(";")[0].strip() if csv_dtypes_str else ""
            if tc_dtype_str != csv_first_dtype:
                continue

            # (N, D) exact match
            if tc_N != csv_N or tc_D != csv_D:
                continue

            # sparse_mode exact match (skip if CSV lacks column)
            if has_sparse_mode_col:
                csv_sparse = int(row["Runtime sparse_mode"])
                if tc_sparse_mode != csv_sparse:
                    continue

            # num_kv_heads exact match (skip if CSV lacks column)
            if has_kv_heads_col and tc_num_kv_heads is not None:
                csv_kv_heads = int(row["Runtime num_key_value_heads"])
                if tc_num_kv_heads != csv_kv_heads:
                    continue

            # avg_seq_len exact match
            if tc_avg_seq_len != csv_avg_seq:
                continue

            # First dim (T): block-padding tolerance
            tc_T = tc_q_3d[0]
            csv_T = csv_q_3d[0]
            if tc_T != csv_T:
                if not _is_block_padded(tc_T, csv_T) and not _is_block_padded(csv_T, tc_T):
                    continue

            lat = float(row[latency_col])
            logger.debug(
                "HIT (attention enriched) %s: Q=%s, avg_seq=%d, "
                "sparse=%d, kv_heads=%s -> %.1f us",
                kernel_type,
                tc_q_3d,
                tc_avg_seq_len,
                tc_sparse_mode,
                tc_num_kv_heads,
                lat,
            )
            return QueryResult(
                latency_us=lat,
                confidence=0.9,
                source=QuerySource.MEASURED,
                details={
                    "kernel_type": kernel_type,
                    "avg_seq_len": tc_avg_seq_len,
                    "sparse_mode": tc_sparse_mode,
                    "num_kv_heads": tc_num_kv_heads,
                },
            )

        self.last_miss_reason = "shape_mismatch"
        return None

    # ---- Elementwise op lookup (output-shape matching) ----

    def _lookup_elementwise(
        self, op_invoke_info: "OpInvokeInfo", mapping: dict
    ) -> Optional[QueryResult]:
        """Look up elementwise op latency by matching output shape.

        Elementwise ops (mul, add, etc.) are bandwidth-bound and their cost
        scales with output size. Instead of matching input shapes (which may
        involve broadcast), we match on the output tensor shape and dtype
        against the CSV's "Output Shapes" / "Output Data Types" columns.

        When the output dtype differs from CSV, latency is scaled by the
        byte-size ratio (bandwidth-bound assumption).

        Falls back to _lookup_compute when output is unavailable.
        """
        # Guard: if output is unavailable, fall back to input-shape matching
        out = op_invoke_info.out
        if out is None:
            return self._lookup_compute(op_invoke_info, mapping)

        # Unwrap tuple outputs (aten ops may return multiple tensors)
        if isinstance(out, (list, tuple)):
            out = out[0]

        # Guard: scalar or empty output -> fall back
        if not isinstance(out, torch.Tensor) or out.ndim == 0 or len(out.shape) < 1:
            return self._lookup_compute(op_invoke_info, mapping)

        tc_output_shape = _strip_batch_dim(tuple(out.shape))
        tc_dtype = out.dtype
        tc_dtype_str = DTYPE_MAP.get(tc_dtype)

        kernel_type = mapping.get("kernel_type")
        if not kernel_type:
            self.last_miss_reason = "unmapped"
            return None

        df = self._load_csv(kernel_type)
        if df is None:
            self.last_miss_reason = "csv_not_found"
            return None

        lat_col = self._latency_col(df)

        for _, row in df.iterrows():
            csv_out_shapes = _parse_shape_str(str(row.get("Output Shapes", "")))
            csv_out_dtypes = _parse_str_list(str(row.get("Output Data Types", "")))
            if not csv_out_shapes:
                continue

            # Match on first output shape (primary output)
            csv_shape = csv_out_shapes[0]
            csv_shape_stripped = _strip_batch_dim(csv_shape)

            shape_matched = (
                tc_output_shape == csv_shape
                or tc_output_shape == csv_shape_stripped
                or self._shapes_match_with_padding(tc_output_shape, csv_shape)
                or self._shapes_match_with_padding(tc_output_shape, csv_shape_stripped)
            )
            # 3D→2D flatten: TC keeps batch dim (B,S,D) but profiling
            # flattens to (B*S,D).  Try merging first two dims.
            if not shape_matched and len(tc_output_shape) == 3 and len(csv_shape) == 2:
                flat = (tc_output_shape[0] * tc_output_shape[1], tc_output_shape[2])
                shape_matched = (
                    flat == csv_shape
                    or flat == csv_shape_stripped
                    or self._shapes_match_with_padding(flat, csv_shape)
                    or self._shapes_match_with_padding(flat, csv_shape_stripped)
                )
            if not shape_matched:
                continue

            # Shape matched — check dtype
            csv_dtype_str = csv_out_dtypes[0] if csv_out_dtypes else None
            latency = float(row[lat_col])

            if tc_dtype_str and csv_dtype_str and tc_dtype_str == csv_dtype_str:
                # Exact dtype match
                logger.debug(
                    "HIT (elementwise) %s: output=%s dtype=%s -> %.2f us",
                    kernel_type,
                    tc_output_shape,
                    tc_dtype_str,
                    latency,
                )
                return QueryResult(
                    latency_us=latency,
                    confidence=1.0,
                    source=QuerySource.MEASURED,
                    details={"kernel_type": kernel_type, "query_mode": "elementwise"},
                )

            # Dtype differs — scale by byte ratio (bandwidth-bound)
            tc_bytes = _dtype_byte_size(tc_dtype_str) if tc_dtype_str else 0
            csv_bytes = _dtype_byte_size(csv_dtype_str) if csv_dtype_str else 0
            if tc_bytes > 0 and csv_bytes > 0:
                scale = tc_bytes / csv_bytes
                scaled_latency = latency * scale
                logger.debug(
                    "HIT (elementwise, dtype-scaled) %s: output=%s "
                    "tc_dtype=%s csv_dtype=%s scale=%.2f -> %.2f us",
                    kernel_type,
                    tc_output_shape,
                    tc_dtype_str,
                    csv_dtype_str,
                    scale,
                    scaled_latency,
                )
                return QueryResult(
                    latency_us=scaled_latency,
                    confidence=0.9,
                    source=QuerySource.MEASURED,
                    details={
                        "kernel_type": kernel_type,
                        "query_mode": "elementwise",
                        "dtype_scale": scale,
                    },
                )

        # No match found
        self.last_miss_reason = "elementwise_output_shape_mismatch"
        logger.debug(
            "MISS (elementwise) %s: output=%s dtype=%s",
            kernel_type,
            tc_output_shape,
            tc_dtype_str,
        )
        return None

    # ---- Compute op lookup (design doc S4.2 _lookup_compute) ----

    def _lookup_compute(
        self, op_invoke_info: "OpInvokeInfo", mapping: dict
    ) -> Optional[QueryResult]:
        # Build list of kernel_types to try: primary + alternates
        kernel_types = [mapping["kernel_type"]]
        for alt in mapping.get("alternate_kernel_types", []):
            if alt not in kernel_types:
                kernel_types.append(alt)

        # Extract tensor shapes and dtypes from OpInvokeInfo.args
        tc_inputs = self._extract_tensor_inputs(op_invoke_info)

        # tc_input_count: only compare the first N TC inputs (MoE ops have
        # extra NPU-internal parameters in profiling CSVs)
        tc_input_count = mapping.get("tc_input_count")
        if tc_input_count is not None:
            tc_inputs = tc_inputs[:tc_input_count]

        # csv_file: decouple CSV filename from kernel_type (e.g., MoE ops
        # where kernel_type != CSV filename)
        csv_file = mapping.get("csv_file")

        # Try each kernel_type until one matches
        for kernel_type in kernel_types:
            # csv_file override only applies to the primary kernel_type.
            # alternate_kernel_types always use their own name as CSV filename.
            # This is sufficient for current MoE ops; if a future alternate
            # needs a different CSV name, extend csv_file to a per-kernel dict.
            load_name = (
                csv_file if csv_file and kernel_type == kernel_types[0] else kernel_type
            )
            df = self._load_csv(load_name)
            if df is None:
                continue

            for _, row in df.iterrows():
                if self._inputs_match(
                    tc_inputs,
                    row,
                    kernel_type=kernel_type,
                    tc_input_count=tc_input_count,
                ):
                    _lat_col = self._latency_col(df)
                    logger.debug(
                        "HIT %s: tc_shapes=%s -> %s (%.1f us)",
                        kernel_type,
                        [s for s, _ in tc_inputs],
                        row.get("Input Shapes", ""),
                        float(row[_lat_col]),
                    )
                    return QueryResult(
                        latency_us=float(row[_lat_col]),
                        confidence=1.0,
                        source=QuerySource.MEASURED,
                        details={"kernel_type": kernel_type},
                    )

        # Log miss with shape details for debugging
        primary_kernel = kernel_types[0]
        load_name = csv_file if csv_file else primary_kernel
        df = self._load_csv(load_name)
        csv_shapes_list = []
        if df is not None:
            for _, row in df.iterrows():
                csv_shapes_list.append(str(row.get("Input Shapes", "")))
        # Determine miss reason: input count mismatch vs shape mismatch
        # When tc_input_count is set, truncate CSV count too for fair comparison
        if df is not None and len(df) > 0:
            csv_first_shapes = _parse_shape_str(str(df.iloc[0].get("Input Shapes", "")))
            effective_csv_count = len(csv_first_shapes)
            effective_tc_count = len(tc_inputs)
            if tc_input_count is not None:
                effective_csv_count = min(effective_csv_count, tc_input_count)
                effective_tc_count = min(effective_tc_count, tc_input_count)
            if effective_tc_count != effective_csv_count:
                self.last_miss_reason = "input_count_mismatch"
            else:
                self.last_miss_reason = "shape_mismatch"
        else:
            self.last_miss_reason = "csv_not_found"
        logger.debug(
            "MISS %s: tc_shapes=%s, csv_shapes=%s",
            primary_kernel,
            [s for s, _ in tc_inputs],
            csv_shapes_list,
        )
        return None

    def _extract_tensor_inputs(
        self, op_invoke_info: "OpInvokeInfo"
    ) -> List[Tuple[Tuple[int, ...], torch.dtype]]:
        """Extract (shape, dtype) for each non-scalar tensor arg.

        Scalar tensors (ndim=0, shape=()) are filtered out because profiling
        CSVs never include scalar inputs in their shape strings.
        """
        inputs = []
        for arg in op_invoke_info.args:
            if isinstance(arg, torch.Tensor) and arg.ndim > 0:
                inputs.append((tuple(arg.shape), arg.dtype))
            elif isinstance(arg, (list, tuple)):
                for item in arg:
                    if isinstance(item, torch.Tensor) and item.ndim > 0:
                        inputs.append((tuple(item.shape), item.dtype))
        return inputs

    def _inputs_match(
        self,
        tc_inputs: List[Tuple[Tuple[int, ...], torch.dtype]],
        csv_row: pd.Series,
        kernel_type: str = "",
        tc_input_count: Optional[int] = None,
    ) -> bool:
        """Match TensorCast input shapes/dtypes against a CSV row.

        Handles:
        - FRACTAL_NZ restoration (design doc S4.9)
        - ND weight transpose for matmul kernels (CSV stores (N,K), TC sees (K,N))
        - Block-padding tolerance (TC pads seq to NPU tile alignment)
        """
        csv_shapes = _parse_shape_str(str(csv_row.get("Input Shapes", "")))
        csv_dtypes = _parse_str_list(str(csv_row.get("Input Data Types", "")))
        csv_formats = _parse_str_list(str(csv_row.get("Input Formats", "")))

        # Truncate CSV shapes/dtypes/formats when tc_input_count is set.
        # NPU profiling CSVs may include internal parameters beyond what TC passes;
        # tc_input_count tells us to only compare the first N inputs.
        # NOTE: when tc_input_count is set in a composite mapping, both tc_inputs
        # (truncated above in _lookup_composite) and csv_shapes (truncated here)
        # are shortened — this double truncation is intentional: tc_inputs is
        # pre-filtered to the relevant tensors, csv_shapes is trimmed to match.
        if tc_input_count is not None:
            csv_shapes = csv_shapes[:tc_input_count]
            csv_dtypes = csv_dtypes[:tc_input_count]
            csv_formats = csv_formats[:tc_input_count]

        # RoPE input normalization: swap Q↔K, transpose (B,H,S,D)→(B,S,H,D).
        # Works with both full (4 inputs) and tc_input_count-truncated (2 inputs).
        tc_inputs_normalized = tc_inputs
        if (
            kernel_type in _ROPE_KERNELS
            and len(tc_inputs) >= 2
            and len(csv_shapes) >= 2
        ):
            tc_inputs_normalized = _normalize_rope_inputs(tc_inputs)

        # SwiGlu input normalization: TC sends 2 inputs (gate, up),
        # profiling CSV has 1 fused input concatenated along last dim.
        if (
            kernel_type in _SWIGLU_KERNELS
            and len(tc_inputs) == 2
            and len(csv_shapes) == 1
        ):
            s1, dtype1 = tc_inputs[0]
            s2, dtype2 = tc_inputs[1]
            s1 = _strip_batch_dim(s1)
            s2 = _strip_batch_dim(s2)
            if len(s1) == len(s2) and s1[:-1] == s2[:-1] and dtype1 == dtype2:
                merged_shape = s1[:-1] + (s1[-1] + s2[-1],)
                tc_inputs_normalized = [(merged_shape, dtype1)]

        if len(tc_inputs_normalized) != len(csv_shapes):
            return False

        for i, (tc_shape, tc_dtype) in enumerate(tc_inputs_normalized):
            # Check dtype
            expected_dtype = DTYPE_MAP.get(tc_dtype)
            if expected_dtype is None or i >= len(csv_dtypes):
                return False
            csv_dtype_i = csv_dtypes[i]
            if expected_dtype != csv_dtype_i:
                # Relaxed dtype matching for specific kernels (e.g. RoPE:
                # NPU records K as FLOAT while TC dispatches BF16)
                if kernel_type in _DTYPE_RELAXED_KERNELS:
                    compat_expected = _DTYPE_COMPAT.get(expected_dtype)
                    compat_csv = _DTYPE_COMPAT.get(csv_dtype_i)
                    if (
                        compat_expected is None
                        or compat_csv is None
                        or compat_expected != compat_csv
                    ):
                        return False
                else:
                    return False

            # Get CSV shape, restore FRACTAL_NZ if needed
            csv_shape = csv_shapes[i]
            fmt = csv_formats[i] if i < len(csv_formats) else "ND"
            if fmt == "FRACTAL_NZ":
                csv_shape = fractal_nz_to_nd(csv_shape)

            if tc_shape == csv_shape:
                continue

            # Strip leading batch dim=1: TC keeps (1, seq, dim), profiling has (seq, dim)
            tc_shape_stripped = _strip_batch_dim(tc_shape)
            csv_shape_stripped = _strip_batch_dim(csv_shape)
            if tc_shape_stripped == csv_shape_stripped:
                continue
            if tc_shape_stripped == csv_shape:
                continue

            # Weight transpose for matmul: CSV stores (N,K), TC sees (K,N)
            # Applies to both ND format and FRACTAL_NZ-restored shapes.
            # FRACTAL_NZ → ND gives (N,K) via fractal_nz_to_nd(); TC has (K,N).
            if (
                kernel_type in _MATMUL_KERNELS
                and i >= 1
                and len(tc_shape_stripped) == 2
                and len(csv_shape) == 2
                and tc_shape_stripped == (csv_shape[1], csv_shape[0])
            ):
                continue

            # Block-padding tolerance: TC pads to NPU tile alignment
            if self._shapes_match_with_padding(tc_shape_stripped, csv_shape):
                continue
            # Also try with both batch dims stripped
            if self._shapes_match_with_padding(tc_shape_stripped, csv_shape_stripped):
                continue

            # 3D→2D flatten for quantize/norm kernels
            if kernel_type in _FLATTEN_BATCH_KERNELS and len(csv_shape) == 2:
                # Use original tc_shape (pre-strip) for 3D checks, since
                # _strip_batch_dim may collapse (1,H,D) → (H,D) losing the
                # 3D structure needed for flatten/merge.
                shape_3d = (
                    tc_shape_stripped
                    if len(tc_shape_stripped) == 3
                    else tc_shape if len(tc_shape) == 3 else None
                )
                if shape_3d is not None:
                    # Flatten first two dims: TC (B, M, D) → CSV (B*M, D)
                    flattened = (
                        shape_3d[0] * shape_3d[1],
                        shape_3d[2],
                    )
                    if flattened == csv_shape:
                        continue
                    if self._shapes_match_with_padding(flattened, csv_shape):
                        continue

                    # Merge last two dims: TC (T, H, D) → CSV (T, H*D)
                    # Only for MLA quantize kernels where NPU reshapes
                    # per-head to hidden_dim before quantize.
                    if kernel_type in _MERGE_LAST_DIMS_KERNELS:
                        merged = (
                            shape_3d[0],
                            shape_3d[1] * shape_3d[2],
                        )
                        if merged == csv_shape:
                            continue
                        if self._shapes_match_with_padding(merged, csv_shape):
                            continue

            return False

        return True

    @staticmethod
    def _shapes_match_with_padding(
        tc_shape: Tuple[int, ...], csv_shape: Tuple[int, ...]
    ) -> bool:
        """Check if shapes match allowing block-padding on any dimension."""
        if len(tc_shape) != len(csv_shape):
            return False
        has_padding = False
        for tc_dim, csv_dim in zip(tc_shape, csv_shape):
            if tc_dim == csv_dim:
                continue
            if _is_block_padded(tc_dim, csv_dim):
                has_padding = True
                continue
            return False
        return has_padding
