"""Operator mapping between VLLM and TensorCast.

This module provides mappings between fused VLLM/Ascend CANN kernels and their
corresponding TensorCast decomposed operations. It supports both direct mapping
and fusion-aware mapping where a single VLLM kernel maps to multiple TC ops.

The module supports two modes:
1. Hardcoded mappings (legacy, for backward compatibility)
2. YAML-based mappings (new, for extensibility)

To use YAML mappings, call FusionAwareMapper.from_yaml("mapping_name") or
FusionAwareMapper.from_yaml_path(path).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class FusionMapping:
    """Defines how a fused VLLM op maps to multiple TensorCast ops.

    Attributes:
        vllm_op: The fused VLLM/CANN kernel name
        tc_ops: List of TensorCast ops that together implement this fusion
        aggregation_method: How to combine TC op times ("sum" or "max")
        description: Human-readable description of the fusion
    """

    vllm_op: str
    tc_ops: List[str]
    aggregation_method: str = "sum"  # "sum" for sequential, "max" for parallel
    description: str = ""


# Fusion-aware mappings for single-step comparison
# These define how fused CANN kernels decompose into TensorCast operations
FUSION_MAPPINGS: List[FusionMapping] = [
    # MoE Expert Computation
    # GroupedMatmul = batched expert MLPs, each expert is a static_quant_linear
    FusionMapping(
        vllm_op="GroupedMatmul",
        tc_ops=["tensor_cast.grouped_matmul_quant", "tensor_cast.static_quant_linear"],
        aggregation_method="sum",
        description="MoE expert MLPs - batched linear projections",
    ),
    # Attention
    # FusedInferAttentionScore = MLA attention + RoPE embedding
    FusionMapping(
        vllm_op="FusedInferAttentionScore",
        tc_ops=["tensor_cast.multihead_latent_attention"],
        aggregation_method="sum",
        description="Fused MLA attention with RoPE",
    ),
    # Linear Projections
    # QuantBatchMatmulV3 = quantized matmul (W8A8)
    FusionMapping(
        vllm_op="QuantBatchMatmulV3",
        tc_ops=["tensor_cast.static_quant_linear", "aten.mm"],
        aggregation_method="sum",
        description="Quantized linear projection (W8A8)",
    ),
    # MoE Token Routing
    FusionMapping(
        vllm_op="MoeDistributeDispatchV2",
        tc_ops=["tensor_cast.permute_tokens", "tensor_cast.all_to_all"],
        aggregation_method="sum",
        description="MoE token dispatch with all-to-all",
    ),
    FusionMapping(
        vllm_op="MoeDistributeCombineV2",
        tc_ops=["tensor_cast.unpermute_tokens", "tensor_cast.all_to_all"],
        aggregation_method="sum",
        description="MoE token combine with all-to-all",
    ),
    # Fused Activation + Quantization
    # DequantSwigluQuant = dequant -> silu -> mul -> quant
    FusionMapping(
        vllm_op="DequantSwigluQuant",
        tc_ops=[
            "tensor_cast.dequantize",
            "aten.silu",
            "aten.mul",
            "tensor_cast.quantize",
        ],
        aggregation_method="sum",
        description="Fused dequant-swiglu-quant activation",
    ),
    # Fused Add + RmsNorm
    FusionMapping(
        vllm_op="InplaceAddRmsNorm",
        tc_ops=["aten.add", "tensor_cast.rmsnorm"],
        aggregation_method="sum",
        description="Fused residual add and RMS normalization",
    ),
    # RmsNorm standalone
    FusionMapping(
        vllm_op="RmsNorm",
        tc_ops=["tensor_cast.rmsnorm"],
        aggregation_method="sum",
        description="RMS normalization",
    ),
    # Quantization ops
    FusionMapping(
        vllm_op="AscendQuantV2",
        tc_ops=["tensor_cast.quantize", "tensor_cast.dynamic_quantize_symmetric"],
        aggregation_method="sum",
        description="Dynamic quantization",
    ),
    FusionMapping(
        vllm_op="DynamicQuant",
        tc_ops=["tensor_cast.quantize", "tensor_cast.dynamic_quantize_symmetric"],
        aggregation_method="sum",
        description="Dynamic quantization",
    ),
    FusionMapping(
        vllm_op="AscendDequantV2",
        tc_ops=["tensor_cast.dequantize"],
        aggregation_method="sum",
        description="Dequantization",
    ),
]


# Direct mapping from VLLM/Ascend operator names to TensorCast operator names
# Note: TensorCast ops include ".default" suffix in simulation output
VLLM_TO_TENSORCAST_MAPPING: Dict[str, List[str]] = {
    # MoE operations
    # GroupedMatmul in VLLM = expert MLPs, in TensorCast simulated as static_quant_linear
    "GroupedMatmul": [
        "tensor_cast.grouped_matmul_quant",
        "tensor_cast.static_quant_linear",
    ],
    "MoeDistributeDispatchV2": ["tensor_cast.permute_tokens"],
    "MoeDistributeCombineV2": ["tensor_cast.unpermute_tokens"],
    "MoeGatingTopK": ["aten.topk"],
    # Attention operations
    "FusedInferAttentionScore": ["tensor_cast.multihead_latent_attention"],
    "KvRmsNormRopeCache": [
        "tensor_cast.reshape_and_cache",
        "tensor_cast.concat_and_cache_mla",
    ],
    "PagedAttention": ["tensor_cast.attention"],
    "InterleaveRope": ["aten.clone", "aten.neg", "aten.mul"],  # RoPE operations
    # Quantization operations
    "QuantBatchMatmulV3": ["tensor_cast.static_quant_linear", "aten.mm"],
    "AscendQuantV2": ["tensor_cast.quantize", "tensor_cast.dynamic_quantize_symmetric"],
    "DynamicQuant": ["tensor_cast.quantize", "tensor_cast.dynamic_quantize_symmetric"],
    "AscendDequantV2": ["tensor_cast.dequantize"],
    "DequantSwigluQuant": ["tensor_cast.dequantize", "aten.silu", "aten.mul"],
    # Normalization operations
    "InplaceAddRmsNorm": [
        "aten.add",
        "tensor_cast.rmsnorm",
        "aten.pow",
        "aten.mean",
        "aten.rsqrt",
        "aten.mul",
    ],
    "RmsNorm": [
        "tensor_cast.rmsnorm",
        "aten.pow",
        "aten.mean",
        "aten.rsqrt",
        "aten.mul",
    ],
    "LayerNorm": ["aten.layer_norm"],
    # Communication operations
    "allgatherAicpuKernel": ["tensor_cast.all_gather"],
    "AllGather": ["tensor_cast.all_gather"],
    "AllReduce": ["tensor_cast.all_reduce"],
    "AllToAll": ["tensor_cast.all_to_all"],
    # Basic compute operations
    "TransposeBatchMatMul": ["aten.bmm"],
    "MatMul": ["aten.mm", "aten.matmul"],
    "MatMulV2": ["aten.mm", "aten.matmul"],
    "BatchMatMul": ["aten.bmm"],
    "Add": ["aten.add"],
    "Mul": ["aten.mul"],
    "Cast": ["aten._to_copy"],
    "Transpose": ["aten.transpose", "aten.permute"],
    "Reshape": ["aten.reshape", "aten.view"],
    "ConcatD": ["aten.cat"],
    "Concat": ["aten.cat"],
    "SplitVD": ["aten.split", "aten.split_with_sizes"],
    "Slice": ["aten.slice"],
    "Gather": ["aten.gather", "aten.index_select"],
    "GatherV2": ["aten.gather", "aten.index_select"],
    "Silu": ["aten.silu"],
    "Gelu": ["aten.gelu"],
    "Sigmoid": ["aten.sigmoid"],
    "TensorMove": ["aten.copy_", "aten.clone"],
    "Index": ["aten.index_select", "aten.gather"],
    "ArgMaxV2": ["aten.argmax"],
    "ReduceMax": ["aten.max"],
    "MaskedFill": ["aten.masked_fill"],
    "Fill": ["aten.fill_", "aten.zeros_like"],
    "ZerosLike": ["aten.zeros_like"],
    "BroadcastTo": ["aten.expand", "aten.broadcast_to"],
    "GreaterEqual": ["aten.ge"],
    "Equal": ["aten.eq"],
    "Less": ["aten.lt"],
    "Range": ["aten.arange"],
    "PadV3": ["aten.pad"],
    # KV Cache operations
    "ScatterPaKvCache": [
        "tensor_cast.reshape_and_cache",
        "tensor_cast.concat_and_cache_mla",
    ],
    "PagedCacheLoadNdKernel": ["aten.index_select"],
    # Buffer/Fusion operations
    "AutomaticBufferFusionOp": ["aten.copy_", "aten.add", "aten.mul"],
}

# Category mapping for unmatched operations
OPERATION_CATEGORIES = {
    "compute": ["MatMul", "BatchMatMul", "GroupedMatmul", "Conv", "Gemm"],
    "memory": ["Cast", "Transpose", "Reshape", "Concat", "Slice", "Gather"],
    "communication": ["AllGather", "AllReduce", "AllToAll", "allgatherAicpuKernel"],
    "attention": ["FusedInferAttentionScore", "PagedAttention"],
    "quantization": ["Quant", "Dequant", "AscendQuant", "AscendDequant"],
    "normalization": ["RmsNorm", "LayerNorm", "BatchNorm"],
    "activation": ["Silu", "Gelu", "Relu", "Swiglu"],
    "moe": ["Moe", "GroupedMatmul", "Expert"],
}


class FusionAwareMapper:
    """Maps fused VLLM ops to multiple TensorCast ops with time aggregation.

    This mapper understands that a single fused CANN kernel in VLLM profiling
    may correspond to multiple decomposed operations in TensorCast simulation.

    Supports both hardcoded mappings (default) and YAML-based mappings.
    """

    def __init__(
        self,
        fusion_mappings: Optional[List[FusionMapping]] = None,
        direct_mappings: Optional[Dict[str, List[str]]] = None,
    ):
        """Initialize mapper with optional custom mappings.

        Args:
            fusion_mappings: Custom fusion mappings (default: FUSION_MAPPINGS)
            direct_mappings: Custom direct mappings (default: VLLM_TO_TENSORCAST_MAPPING)
        """
        if fusion_mappings is None:
            fusion_mappings = FUSION_MAPPINGS
        if direct_mappings is None:
            direct_mappings = VLLM_TO_TENSORCAST_MAPPING

        self._direct_mappings = direct_mappings

        # Build lookup from VLLM op to fusion mapping
        self._fusion_map: Dict[str, FusionMapping] = {}
        for fm in fusion_mappings:
            self._fusion_map[fm.vllm_op] = fm

    @classmethod
    def from_yaml(cls, mapping_name: str = "default") -> "FusionAwareMapper":
        """Create mapper from YAML mapping file.

        Args:
            mapping_name: Name of the mapping file (e.g., "default", "qwen3")

        Returns:
            FusionAwareMapper instance configured from YAML
        """
        from .mapping_loader import load_mappings

        config = load_mappings(mapping_name)
        return cls._from_mapping_config(config)

    @classmethod
    def from_yaml_path(cls, yaml_path: Path) -> "FusionAwareMapper":
        """Create mapper from a specific YAML file path.

        Args:
            yaml_path: Path to the YAML mapping file

        Returns:
            FusionAwareMapper instance configured from YAML
        """
        from .mapping_loader import load_mappings_from_path

        config = load_mappings_from_path(yaml_path)
        return cls._from_mapping_config(config)

    @classmethod
    def _from_mapping_config(cls, config) -> "FusionAwareMapper":
        """Create mapper from MappingConfig.

        Args:
            config: MappingConfig instance

        Returns:
            FusionAwareMapper instance
        """
        # Convert YAML fusion mappings to FusionMapping objects
        fusion_mappings = []

        # VLLM fusions: one VLLM op -> multiple TC ops
        for yaml_fm in config.vllm_fusions:
            # Create one FusionMapping per VLLM op
            for vllm_op in yaml_fm.vllm_ops:
                fusion_mappings.append(
                    FusionMapping(
                        vllm_op=vllm_op,
                        tc_ops=yaml_fm.tc_ops,
                        aggregation_method=yaml_fm.aggregation,
                        description=yaml_fm.description,
                    )
                )

        # TC fusions: multiple VLLM ops -> one TC op
        # These are handled differently - we need to track them separately
        # For now, we store the first VLLM op as the key
        for yaml_fm in config.tc_fusions:
            if yaml_fm.vllm_ops:
                fusion_mappings.append(
                    FusionMapping(
                        vllm_op=yaml_fm.vllm_ops[0],  # Primary VLLM op
                        tc_ops=yaml_fm.tc_ops,
                        aggregation_method=yaml_fm.aggregation,
                        description=yaml_fm.description,
                    )
                )

        return cls(
            fusion_mappings=fusion_mappings,
            direct_mappings=config.direct_mappings,
        )

    def get_fusion_mapping(self, vllm_op: str) -> Optional[FusionMapping]:
        """Get fusion mapping for a VLLM operation."""
        return self._fusion_map.get(vllm_op)

    def get_tc_ops_for_vllm(self, vllm_op: str) -> List[str]:
        """Get list of TensorCast ops that implement a VLLM op."""
        fm = self._fusion_map.get(vllm_op)
        if fm:
            return fm.tc_ops

        # Fall back to direct mapping (instance or global)
        return self._direct_mappings.get(vllm_op, [])

    def aggregate_tc_times(
        self,
        vllm_op: str,
        tc_op_times: Dict[str, float],
    ) -> Tuple[float, List[str]]:
        """Aggregate TensorCast op times for a fused VLLM op.

        Args:
            vllm_op: The VLLM operation type
            tc_op_times: Dict mapping TC op names to their total times

        Returns:
            Tuple of (aggregated_time, list of matched TC ops)
        """
        fm = self._fusion_map.get(vllm_op)
        if fm:
            matched_ops = []
            times = []
            for tc_op in fm.tc_ops:
                # Try exact match first
                if tc_op in tc_op_times:
                    matched_ops.append(tc_op)
                    times.append(tc_op_times[tc_op])
                else:
                    # Try substring match
                    for tc_name, tc_time in tc_op_times.items():
                        if tc_op in tc_name or tc_name.endswith(tc_op.split(".")[-1]):
                            matched_ops.append(tc_name)
                            times.append(tc_time)
                            break

            if times:
                if fm.aggregation_method == "sum":
                    return sum(times), matched_ops
                else:  # max
                    return max(times), matched_ops

        # Fall back to direct mapping (instance or global)
        candidates = self._direct_mappings.get(vllm_op, [])
        for candidate in candidates:
            if candidate in tc_op_times:
                return tc_op_times[candidate], [candidate]
            # Substring match
            for tc_name, tc_time in tc_op_times.items():
                if candidate in tc_name:
                    return tc_time, [tc_name]

        return 0.0, []

    def match_single_step_ops(
        self,
        vllm_step_stats: Dict[str, dict],
        tc_step_stats: Dict[str, dict],
    ) -> List[Dict]:
        """Match VLLM single-step operations to TensorCast operations.

        Args:
            vllm_step_stats: Aggregated VLLM op stats from single step
            tc_step_stats: Aggregated TensorCast op stats

        Returns:
            List of match records with comparison data
        """
        # Build TC time lookup (use base names without .default)
        tc_times: Dict[str, float] = {}
        tc_counts: Dict[str, int] = {}
        for tc_name, tc_data in tc_step_stats.items():
            base_name = tc_name.replace(".default", "")
            tc_times[base_name] = tc_data.get(
                "total_time_us", tc_data.get("total_duration_us", 0)
            )
            tc_counts[base_name] = tc_data.get("count", 0)

        matches = []
        matched_tc_ops = set()

        # Match VLLM ops to TC ops
        for vllm_op, vllm_data in sorted(
            vllm_step_stats.items(),
            key=lambda x: x[1].get("total_duration_us", 0),
            reverse=True,
        ):
            vllm_time = vllm_data.get("total_duration_us", 0)
            vllm_count = vllm_data.get("count", 0)

            tc_time, tc_matched = self.aggregate_tc_times(vllm_op, tc_times)
            tc_count = sum(
                tc_counts.get(op.replace(".default", ""), 0) for op in tc_matched
            )

            for op in tc_matched:
                matched_tc_ops.add(op.replace(".default", ""))

            # Calculate difference
            if vllm_time > 0:
                diff_us = tc_time - vllm_time
                diff_pct = (diff_us / vllm_time) * 100
            else:
                diff_us = tc_time
                diff_pct = 0 if tc_time == 0 else float("inf")

            # Determine match status
            if tc_matched:
                if abs(diff_pct) < 20:
                    match_status = "exact"
                elif abs(diff_pct) < 50:
                    match_status = "partial"
                else:
                    match_status = "partial"
            else:
                match_status = "missing_in_tc"

            matches.append(
                {
                    "vllm_op": vllm_op,
                    "tc_ops": tc_matched,
                    "vllm_duration_us": vllm_time,
                    "tc_duration_us": tc_time,
                    "vllm_count": vllm_count,
                    "tc_count": tc_count,
                    "difference_us": diff_us,
                    "difference_pct": diff_pct,
                    "match_status": match_status,
                }
            )

        # Add TC-only ops (not matched to any VLLM op)
        for tc_name, tc_data in tc_step_stats.items():
            base_name = tc_name.replace(".default", "")
            if base_name not in matched_tc_ops:
                tc_time = tc_data.get(
                    "total_time_us", tc_data.get("total_duration_us", 0)
                )
                tc_count = tc_data.get("count", 0)

                matches.append(
                    {
                        "vllm_op": "[Not Matched]",
                        "tc_ops": [tc_name],
                        "vllm_duration_us": 0,
                        "tc_duration_us": tc_time,
                        "vllm_count": 0,
                        "tc_count": tc_count,
                        "difference_us": tc_time,
                        "difference_pct": float("inf") if tc_time > 0 else 0,
                        "match_status": "missing_in_vllm",
                    }
                )

        return matches


class OpMapper:
    """Maps VLLM operations to TensorCast operations."""

    def __init__(self):
        self.mapping = VLLM_TO_TENSORCAST_MAPPING
        self.categories = OPERATION_CATEGORIES

    def get_candidate_ops(self, vllm_op_type: str) -> List[str]:
        """Get candidate TensorCast operations for a VLLM operation."""
        # Direct mapping
        if vllm_op_type in self.mapping:
            return self.mapping[vllm_op_type]

        # Fuzzy matching by substring
        candidates = []
        for vllm_key, tc_ops in self.mapping.items():
            if (
                vllm_key.lower() in vllm_op_type.lower()
                or vllm_op_type.lower() in vllm_key.lower()
            ):
                candidates.extend(tc_ops)

        return candidates

    def compute_match_confidence(
        self,
        vllm_op_type: str,
        tensorcast_op_name: str,
        vllm_count: int,
        tensorcast_count: int,
    ) -> Tuple[float, str]:
        """
        Compute match confidence between VLLM and TensorCast operations.

        Returns:
            Tuple of (confidence_score, match_status)
            - confidence_score: 0.0 to 1.0
            - match_status: "exact", "partial", "fuzzy", "none"
        """
        # Check direct mapping
        candidates = self.mapping.get(vllm_op_type, [])

        if tensorcast_op_name in candidates:
            # Exact name match
            if abs(vllm_count - tensorcast_count) / max(vllm_count, 1) < 0.1:
                return 1.0, "exact"
            else:
                return 0.8, "partial"

        # Check if TensorCast op matches any substring
        for candidate in candidates:
            if candidate in tensorcast_op_name:
                return 0.7, "partial"

        # Fuzzy matching by category
        vllm_category = self._get_category(vllm_op_type)
        tc_category = self._get_category(tensorcast_op_name)

        if vllm_category and vllm_category == tc_category:
            return 0.5, "fuzzy"

        # Substring matching
        if vllm_op_type.lower() in tensorcast_op_name.lower():
            return 0.4, "fuzzy"

        return 0.0, "none"

    def _get_category(self, op_name: str) -> Optional[str]:
        """Get category for an operation."""
        op_name_lower = op_name.lower()

        for category, keywords in self.categories.items():
            for keyword in keywords:
                if keyword.lower() in op_name_lower:
                    return category

        return None

    def get_match_status(
        self, vllm_op_type: str, tensorcast_op_name: Optional[str]
    ) -> str:
        """
        Get match status label.

        Returns:
            "exact": Direct mapping with exact match
            "partial": Mapped but with differences
            "missing_in_tc": No corresponding TensorCast operation found
        """
        if tensorcast_op_name is None:
            return "missing_in_tc"

        candidates = self.mapping.get(vllm_op_type, [])

        if tensorcast_op_name in candidates:
            return "exact"

        for candidate in candidates:
            if candidate in tensorcast_op_name:
                return "partial"

        return "missing_in_tc"
