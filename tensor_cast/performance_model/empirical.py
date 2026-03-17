"""EmpiricalPerformanceModel: measurement-based performance model.

Design doc reference: §4.3
"""

import logging
from collections import Counter
from typing import Optional

import torch

from overrides import override

from ..device import DeviceProfile
from .base import PerformanceModel
from .op_invoke_info import OpInvokeInfo
from .perf_database.data_source import DataSource

logger = logging.getLogger(__name__)

# Human-readable descriptions for miss reason codes
_MISS_REASON_LABELS = {
    "unmapped": "not in op_mapping.yaml",
    "shape_mismatch": "kernel found, no matching shape in CSV",
    "input_count_mismatch": "TC input count differs from CSV",
    "csv_format_raw": "CSV has raw profiling format (needs microbenchmark)",
    "csv_not_found": "kernel CSV file missing",
    "no_sub_kernels": "composite op has no sub_kernels defined",
    "invalid_args": "op args could not be parsed",
}

# Default fused op groups — maps NPU fusion name to constituent TC op prefixes
DEFAULT_FUSED_GROUPS = {
    "DispatchFFNCombine": [
        "tensor_cast.permute_tokens",
        "tensor_cast.grouped_matmul",  # prefix covers all variants
        "tensor_cast.unpermute_tokens",
        "tensor_cast.all_to_all",
    ],
    "MLAPO": [
        "tensor_cast.mlapo",
        "tensor_cast.mlapo_quant",
    ],
    "MLA": [
        "tensor_cast.multihead_latent_attention",
    ],
    "MC2": [
        "tensor_cast.matmul_all_reduce",
        "tensor_cast.static_quant_linear_all_reduce",
        "tensor_cast.fp8_linear_all_reduce",
    ],
}


def compute_fused_op_stats(
    hit_details: list[tuple[str, str, tuple, float]],
    miss_details: list[tuple[str, str, list]],
    fused_groups: dict[str, list[str]] | None = None,
) -> dict:
    """Compute Fused Op Match Rate with pessimistic grouping.

    Phase 1 metrics (M1-M3):
    - M1 (Raw Op-Count HR): reported separately by EmpiricalPerformanceModel
    - M2 (Fused Op HR): per unique func_name, pessimistic rule, with fused grouping
    - M3 (Fused Op HR w/o zc): same as M2 excluding zero_cost ops

    Pessimistic rule: if an op appears in BOTH hits and misses (different
    shapes), it counts as MISS. An op is HIT only if ALL its invocations HIT.

    Fused grouping: DFC/MLAPO/MLA/MC2 constituent ops collapse to 1 fused op.
    A fused group is HIT only if ALL members are HIT and NONE MISS.

    Args:
        hit_details: list of (func_name, kernel_type, shape_sig, latency_s) tuples
        miss_details: list of (func_name, reason, shapes) tuples
        fused_groups: map of group_name -> list of TC op prefixes to group

    Returns:
        dict with fused_hit, fused_miss, fused_total, fused_hr,
        _no_zc variants, and per_shape stats.
    """
    if fused_groups is None:
        fused_groups = DEFAULT_FUSED_GROUPS

    # Build reverse map: tc_op_prefix -> group_name
    op_to_group: dict[str, str] = {}
    for group_name, prefixes in fused_groups.items():
        for prefix in prefixes:
            op_to_group[prefix] = group_name

    def _get_group(func_name: str) -> str | None:
        for prefix, group in op_to_group.items():
            if func_name.startswith(prefix):
                return group
        return None

    # --- Phase 1: Pessimistic per-func_name counting ---
    # Collect all unique func_names and which ones ever missed
    all_func_names: set[str] = set()
    miss_func_names: set[str] = set()
    zero_cost_funcs: set[str] = set()

    for func_name, kernel_type, _shape_sig, _latency_s in hit_details:
        all_func_names.add(func_name)
        if kernel_type == "zero_cost":
            zero_cost_funcs.add(func_name)

    for func_name, _reason, _shapes in miss_details:
        all_func_names.add(func_name)
        miss_func_names.add(func_name)

    # Pessimistic: HIT only if NEVER missed
    hit_func_names = all_func_names - miss_func_names

    # Group hits
    ungrouped_hits: set[str] = set()
    hit_groups_seen: dict[str, set[str]] = {}
    for func_name in hit_func_names:
        group = _get_group(func_name)
        if group:
            hit_groups_seen.setdefault(group, set()).add(func_name)
        else:
            ungrouped_hits.add(func_name)

    # Group misses
    miss_groups_seen: set[str] = set()
    ungrouped_misses: set[str] = set()
    for func_name in miss_func_names:
        group = _get_group(func_name)
        if group:
            miss_groups_seen.add(group)
        else:
            ungrouped_misses.add(func_name)

    # A fused group is HIT only if ALL members HIT and NONE MISS
    grouped_hits: set[str] = set()
    for group in hit_groups_seen:
        if group not in miss_groups_seen:
            grouped_hits.add(group)

    all_groups = set(hit_groups_seen.keys()) | miss_groups_seen

    fused_hit = len(ungrouped_hits) + len(grouped_hits)
    fused_miss = len(ungrouped_misses) + len(all_groups - grouped_hits)
    fused_total = fused_hit + fused_miss

    # No zero_cost view
    fused_hit_no_zc = len(ungrouped_hits - zero_cost_funcs) + len(grouped_hits)
    fused_total_no_zc = fused_total - len(zero_cost_funcs & hit_func_names)

    # Per-shape stats computed separately by compute_per_shape_stats()

    return {
        "fused_hit": fused_hit,
        "fused_miss": fused_miss,
        "fused_total": fused_total,
        "fused_hr": fused_hit / fused_total if fused_total > 0 else 0,
        "fused_hit_no_zc": fused_hit_no_zc,
        "fused_total_no_zc": fused_total_no_zc,
        "fused_hr_no_zc": (
            fused_hit_no_zc / fused_total_no_zc if fused_total_no_zc > 0 else 0
        ),
    }


def compute_per_shape_stats(
    hit_details: list[tuple[str, str, tuple, float]],
    miss_details: list[tuple[str, str, list]],
) -> dict:
    """M4: Per-Shape Match HR (unique shape variants, excl zero_cost).

    Each unique (func_name, shape_sig) pair is counted independently.
    No pessimistic rule, no fused grouping.

    Returns:
        dict with hit_shapes, total_shapes, m4, miss_shape_list.
    """
    hit_shapes: set[tuple[str, tuple]] = set()
    for func_name, kernel_type, shape_sig, _latency_s in hit_details:
        if kernel_type == "zero_cost":
            continue
        hit_shapes.add((func_name, shape_sig))

    all_shapes: set[tuple[str, tuple]] = set(hit_shapes)
    for func_name, _reason, tc_shapes in miss_details:
        shape_sig = tuple(tuple(s) for s in tc_shapes) if tc_shapes else ()
        all_shapes.add((func_name, shape_sig))

    m4 = len(hit_shapes) / len(all_shapes) if all_shapes else 0.0
    miss_shape_list = sorted(all_shapes - hit_shapes)
    return {
        "hit_shapes": len(hit_shapes),
        "total_shapes": len(all_shapes),
        "m4": m4,
        "miss_shape_list": miss_shape_list,
    }


class EmpiricalPerformanceModel(PerformanceModel):
    """Performance model based on measured data from a DataSource.

    Design doc §4.3: accepts DataSource instance, process_op() queries
    data source first, falls back to fallback_model on miss.

    Usage (design doc §5.1):
        data_source = ProfilingDataSource(data_dir, comm_grid=...)
        pm = EmpiricalPerformanceModel(device_profile, data_source)
    """

    def __init__(
        self,
        device_profile: DeviceProfile,
        data_source: DataSource,
        fallback_model: Optional[PerformanceModel] = None,
    ):
        super().__init__("empirical", device_profile)
        self.data_source = data_source
        self._fallback_model = fallback_model
        self._stats = {"hit": 0, "miss": 0}
        self._hit_details: list[tuple[str, str, tuple, float]] = []
        # Each miss: (func_name, reason, tc_shapes)
        self._miss_details: list[tuple[str, str, list[tuple]]] = []

    @property
    def fallback_model(self) -> PerformanceModel:
        if self._fallback_model is None:
            from .analytic import AnalyticPerformanceModel

            self._fallback_model = AnalyticPerformanceModel(self.device_profile)
        return self._fallback_model

    @override
    def process_op(self, op_invoke_info: OpInvokeInfo) -> PerformanceModel.Result:
        result = self.data_source.lookup(op_invoke_info)
        func_name = str(op_invoke_info.func).removeprefix("torch.ops.")
        if result is not None:
            self._stats["hit"] += 1
            kernel_type = result.details.get("kernel_type", "?")
            tc_shapes = [
                tuple(a.shape)
                for a in op_invoke_info.args
                if isinstance(a, torch.Tensor)
            ]
            shape_sig = tuple(tc_shapes)
            self._hit_details.append(
                (func_name, kernel_type, shape_sig, result.latency_us * 1e-6)
            )
            return PerformanceModel.Result(
                execution_time_s=result.latency_us * 1e-6,
                statistics={
                    "source": result.source.name,
                    "confidence": result.confidence,
                    **result.details,
                },
            )
        self._stats["miss"] += 1
        tc_shapes = [
            tuple(a.shape) for a in op_invoke_info.args if isinstance(a, torch.Tensor)
        ]
        # Read miss reason from data source (if it supports it)
        reason = getattr(self.data_source, "last_miss_reason", "unknown")
        self._miss_details.append((func_name, reason, tc_shapes))
        return self.fallback_model.process_op(op_invoke_info)

    def get_stats(self) -> dict:
        total = self._stats["hit"] + self._stats["miss"]
        return {
            **self._stats,
            "total": total,
            "hit_rate": self._stats["hit"] / total if total > 0 else 0,
        }

    def log_stats(self):
        stats = self.get_stats()
        logger.info(
            "EmpiricalPerformanceModel: %d/%d ops matched (%.1f%%)",
            stats["hit"],
            stats["total"],
            stats["hit_rate"] * 100,
        )

        # Deduplicated HITs: count occurrences of each mapping
        if self._hit_details:
            display_keys = [f"{fn}->{kt}" for fn, kt, _, _ in self._hit_details]
            hit_counts = Counter(display_keys)
            hit_lines = [
                f"  {mapping} (x{count})" if count > 1 else f"  {mapping}"
                for mapping, count in hit_counts.most_common()
            ]
            logger.info(
                "  HITs (%d unique):\n%s", len(hit_counts), "\n".join(hit_lines)
            )

        # MISSes grouped by reason category
        if self._miss_details:
            by_reason: dict[str, list[tuple[str, list[tuple]]]] = {}
            for func_name, reason, tc_shapes in self._miss_details:
                by_reason.setdefault(reason, []).append((func_name, tc_shapes))

            miss_lines = []
            for reason, ops in sorted(by_reason.items()):
                label = _MISS_REASON_LABELS.get(reason, reason)
                # Deduplicate ops with same func_name
                op_counts = Counter(func_name for func_name, _ in ops)
                op_strs = [
                    f"{name} (x{count})" if count > 1 else name
                    for name, count in op_counts.most_common()
                ]
                miss_lines.append(f"  [{reason}] {label}: {', '.join(op_strs)}")
                # Log shape details at DEBUG level
                for func_name, tc_shapes in ops:
                    logger.debug("    %s shapes: %s", func_name, tc_shapes)

            logger.info(
                "  MISSes (%d unique reasons):\n%s",
                len(by_reason),
                "\n".join(miss_lines),
            )

        # Fused Op Match Rate
        fused = compute_fused_op_stats(self._hit_details, self._miss_details)
        logger.info(
            "Fused Op Match Rate: %d/%d (%.1f%%) [GO/NO-GO]",
            fused["fused_hit"],
            fused["fused_total"],
            fused["fused_hr"] * 100,
        )
        logger.info(
            "Fused Op Match Rate (excl zero_cost): %d/%d (%.1f%%) [Reference]",
            fused["fused_hit_no_zc"],
            fused["fused_total_no_zc"],
            fused["fused_hr_no_zc"] * 100,
        )

        # M4: Per-Shape Match Rate
        shape_stats = compute_per_shape_stats(self._hit_details, self._miss_details)
        logger.info(
            "Per-Shape Match Rate: %d/%d (%.1f%%)",
            shape_stats["hit_shapes"],
            shape_stats["total_shapes"],
            shape_stats["m4"] * 100,
        )
        if shape_stats["miss_shape_list"]:
            miss_lines = [
                f"  {fn} {ss}" for fn, ss in shape_stats["miss_shape_list"][:20]
            ]
            remaining = len(shape_stats["miss_shape_list"]) - 20
            if remaining > 0:
                miss_lines.append(f"  ... and {remaining} more")
            logger.info(
                "  MISS shapes (%d):\n%s",
                len(shape_stats["miss_shape_list"]),
                "\n".join(miss_lines),
            )
