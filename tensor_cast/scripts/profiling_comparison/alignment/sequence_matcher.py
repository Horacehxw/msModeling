"""Order-based sequence matching between VLLM and TensorCast operations.

This module provides a lockstep matching algorithm that walks both the VLLM
and TensorCast operation sequences in execution order, consuming TC ops for
each VLLM op according to a decomposition mapping. This eliminates
double-counting entirely since each TC op is consumed exactly once.

Algorithm complexity: O(N + M) where N = VLLM ops, M = TC ops.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

from ..parsers.kernel_details_parser import KernelOp
from ..parsers.chrome_trace_parser import TraceEvent

logger = logging.getLogger(__name__)

# Directory containing mapping YAML files
MAPPINGS_DIR = Path(__file__).parent / "mappings"


@dataclass
class SequenceMatch:
    """Result of matching a single VLLM op to its consumed TC ops.

    Attributes:
        vllm_op: The VLLM kernel operation (or None for unmatched TC ops)
        tc_events: List of consumed TensorCast trace events
        vllm_duration_us: VLLM operation duration in microseconds
        tc_duration_us: Sum of TC event durations in microseconds
        difference_us: TC - VLLM duration difference
        difference_pct: Difference as percentage of VLLM duration
        match_status: One of "matched", "mismatch", "unmatched_vllm", "unmatched_tc"
        decomposition_name: Which decomposition rule was applied (empty for unmatched)
        position: Position index in the sequence (for ordering)
    """

    vllm_op: Optional[KernelOp]
    tc_events: List[TraceEvent]
    vllm_duration_us: float
    tc_duration_us: float
    difference_us: float
    difference_pct: float
    match_status: str
    decomposition_name: str = ""
    position: int = 0


@dataclass
class DecompositionConfig:
    """Parsed decomposition mapping configuration.

    Attributes:
        version: Config version string
        decompositions: Maps VLLM op type -> list of expected TC op names
        ignored_vllm_ops: VLLM op types to skip during matching
        ignored_tc_ops: TC op names to skip during matching
    """

    version: str = "2.0"
    decompositions: Dict[str, List[str]] = field(default_factory=dict)
    ignored_vllm_ops: Set[str] = field(default_factory=set)
    ignored_tc_ops: Set[str] = field(default_factory=set)


def load_decomposition_config(mapping_name: str = "default") -> DecompositionConfig:
    """Load a decomposition mapping from YAML file.

    Args:
        mapping_name: Name of the mapping file (without extension)

    Returns:
        DecompositionConfig instance

    Raises:
        FileNotFoundError: If mapping file doesn't exist
    """
    yaml_path = MAPPINGS_DIR / f"{mapping_name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Mapping '{mapping_name}' not found at {yaml_path}"
        )
    return load_decomposition_config_from_path(yaml_path)


def load_decomposition_config_from_path(yaml_path: Path) -> DecompositionConfig:
    """Load a decomposition mapping from a specific YAML file path.

    Args:
        yaml_path: Path to the YAML file

    Returns:
        DecompositionConfig instance
    """
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty mapping file: {yaml_path}")

    config = DecompositionConfig(version=data.get("version", "2.0"))

    if "decompositions" in data:
        for vllm_op, entry in data["decompositions"].items():
            if isinstance(entry, dict):
                tc_ops = entry.get("tc_ops", [])
            elif isinstance(entry, list):
                tc_ops = entry
            else:
                continue
            config.decompositions[vllm_op] = tc_ops

    if "ignored_vllm_ops" in data:
        config.ignored_vllm_ops = set(data["ignored_vllm_ops"])

    if "ignored_tc_ops" in data:
        config.ignored_tc_ops = set(data["ignored_tc_ops"])

    return config


def merge_decomposition_configs(*configs: DecompositionConfig) -> DecompositionConfig:
    """Merge multiple decomposition configs (later configs take precedence).

    Args:
        *configs: DecompositionConfig instances to merge

    Returns:
        Merged DecompositionConfig
    """
    merged = DecompositionConfig()
    for config in configs:
        merged.decompositions.update(config.decompositions)
        merged.ignored_vllm_ops |= config.ignored_vllm_ops
        merged.ignored_tc_ops |= config.ignored_tc_ops
    return merged


def match_by_sequence(
    vllm_ops: List[KernelOp],
    tc_events: List[TraceEvent],
    decompositions: Dict[str, List[str]],
    ignored_vllm_ops: Set[str],
    ignored_tc_ops: Set[str],
) -> List[SequenceMatch]:
    """Match VLLM ops to TC events by walking both sequences in lockstep.

    For each VLLM op (in execution order):
    1. Skip if its type is in ignored_vllm_ops
    2. Look up expected TC ops from decompositions
    3. Advance through TC events, skipping ignored ones
    4. Consume matching TC events in order
    5. If mismatch, log warning and emit mismatch record

    Remaining unconsumed TC events are emitted as "unmatched_tc".

    Args:
        vllm_ops: Ordered list of VLLM kernel operations
        tc_events: Ordered list of TensorCast trace events
        decompositions: Maps VLLM op type -> expected TC op names
        ignored_vllm_ops: VLLM op types to skip
        ignored_tc_ops: TC op names to skip

    Returns:
        List of SequenceMatch records
    """
    matches: List[SequenceMatch] = []
    tc_idx = 0
    position = 0

    for vllm_op in vllm_ops:
        # Skip ignored VLLM ops
        if vllm_op.op_type in ignored_vllm_ops:
            continue

        # Check if this VLLM op has a decomposition mapping
        if vllm_op.op_type not in decompositions:
            matches.append(
                SequenceMatch(
                    vllm_op=vllm_op,
                    tc_events=[],
                    vllm_duration_us=vllm_op.duration_us,
                    tc_duration_us=0.0,
                    difference_us=-vllm_op.duration_us,
                    difference_pct=-100.0,
                    match_status="unmatched_vllm",
                    position=position,
                )
            )
            position += 1
            continue

        expected_tc_ops = decompositions[vllm_op.op_type]
        consumed: List[TraceEvent] = []
        match_ok = True

        for expected_name in expected_tc_ops:
            # Advance past ignored TC ops
            while tc_idx < len(tc_events) and tc_events[tc_idx].name in ignored_tc_ops:
                tc_idx += 1

            if tc_idx >= len(tc_events):
                logger.warning(
                    "Ran out of TC events while matching VLLM op '%s' at position %d. "
                    "Expected TC op '%s'.",
                    vllm_op.op_type,
                    position,
                    expected_name,
                )
                match_ok = False
                break

            if tc_events[tc_idx].name == expected_name:
                consumed.append(tc_events[tc_idx])
                tc_idx += 1
            else:
                logger.warning(
                    "TC op mismatch at position %d: VLLM op '%s' expected TC op '%s' "
                    "but found '%s' (tc_idx=%d).",
                    position,
                    vllm_op.op_type,
                    expected_name,
                    tc_events[tc_idx].name,
                    tc_idx,
                )
                match_ok = False
                break

        tc_time = sum(e.duration_us for e in consumed)
        vllm_time = vllm_op.duration_us

        if vllm_time > 0:
            diff_us = tc_time - vllm_time
            diff_pct = (diff_us / vllm_time) * 100
        else:
            diff_us = tc_time
            diff_pct = 0.0 if tc_time == 0 else float("inf")

        if match_ok and consumed:
            status = "matched"
        elif consumed:
            status = "mismatch"
        else:
            status = "mismatch"

        matches.append(
            SequenceMatch(
                vllm_op=vllm_op,
                tc_events=consumed,
                vllm_duration_us=vllm_time,
                tc_duration_us=tc_time,
                difference_us=diff_us,
                difference_pct=diff_pct,
                match_status=status,
                decomposition_name=vllm_op.op_type,
                position=position,
            )
        )
        position += 1

    # Collect remaining unconsumed TC ops (skipping ignored ones)
    while tc_idx < len(tc_events):
        if tc_events[tc_idx].name in ignored_tc_ops:
            tc_idx += 1
            continue

        event = tc_events[tc_idx]
        matches.append(
            SequenceMatch(
                vllm_op=None,
                tc_events=[event],
                vllm_duration_us=0.0,
                tc_duration_us=event.duration_us,
                difference_us=event.duration_us,
                difference_pct=float("inf") if event.duration_us > 0 else 0.0,
                match_status="unmatched_tc",
                position=position,
            )
        )
        position += 1
        tc_idx += 1

    return matches
