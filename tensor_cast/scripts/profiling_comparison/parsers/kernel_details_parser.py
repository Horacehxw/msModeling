"""Parser for extracting single decode step from kernel_details.csv.

This module provides functionality to parse VLLM/Ascend profiling data and extract
operations from a single decode step for fair comparison with TensorCast simulation.

The key insight is that each decode step contains exactly one FusedInferAttentionScore
kernel, which serves as an anchor to identify step boundaries.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class KernelOp:
    """Represents a single kernel operation from kernel_details.csv."""

    name: str
    op_type: str
    start_time_us: float
    duration_us: float
    end_time_us: float
    core_type: str
    input_shapes: str
    output_shapes: str

    @property
    def is_attention_anchor(self) -> bool:
        """Check if this operation is the step boundary anchor."""
        return self.op_type == "FusedInferAttentionScore"


@dataclass
class SingleStepData:
    """Container for single decode step data."""

    step_index: int
    operations: List[KernelOp]
    start_time_us: float
    end_time_us: float
    total_duration_us: float

    def __post_init__(self):
        if self.operations:
            self.total_duration_us = sum(op.duration_us for op in self.operations)

    def get_ops_by_type(self) -> Dict[str, List[KernelOp]]:
        """Group operations by type."""
        ops_by_type: Dict[str, List[KernelOp]] = {}
        for op in self.operations:
            if op.op_type not in ops_by_type:
                ops_by_type[op.op_type] = []
            ops_by_type[op.op_type].append(op)
        return ops_by_type

    def get_aggregated_stats(self) -> Dict[str, dict]:
        """Get aggregated statistics by operation type."""
        stats: Dict[str, dict] = {}
        for op in self.operations:
            if op.op_type not in stats:
                stats[op.op_type] = {
                    "count": 0,
                    "total_duration_us": 0.0,
                    "core_types": set(),
                    "input_shapes": set(),
                    "output_shapes": set(),
                }
            s = stats[op.op_type]
            s["count"] += 1
            s["total_duration_us"] += op.duration_us
            s["core_types"].add(op.core_type)
            if op.input_shapes:
                s["input_shapes"].add(op.input_shapes)
            if op.output_shapes:
                s["output_shapes"].add(op.output_shapes)

        # Calculate averages
        for s in stats.values():
            s["avg_duration_us"] = (
                s["total_duration_us"] / s["count"] if s["count"] > 0 else 0.0
            )
            s["core_types"] = list(s["core_types"])
            s["input_shapes"] = list(s["input_shapes"])
            s["output_shapes"] = list(s["output_shapes"])

        return stats


class KernelDetailsParser:
    """Parser for kernel_details.csv that can extract single decode steps."""

    # Column name variations in kernel_details.csv
    COLUMN_MAPPINGS = {
        "name": ["Name", "Op Name", "name"],
        "op_type": ["Type", "Op Type", "OP Type", "type"],
        "start_time": ["Start Time(us)", "Start Time (us)", "start_time_us"],
        "duration": ["Duration(us)", "Duration (us)", "duration_us"],
        "core_type": ["Accelerator Core", "Core Type", "core_type"],
        "input_shapes": ["Input Shapes", "input_shapes"],
        "output_shapes": ["Output Shapes", "output_shapes"],
    }

    def __init__(self, csv_path: Path):
        """Initialize parser with path to kernel_details.csv.

        Args:
            csv_path: Path to the kernel_details.csv file
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"kernel_details.csv not found: {self.csv_path}")

        self._operations: Optional[List[KernelOp]] = None
        self._step_boundaries: Optional[List[float]] = None

    def _find_column(self, row_keys: List[str], column_type: str) -> Optional[str]:
        """Find the actual column name for a given column type."""
        for candidate in self.COLUMN_MAPPINGS[column_type]:
            for key in row_keys:
                # Handle potential tab/whitespace in column names
                clean_key = key.strip().replace("\t", "")
                if clean_key == candidate:
                    return key
        return None

    def parse_all_operations(self) -> List[KernelOp]:
        """Parse all operations from kernel_details.csv.

        Returns:
            List of KernelOp objects sorted by start time
        """
        if self._operations is not None:
            return self._operations

        operations = []

        with open(self.csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row_keys = reader.fieldnames if reader.fieldnames else []

            # Find actual column names
            name_col = self._find_column(row_keys, "name")
            type_col = self._find_column(row_keys, "op_type")
            start_col = self._find_column(row_keys, "start_time")
            dur_col = self._find_column(row_keys, "duration")
            core_col = self._find_column(row_keys, "core_type")
            input_col = self._find_column(row_keys, "input_shapes")
            output_col = self._find_column(row_keys, "output_shapes")

            if not all([name_col, type_col, start_col, dur_col]):
                raise ValueError(
                    f"Missing required columns. Found: {row_keys}. "
                    f"name={name_col}, type={type_col}, start={start_col}, dur={dur_col}"
                )

            for row in reader:
                try:
                    # Parse start time (handle potential tab suffix)
                    start_time_str = row.get(start_col, "0").strip().replace("\t", "")
                    start_time = float(start_time_str) if start_time_str else 0.0

                    # Parse duration
                    duration_str = row.get(dur_col, "0").strip()
                    duration = float(duration_str) if duration_str else 0.0

                    op = KernelOp(
                        name=row.get(name_col, "").strip(),
                        op_type=row.get(type_col, "").strip(),
                        start_time_us=start_time,
                        duration_us=duration,
                        end_time_us=start_time + duration,
                        core_type=row.get(core_col, "").strip() if core_col else "",
                        input_shapes=row.get(input_col, "").strip()
                        if input_col
                        else "",
                        output_shapes=row.get(output_col, "").strip()
                        if output_col
                        else "",
                    )

                    # Skip invalid entries
                    if op.op_type and op.start_time_us >= 0:
                        operations.append(op)

                except (ValueError, KeyError):
                    # Skip malformed rows
                    continue

        # Sort by start time
        self._operations = sorted(operations, key=lambda x: x.start_time_us)
        return self._operations

    def find_step_boundaries(self) -> List[float]:
        """Find decode step boundaries using FusedInferAttentionScore as anchor.

        Each decode step contains exactly one FusedInferAttentionScore kernel.
        The start time of each such kernel marks the beginning of a decode step.

        Returns:
            List of start times (in us) marking step boundaries
        """
        if self._step_boundaries is not None:
            return self._step_boundaries

        operations = self.parse_all_operations()

        boundaries = []
        for op in operations:
            if op.is_attention_anchor:
                boundaries.append(op.start_time_us)

        self._step_boundaries = sorted(boundaries)
        return self._step_boundaries

    def get_num_decode_steps(self) -> int:
        """Get the total number of decode steps detected."""
        return len(self.find_step_boundaries())

    def extract_single_step(self, step_index: int = 100) -> SingleStepData:
        """Extract all operations within one decode step.

        Args:
            step_index: Which decode step to extract (0-indexed).
                        Default is 100 to avoid warmup effects at the beginning.

        Returns:
            SingleStepData containing all operations in the specified step

        Raises:
            ValueError: If step_index is out of range
        """
        boundaries = self.find_step_boundaries()
        num_steps = len(boundaries)

        if num_steps == 0:
            raise ValueError(
                "No decode steps detected (no FusedInferAttentionScore found)"
            )

        if step_index < 0 or step_index >= num_steps:
            raise ValueError(
                f"step_index {step_index} out of range. "
                f"Valid range: 0 to {num_steps - 1}"
            )

        # Get step boundaries
        start_time = boundaries[step_index]

        # End time is either next step boundary or end of trace
        if step_index + 1 < num_steps:
            end_time = boundaries[step_index + 1]
        else:
            # Last step: include all remaining ops
            operations = self.parse_all_operations()
            end_time = (
                max(op.end_time_us for op in operations) if operations else start_time
            )

        # Extract operations within this step
        operations = self.parse_all_operations()
        step_ops = [
            op for op in operations if start_time <= op.start_time_us < end_time
        ]

        return SingleStepData(
            step_index=step_index,
            operations=step_ops,
            start_time_us=start_time,
            end_time_us=end_time,
            total_duration_us=0.0,  # Will be calculated in __post_init__
        )

    def get_step_statistics(self) -> Dict[str, float]:
        """Get statistics about decode steps.

        Returns:
            Dictionary with step statistics
        """
        boundaries = self.find_step_boundaries()
        num_steps = len(boundaries)

        if num_steps < 2:
            return {
                "num_steps": num_steps,
                "avg_step_duration_us": 0.0,
                "min_step_duration_us": 0.0,
                "max_step_duration_us": 0.0,
            }

        # Calculate step durations (time between boundaries)
        step_durations = []
        for i in range(len(boundaries) - 1):
            duration = boundaries[i + 1] - boundaries[i]
            step_durations.append(duration)

        return {
            "num_steps": num_steps,
            "avg_step_duration_us": sum(step_durations) / len(step_durations),
            "min_step_duration_us": min(step_durations),
            "max_step_duration_us": max(step_durations),
        }

    def get_operations_per_step_estimate(self) -> float:
        """Estimate average number of operations per decode step."""
        num_steps = self.get_num_decode_steps()
        if num_steps == 0:
            return 0.0

        total_ops = len(self.parse_all_operations())
        return total_ops / num_steps


def parse_kernel_details(csv_path: Path) -> List[KernelOp]:
    """Convenience function to parse all kernel operations.

    Args:
        csv_path: Path to kernel_details.csv

    Returns:
        List of KernelOp objects
    """
    parser = KernelDetailsParser(csv_path)
    return parser.parse_all_operations()


def find_step_boundaries(ops: List[KernelOp]) -> List[float]:
    """Find decode step boundaries using FusedInferAttentionScore as anchor.

    Args:
        ops: List of kernel operations

    Returns:
        List of start times marking step boundaries
    """
    return sorted([op.start_time_us for op in ops if op.is_attention_anchor])


def extract_single_step(
    ops: List[KernelOp], boundaries: List[float], step_index: int = 100
) -> List[KernelOp]:
    """Extract all operations within one decode step.

    Args:
        ops: List of all kernel operations
        boundaries: List of step boundary times
        step_index: Which step to extract (default 100 to avoid warmup)

    Returns:
        List of operations in the specified step
    """
    if not boundaries or step_index >= len(boundaries):
        raise ValueError(
            f"Invalid step_index {step_index} for {len(boundaries)} boundaries"
        )

    start = boundaries[step_index]
    end = (
        boundaries[step_index + 1] if step_index + 1 < len(boundaries) else float("inf")
    )

    return [op for op in ops if start <= op.start_time_us < end]
