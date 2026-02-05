"""Parser for extracting single decode step from kernel_details.csv.

This module provides functionality to parse VLLM/Ascend profiling data and extract
operations from a single decode step for fair comparison with TensorCast simulation.

A "decode step" is defined as ONE COMPLETE forward pass through ALL layers of the model.
For a model with N layers, each forward pass contains exactly N FusedInferAttentionScore
operations (one per layer). We group attention operations by layer count to define
step boundaries.

Example: For DeepSeek-V3 (61 layers):
- Every 61 consecutive FusedInferAttentionScore operations = 1 complete forward pass
- Step 0: attention ops 0-60, Step 1: attention ops 61-121, etc.
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

    def find_step_boundaries(self, num_layers: Optional[int] = None) -> List[float]:
        """Find complete forward pass boundaries.

        A complete forward pass contains exactly `num_layers` attention operations.
        We group attention operations by layer count to define step boundaries.

        Args:
            num_layers: Number of decoder layers in the model. If None, falls back
                       to treating each attention op as a step boundary (legacy behavior).

        Returns:
            List of start times (in us) marking step boundaries
        """
        # Note: We don't cache boundaries when num_layers is provided since it may vary
        if num_layers is None and self._step_boundaries is not None:
            return self._step_boundaries

        operations = self.parse_all_operations()

        # Collect all attention anchors sorted by time
        attention_ops = [op for op in operations if op.is_attention_anchor]

        if num_layers is None or num_layers <= 0:
            # Legacy behavior: each attention op is a step boundary
            boundaries = [op.start_time_us for op in attention_ops]
        else:
            # New behavior: group by num_layers
            # Every num_layers consecutive attention ops = 1 complete forward pass
            boundaries = []
            for i in range(0, len(attention_ops), num_layers):
                if i + num_layers <= len(attention_ops):
                    # Only include complete forward passes
                    boundaries.append(attention_ops[i].start_time_us)

        boundaries = sorted(boundaries)

        # Only cache if using legacy behavior
        if num_layers is None:
            self._step_boundaries = boundaries

        return boundaries

    def get_num_decode_steps(self, num_layers: Optional[int] = None) -> int:
        """Get the total number of decode steps detected.

        Args:
            num_layers: Number of decoder layers in the model.

        Returns:
            Number of complete forward passes detected.
        """
        return len(self.find_step_boundaries(num_layers))

    def extract_single_step(
        self, step_index: int = 0, num_layers: Optional[int] = None
    ) -> SingleStepData:
        """Extract all operations within one complete forward pass.

        Args:
            step_index: Which forward pass to extract (0-indexed).
                        Default is 0 (first complete forward pass).
            num_layers: Number of decoder layers in the model. If provided,
                       a "step" is defined as num_layers consecutive attention
                       operations (one complete forward pass).

        Returns:
            SingleStepData containing all operations in the specified step

        Raises:
            ValueError: If step_index is out of range
        """
        boundaries = self.find_step_boundaries(num_layers)
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

    def get_step_statistics(self, num_layers: Optional[int] = None) -> Dict[str, float]:
        """Get statistics about complete forward passes.

        Args:
            num_layers: Number of decoder layers in the model.

        Returns:
            Dictionary with step statistics
        """
        boundaries = self.find_step_boundaries(num_layers)
        num_steps = len(boundaries)

        # Also count total attention ops for debugging
        operations = self.parse_all_operations()
        total_attention_ops = sum(1 for op in operations if op.is_attention_anchor)

        if num_steps < 2:
            return {
                "num_steps": num_steps,
                "total_attention_ops": total_attention_ops,
                "num_layers": num_layers or 0,
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
            "total_attention_ops": total_attention_ops,
            "num_layers": num_layers or 0,
            "avg_step_duration_us": sum(step_durations) / len(step_durations),
            "min_step_duration_us": min(step_durations),
            "max_step_duration_us": max(step_durations),
        }

    def get_operations_per_step_estimate(
        self, num_layers: Optional[int] = None
    ) -> float:
        """Estimate average number of operations per complete forward pass.

        Args:
            num_layers: Number of decoder layers in the model.

        Returns:
            Average number of operations per forward pass.
        """
        num_steps = self.get_num_decode_steps(num_layers)
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


def find_step_boundaries(
    ops: List[KernelOp], num_layers: Optional[int] = None
) -> List[float]:
    """Find complete forward pass boundaries.

    Args:
        ops: List of kernel operations
        num_layers: Number of decoder layers in the model.

    Returns:
        List of start times marking step boundaries
    """
    attention_ops = sorted(
        [op for op in ops if op.is_attention_anchor],
        key=lambda x: x.start_time_us,
    )

    if num_layers is None or num_layers <= 0:
        return [op.start_time_us for op in attention_ops]

    boundaries = []
    for i in range(0, len(attention_ops), num_layers):
        if i + num_layers <= len(attention_ops):
            boundaries.append(attention_ops[i].start_time_us)
    return boundaries


def extract_single_step(
    ops: List[KernelOp], boundaries: List[float], step_index: int = 0
) -> List[KernelOp]:
    """Extract all operations within one complete forward pass.

    Args:
        ops: List of all kernel operations
        boundaries: List of step boundary times
        step_index: Which step to extract (default 0 for first complete step)

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
