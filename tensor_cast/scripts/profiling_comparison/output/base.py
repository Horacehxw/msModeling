"""Base output formatter protocol and data structures.

This module defines the interface for output formatters and common
data structures used across all formatters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Protocol, runtime_checkable


@dataclass
class OperationMatch:
    """Result of matching a VLLM operation to TensorCast operations.

    Attributes:
        vllm_op: VLLM operation name
        tc_ops: List of matched TensorCast operation names
        vllm_duration_us: VLLM operation duration in microseconds
        tc_duration_us: TensorCast operation duration in microseconds
        vllm_count: Number of VLLM operation invocations
        tc_count: Number of TensorCast operation invocations
        difference_us: Time difference (TC - VLLM) in microseconds
        difference_pct: Time difference as percentage
        match_status: Match quality ("exact", "partial", "missing_in_tc", "missing_in_vllm")
    """

    vllm_op: str
    tc_ops: List[str]
    vllm_duration_us: float
    tc_duration_us: float
    vllm_count: int = 0
    tc_count: int = 0
    difference_us: float = 0.0
    difference_pct: float = 0.0
    match_status: str = "unknown"


@dataclass
class ComparisonSummary:
    """Summary statistics for a comparison.

    Attributes:
        vllm_total_time_us: Total VLLM execution time
        tc_total_time_us: Total TensorCast execution time
        overall_diff_us: Overall time difference
        overall_diff_pct: Overall difference percentage
        num_matched: Number of matched operations
        num_missing_in_tc: Operations in VLLM but not matched to TC
        num_missing_in_vllm: Operations in TC but not matched to VLLM
        coverage_pct: Percentage of VLLM time covered by matches
        avg_abs_diff_pct: Average absolute difference percentage
    """

    vllm_total_time_us: float
    tc_total_time_us: float
    overall_diff_us: float
    overall_diff_pct: float
    num_matched: int
    num_missing_in_tc: int
    num_missing_in_vllm: int
    coverage_pct: float
    avg_abs_diff_pct: float


@dataclass
class ComparisonResult:
    """Complete result of a profiling comparison.

    Attributes:
        config: Configuration used for comparison
        vllm_operations: VLLM operation statistics
        tc_operations: TensorCast operation statistics
        matches: List of operation matches
        summary: Comparison summary statistics
        metadata: Additional metadata
    """

    config: Dict
    vllm_operations: List[Dict]
    tc_operations: List[Dict]
    matches: List[OperationMatch]
    summary: ComparisonSummary
    metadata: Dict = field(default_factory=dict)

    def get_matches_by_status(self, status: str) -> List[OperationMatch]:
        """Get matches filtered by status."""
        return [m for m in self.matches if m.match_status == status]

    def get_top_differences(self, n: int = 10) -> List[OperationMatch]:
        """Get top N matches by absolute difference percentage."""
        valid = [m for m in self.matches if m.difference_pct != float("inf")]
        return sorted(valid, key=lambda x: abs(x.difference_pct), reverse=True)[:n]


@runtime_checkable
class FormatterProtocol(Protocol):
    """Protocol for output formatters."""

    def format(self, result: ComparisonResult, output_path: Path) -> None:
        """Format and save comparison result.

        Args:
            result: Comparison result to format
            output_path: Path to save output
        """
        ...


class BaseFormatter(ABC):
    """Abstract base class for output formatters."""

    @abstractmethod
    def format(self, result: ComparisonResult, output_path: Path) -> None:
        """Format and save comparison result.

        Args:
            result: Comparison result to format
            output_path: Path to save output
        """

    @staticmethod
    def _to_float(value) -> float:
        """Convert a value to float, handling PyTorch tensors."""
        if hasattr(value, "item"):
            return value.item()
        return float(value)
