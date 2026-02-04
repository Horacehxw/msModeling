"""Base parser protocol for profiling data.

This module defines the abstract interface that all parsers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class OperationData:
    """Unified operation data structure.

    This dataclass provides a common representation for operations
    from both VLLM profiling and TensorCast simulation.

    Attributes:
        op_type: Operation type name
        count: Number of times this operation was executed
        total_time_us: Total execution time in microseconds
        avg_time_us: Average execution time in microseconds
        input_shapes: List of input tensor shapes
        output_shapes: List of output tensor shapes
        core_type: Accelerator core type (VLLM specific)
        bound_classification: Compute/memory bound (TensorCast specific)
        metadata: Additional operation-specific metadata
    """

    op_type: str
    count: int
    total_time_us: float
    avg_time_us: float
    input_shapes: List[str]
    output_shapes: List[str]
    core_type: str = ""
    bound_classification: str = ""
    metadata: Optional[Dict] = None

    @property
    def total_time_ms(self) -> float:
        """Get total time in milliseconds."""
        return self.total_time_us / 1000.0

    @property
    def avg_time_ms(self) -> float:
        """Get average time in milliseconds."""
        return self.avg_time_us / 1000.0


@dataclass
class ProfilingResult:
    """Container for parsed profiling data.

    Attributes:
        operations: List of operation data
        total_time_us: Total execution time in microseconds
        metadata: Additional result metadata
        source: Source identifier ("vllm" or "tensorcast")
    """

    operations: List[OperationData]
    total_time_us: float
    source: str
    metadata: Optional[Dict] = None

    def get_ops_by_type(self, op_type: str) -> List[OperationData]:
        """Get all operations of a specific type."""
        return [op for op in self.operations if op.op_type == op_type]

    def get_top_ops(self, n: int = 20) -> List[OperationData]:
        """Get top N operations by total time."""
        return sorted(self.operations, key=lambda x: x.total_time_us, reverse=True)[:n]

    def get_time_breakdown(self) -> Dict[str, float]:
        """Get time breakdown by operation type."""
        breakdown = {}
        for op in self.operations:
            if op.op_type not in breakdown:
                breakdown[op.op_type] = 0.0
            breakdown[op.op_type] += op.total_time_us
        return breakdown


@runtime_checkable
class Parser(Protocol):
    """Protocol for profiling data parsers.

    All parsers must implement this interface to be usable
    by the profiling comparison tool.
    """

    def parse(self) -> ProfilingResult:
        """Parse profiling data and return unified result.

        Returns:
            ProfilingResult containing parsed operation data
        """
        ...


class BaseParser(ABC):
    """Abstract base class for profiling data parsers.

    Provides common functionality for all parsers.
    """

    @abstractmethod
    def parse(self) -> ProfilingResult:
        """Parse profiling data and return unified result.

        Returns:
            ProfilingResult containing parsed operation data
        """

    @staticmethod
    def _to_float(value) -> float:
        """Convert a value to float, handling PyTorch tensors.

        Args:
            value: Value to convert

        Returns:
            Float value
        """
        if hasattr(value, "item"):
            return value.item()
        return float(value)
