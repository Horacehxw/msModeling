from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..op_invoke_info import OpInvokeInfo


class QuerySource(Enum):
    MEASURED = auto()
    INTERPOLATED = auto()
    EXTRAPOLATED = auto()
    # Forward-declared: returned by _lookup_composite_decomposed when some
    # (but not all) sub-kernels hit, enabling partial composite estimation.
    PARTIAL = auto()


@dataclass
class QueryResult:
    latency_us: float
    confidence: float
    source: QuerySource
    details: Dict[str, Any] = field(default_factory=dict)


class DataSourcePerformanceModel(ABC):
    """Abstract base class for performance data sources.
    TensorCast queries via OpInvokeInfo only, unaware of underlying data format.
    (Design doc §4.1)"""

    @abstractmethod
    def lookup(self, op_invoke_info: "OpInvokeInfo") -> Optional[QueryResult]:
        """Query operator performance from OpInvokeInfo."""
        ...

    def store(self, op_invoke_info: "OpInvokeInfo", result: QueryResult) -> None:
        """Store performance data (optional). Default: read-only."""
        raise NotImplementedError("This data source is read-only")
