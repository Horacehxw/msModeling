from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class QueryResult:
    kernel_type: str
    latency_us: float
    source: str
    metadata: Mapping[str, Any] | None = None


class DataSource(ABC):
    @abstractmethod
    def query(self, kernel_type: str, features: Mapping[str, Any]) -> QueryResult | None:
        """Query perf data by kernel type and input features."""
