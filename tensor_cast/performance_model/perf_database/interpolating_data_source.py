from typing import Any, Mapping

from .data_source import DataSource, QueryResult


class InterpolatingDataSource(DataSource):
    """Wrapper datasource for future interpolation fallback."""

    def __init__(self, base: DataSource):
        self.base = base

    def query(self, kernel_type: str, features: Mapping[str, Any]) -> QueryResult | None:
        result = self.base.query(kernel_type, features)
        if result is not None:
            return result
        return None
