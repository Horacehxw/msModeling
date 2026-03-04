from typing import Optional, TYPE_CHECKING

from .data_source import DataSource, QueryResult

if TYPE_CHECKING:
    from ..op_invoke_info import OpInvokeInfo


class InterpolatingDataSource(DataSource):
    """Wrapper datasource for future interpolation fallback. (Design doc §4.4)"""

    def __init__(self, base: DataSource):
        self.base = base

    def lookup(self, op_invoke_info: "OpInvokeInfo") -> Optional[QueryResult]:
        result = self.base.lookup(op_invoke_info)
        if result is not None:
            return result
        # TODO: interpolation logic (Phase 2)
        return None
