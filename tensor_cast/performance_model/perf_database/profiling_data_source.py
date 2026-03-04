import csv
from pathlib import Path
from typing import Any, Mapping

from .data_source import DataSource, QueryResult


class ProfilingDataSource(DataSource):
    """CSV-backed datasource with exact-match query on input/output shapes."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def query(self, kernel_type: str, features: Mapping[str, Any]) -> QueryResult | None:
        csv_path = self.root / f"{kernel_type}.csv"
        if not csv_path.exists():
            return None

        input_shapes = str(features.get("input_shapes", ""))
        output_shapes = str(features.get("output_shapes", ""))
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (
                    row.get("Input Shapes", "") == input_shapes
                    and row.get("Output Shapes", "") == output_shapes
                ):
                    latency = float(row.get("Average Duration(us)", "0") or 0.0)
                    return QueryResult(
                        kernel_type=kernel_type,
                        latency_us=latency,
                        source="profiling_csv",
                        metadata={"csv_path": str(csv_path)},
                    )
        return None
