"""Parser for TensorCast chrome trace JSON files.

This module parses the chrome trace JSON output from TensorCast's
text_generate.py --chrome-trace option and extracts ordered operation events.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class TraceEvent:
    """A single operation event from a chrome trace file.

    Attributes:
        name: Normalized operation name (e.g., "aten.mm")
        duration_us: Duration in microseconds
        start_us: Start timestamp in microseconds
        args: Additional arguments from the trace event
    """

    name: str
    duration_us: float
    start_us: float
    args: Dict = field(default_factory=dict)


def normalize_trace_name(raw_name: str) -> str:
    """Normalize a chrome trace operation name.

    Converts PyTorch-style names to dot-separated format:
      "aten::mm::default" -> "aten.mm"
      "tensor_cast::attention::default" -> "tensor_cast.attention"
      "aten::_to_copy::default" -> "aten._to_copy"

    Internal marker events are preserved as-is:
      "_internal_mark_region_begin" -> "_internal_mark_region_begin"

    Args:
        raw_name: Raw operation name from chrome trace

    Returns:
        Normalized name string
    """
    if "::" not in raw_name:
        return raw_name

    parts = raw_name.split("::")
    # Filter out "default" suffix
    parts = [p for p in parts if p != "default"]
    return ".".join(parts)


def parse_chrome_trace(path: Path) -> List[TraceEvent]:
    """Parse a chrome trace JSON file into ordered TraceEvent list.

    Filters for complete duration events (ph=="X"), sorts by start
    timestamp, and normalizes operation names.

    Args:
        path: Path to the chrome trace JSON file

    Returns:
        List of TraceEvent objects sorted by start time

    Raises:
        FileNotFoundError: If the trace file doesn't exist
        ValueError: If the JSON is invalid or missing traceEvents
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Chrome trace file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Chrome trace format: {"traceEvents": [...]} or just [...]
    if isinstance(data, dict):
        events_raw = data.get("traceEvents", [])
    elif isinstance(data, list):
        events_raw = data
    else:
        raise ValueError(f"Unexpected chrome trace format: {type(data)}")

    events = []
    for entry in events_raw:
        # Only process complete duration events
        if entry.get("ph") != "X":
            continue

        raw_name = entry.get("name", "")
        if not raw_name:
            continue

        duration_us = entry.get("dur", 0.0)
        start_us = entry.get("ts", 0.0)
        args = entry.get("args", {})

        events.append(
            TraceEvent(
                name=normalize_trace_name(raw_name),
                duration_us=float(duration_us),
                start_us=float(start_us),
                args=args,
            )
        )

    # Sort by start timestamp
    events.sort(key=lambda e: e.start_us)
    return events
