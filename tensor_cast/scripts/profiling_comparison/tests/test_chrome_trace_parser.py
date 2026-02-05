"""Tests for chrome trace parser module."""

import json
import tempfile
from pathlib import Path

import pytest

from tensor_cast.scripts.profiling_comparison.parsers.chrome_trace_parser import (
    normalize_trace_name,
    parse_chrome_trace,
    TraceEvent,
)


class TestNormalizeTraceName:
    """Tests for name normalization."""

    def test_aten_op(self):
        assert normalize_trace_name("aten::mm::default") == "aten.mm"

    def test_tensor_cast_op(self):
        assert (
            normalize_trace_name("tensor_cast::attention::default")
            == "tensor_cast.attention"
        )

    def test_underscore_op(self):
        assert (
            normalize_trace_name("aten::_to_copy::default") == "aten._to_copy"
        )

    def test_no_colons(self):
        assert (
            normalize_trace_name("_internal_mark_region_begin")
            == "_internal_mark_region_begin"
        )

    def test_two_parts(self):
        assert normalize_trace_name("aten::silu") == "aten.silu"

    def test_empty_string(self):
        assert normalize_trace_name("") == ""


class TestTraceEvent:
    """Tests for TraceEvent dataclass."""

    def test_basic_creation(self):
        event = TraceEvent(
            name="aten.mm",
            duration_us=100.0,
            start_us=1000.0,
        )
        assert event.name == "aten.mm"
        assert event.duration_us == 100.0
        assert event.start_us == 1000.0
        assert event.args == {}

    def test_with_args(self):
        event = TraceEvent(
            name="aten.mm",
            duration_us=50.0,
            start_us=0.0,
            args={"shape": "[136, 4096]"},
        )
        assert event.args["shape"] == "[136, 4096]"


class TestParseChrome:
    """Tests for parse_chrome_trace function."""

    def _write_trace(self, events):
        """Write a chrome trace JSON to a temp file."""
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump({"traceEvents": events}, f)
        f.flush()
        return Path(f.name)

    def test_parse_basic(self):
        path = self._write_trace([
            {"ph": "X", "name": "aten::mm::default", "ts": 100, "dur": 50},
            {"ph": "X", "name": "aten::add::default", "ts": 200, "dur": 30},
        ])
        events = parse_chrome_trace(path)
        assert len(events) == 2
        assert events[0].name == "aten.mm"
        assert events[0].start_us == 100
        assert events[0].duration_us == 50
        assert events[1].name == "aten.add"

    def test_sorted_by_start(self):
        path = self._write_trace([
            {"ph": "X", "name": "op_b", "ts": 200, "dur": 10},
            {"ph": "X", "name": "op_a", "ts": 100, "dur": 20},
        ])
        events = parse_chrome_trace(path)
        assert events[0].name == "op_a"
        assert events[1].name == "op_b"

    def test_filters_non_x_events(self):
        path = self._write_trace([
            {"ph": "X", "name": "real_op", "ts": 100, "dur": 50},
            {"ph": "B", "name": "begin_op", "ts": 100},
            {"ph": "E", "name": "end_op", "ts": 150},
            {"ph": "M", "name": "metadata", "ts": 0},
        ])
        events = parse_chrome_trace(path)
        assert len(events) == 1
        assert events[0].name == "real_op"

    def test_handles_args(self):
        path = self._write_trace([
            {
                "ph": "X",
                "name": "aten::mm::default",
                "ts": 0,
                "dur": 100,
                "args": {"input_shapes": "[[136, 4096], [4096, 5120]]"},
            },
        ])
        events = parse_chrome_trace(path)
        assert "input_shapes" in events[0].args

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_chrome_trace(Path("/nonexistent/trace.json"))

    def test_list_format(self):
        """Test parsing when trace is a plain list instead of dict."""
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(
            [{"ph": "X", "name": "op1", "ts": 0, "dur": 10}],
            f,
        )
        f.flush()
        events = parse_chrome_trace(Path(f.name))
        assert len(events) == 1

    def test_empty_trace(self):
        path = self._write_trace([])
        events = parse_chrome_trace(path)
        assert len(events) == 0

    def test_skips_empty_name(self):
        path = self._write_trace([
            {"ph": "X", "name": "", "ts": 0, "dur": 10},
            {"ph": "X", "name": "real", "ts": 10, "dur": 5},
        ])
        events = parse_chrome_trace(path)
        assert len(events) == 1
        assert events[0].name == "real"
