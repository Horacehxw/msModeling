"""Tests for PR review fixes: kernel batch, allReduce overhead, error handling.

Covers:
  P0: kernel mode uses batch profiler (parse_fn parameter)
  P1: allReduce overhead correctly applied
  P1: profiler_batch raises on empty results
  P2: write_csv handles empty dirname
  P2: comment accuracy (not testable, verified by inspection)
"""

import csv
import inspect
import os
import tempfile

import pytest


# ---------------------------------------------------------------------------
# P0: _run_bench_profiler_batch accepts parse_fn parameter
# ---------------------------------------------------------------------------

def test_profiler_batch_accepts_parse_fn():
    """_run_bench_profiler_batch should accept a parse_fn keyword argument."""
    from tools.perf_data_collection.generate_comm_microbench import (
        _run_bench_profiler_batch,
    )
    sig = inspect.signature(_run_bench_profiler_batch)
    assert "parse_fn" in sig.parameters, (
        "_run_bench_profiler_batch must accept parse_fn parameter "
        "to support kernel_details parsing in batch mode"
    )
    param = sig.parameters["parse_fn"]
    assert param.default is None, "parse_fn should default to None"


def test_kernel_branch_exists_in_main():
    """main() should have a dedicated kernel branch, not fall through to else."""
    from tools.perf_data_collection import generate_comm_microbench as mod
    source = inspect.getsource(mod.main)
    # The fix adds: elif args.bench_mode == "kernel"
    assert 'bench_mode == "kernel"' in source, (
        "main() must have an explicit kernel branch to avoid "
        "per-point profiler restart (CANN constraint)"
    )


def test_kernel_batch_uses_kernel_parse_fn():
    """The kernel branch in main() should pass _parse_kernel_comm_duration as parse_fn."""
    from tools.perf_data_collection import generate_comm_microbench as mod
    source = inspect.getsource(mod.main)
    assert "_parse_kernel_comm_duration" in source, (
        "kernel batch branch must use _parse_kernel_comm_duration "
        "to parse kernel_details.csv instead of operator_details.csv"
    )


# ---------------------------------------------------------------------------
# P1: allReduce overhead correctly applied
# ---------------------------------------------------------------------------

def test_allreduce_overhead_applied_nd16():
    """build_allreduce should add 7.7us overhead for nd=16."""
    from tools.perf_data_collection.build_comm_csv import build_allreduce
    rows = [{"message_bytes": 160000, "num_devices": 16, "dtype": "DT_BF16",
             "topology_tier": 1, "Duration(us)": 12.1, "bandwidth_gbps": 0.0}]
    result = build_allreduce(rows)
    assert len(result) == 1
    assert abs(result[0]["Duration(us)"] - 19.8) < 0.1, (
        f"nd=16 allReduce should be 12.1 + 7.7 = 19.8, got {result[0]['Duration(us)']}"
    )


def test_allreduce_no_overhead_nd2():
    """build_allreduce should NOT add overhead for nd=2 (no entry in OVERHEAD)."""
    from tools.perf_data_collection.build_comm_csv import build_allreduce
    rows = [{"message_bytes": 160000, "num_devices": 2, "dtype": "DT_BF16",
             "topology_tier": 1, "Duration(us)": 50.0, "bandwidth_gbps": 0.0}]
    result = build_allreduce(rows)
    assert result[0]["Duration(us)"] == 50.0, (
        "nd=2 allReduce should have no overhead correction"
    )


def test_allreduce_immutability():
    """build_allreduce should not mutate input rows."""
    from tools.perf_data_collection.build_comm_csv import build_allreduce
    rows = [{"message_bytes": 160000, "num_devices": 16, "dtype": "DT_BF16",
             "topology_tier": 1, "Duration(us)": 12.1, "bandwidth_gbps": 0.0}]
    original_dur = rows[0]["Duration(us)"]
    build_allreduce(rows)
    assert rows[0]["Duration(us)"] == original_dur, "Input rows must not be mutated"


# ---------------------------------------------------------------------------
# P1: allGather/reduceScatter overhead for kernel-mode small messages
# ---------------------------------------------------------------------------

def test_allgather_kernel_overhead_nd16():
    """build_allgather should add 14.6us overhead for nd=16 small messages from kernel."""
    from tools.perf_data_collection.build_comm_csv import build_allgather
    kern_rows = [{"message_bytes": 65536, "num_devices": 16, "dtype": "DT_BF16",
                  "topology_tier": 1, "Duration(us)": 10.0, "bandwidth_gbps": 0.0}]
    result = build_allgather(alt_rows=[], kern_rows=kern_rows, profiler_p50=0.0)
    assert len(result) == 1
    assert abs(result[0]["Duration(us)"] - 24.6) < 0.1, (
        f"nd=16 small allGather should be 10.0 + 14.6 = 24.6, got {result[0]['Duration(us)']}"
    )


def test_allgather_alternating_no_overhead_large():
    """build_allgather should NOT add overhead for large messages from alternating."""
    from tools.perf_data_collection.build_comm_csv import build_allgather
    alt_rows = [{"message_bytes": 2_000_000, "num_devices": 16, "dtype": "DT_BF16",
                 "topology_tier": 1, "Duration(us)": 300.0, "bandwidth_gbps": 0.0}]
    result = build_allgather(alt_rows=alt_rows, kern_rows=[], profiler_p50=0.0)
    assert len(result) == 1
    assert result[0]["Duration(us)"] == 300.0, (
        "Large message alternating data should have no overhead"
    )


def test_allgather_768kb_nd8_uses_profiler_p50():
    """build_allgather should use profiler P50 for 768KB nd=8."""
    from tools.perf_data_collection.build_comm_csv import build_allgather
    kern_rows = [{"message_bytes": 786432, "num_devices": 8, "dtype": "DT_BF16",
                  "topology_tier": 1, "Duration(us)": 156.4, "bandwidth_gbps": 0.0}]
    result = build_allgather(alt_rows=[], kern_rows=kern_rows, profiler_p50=75.4)
    assert len(result) == 1
    assert result[0]["Duration(us)"] == 75.4, (
        f"768KB nd=8 should use profiler P50=75.4, got {result[0]['Duration(us)']}"
    )


def test_reducescatter_kernel_overhead_nd8():
    """build_reducescatter should add 2.0us overhead for nd=8 small messages."""
    from tools.perf_data_collection.build_comm_csv import build_reducescatter
    kern_rows = [{"message_bytes": 14336, "num_devices": 8, "dtype": "DT_BF16",
                  "topology_tier": 1, "Duration(us)": 6.1, "bandwidth_gbps": 0.0}]
    result = build_reducescatter(alt_rows=[], kern_rows=kern_rows)
    assert len(result) == 1
    assert abs(result[0]["Duration(us)"] - 8.1) < 0.1, (
        f"nd=8 small reduceScatter should be 6.1 + 2.0 = 8.1, got {result[0]['Duration(us)']}"
    )


# ---------------------------------------------------------------------------
# P2: write_csv handles empty dirname
# ---------------------------------------------------------------------------

def test_write_csv_no_dirname():
    """write_csv should handle path without directory prefix."""
    from tools.perf_data_collection.build_comm_csv import write_csv
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use a bare filename (no directory prefix)
        old_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            write_csv([], "test_output.csv")
            assert os.path.exists("test_output.csv")
            with open("test_output.csv") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 0
        finally:
            os.chdir(old_cwd)


def test_write_csv_with_dirname():
    """write_csv should create parent directories."""
    from tools.perf_data_collection.build_comm_csv import write_csv
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sub", "dir", "output.csv")
        write_csv([], path)
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# P1: profiler_batch raises on empty parse results
# ---------------------------------------------------------------------------

def test_profiler_batch_raises_on_empty():
    """_run_bench_profiler_batch should raise RuntimeError when parse returns empty.

    We verify this by checking the source code contains the raise statement,
    since actually running the function requires NPU hardware.
    """
    from tools.perf_data_collection import generate_comm_microbench as mod
    source = inspect.getsource(mod._run_bench_profiler_batch)
    assert "raise RuntimeError" in source, (
        "_run_bench_profiler_batch must raise RuntimeError when "
        "parse returns empty results (not silently drop data)"
    )
    # Verify it checks 'if not durations'
    assert "if not durations:" in source, (
        "Should check 'if not durations:' before raising"
    )
