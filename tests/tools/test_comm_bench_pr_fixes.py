"""Tests for PR review fixes: kernel batch and error handling.

Covers:
  P0: kernel mode uses batch profiler (parse_fn parameter)
  P1: profiler_batch returns empty on no durations (tolerant error handling)
"""

import inspect


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
# P1: profiler_batch returns empty on no durations (tolerant error handling)
# ---------------------------------------------------------------------------

def test_profiler_batch_returns_empty_on_no_durations():
    """_run_bench_profiler_batch should return empty dict when parse returns empty.

    Changed from raising RuntimeError to returning {} for better error tolerance
    -- individual session failures should not abort the entire collection run.
    """
    from tools.perf_data_collection import generate_comm_microbench as mod
    source = inspect.getsource(mod._run_bench_profiler_batch)
    assert "return {}" in source, (
        "_run_bench_profiler_batch must return empty dict when "
        "parse returns empty results (tolerant error handling)"
    )
    assert "if not durations:" in source, (
        "Should check 'if not durations:' before returning empty"
    )


def test_profiler_batch_accepts_no_sync():
    """_run_bench_profiler_batch should accept a no_sync keyword argument."""
    from tools.perf_data_collection.generate_comm_microbench import (
        _run_bench_profiler_batch,
    )
    sig = inspect.signature(_run_bench_profiler_batch)
    assert "no_sync" in sig.parameters, (
        "_run_bench_profiler_batch must accept no_sync parameter "
        "to skip synchronize() between iterations"
    )
    param = sig.parameters["no_sync"]
    assert param.default is False, "no_sync should default to False"
