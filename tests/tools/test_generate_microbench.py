"""Tests for tools/perf_data_collection/generate_microbench.py (TCX production version)."""

import sys
from pathlib import Path


sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "tools" / "perf_data_collection")
)

from generate_microbench import (
    FALLBACK_API_BY_KERNEL,
    resolve_microbench_api,
    SUPPORTED_MICROBENCH_APIS,
)


class TestApiResolution:
    # resolve_microbench_api(reference_mapping, kernel_type) -- mapping first!

    def test_fallback_covers_common_kernels(self):
        expected = {"MatMulV3", "Mul", "Sub", "Fill", "Sort"}
        for kernel in expected:
            api = resolve_microbench_api({}, kernel)
            assert api is not None, f"No API resolved for {kernel}"

    def test_op_mapping_overrides_fallback(self):
        mapping = {"MatMulV3": {"microbench_api": "torch.custom_mm"}}
        api = resolve_microbench_api(mapping, "MatMulV3")
        assert api == "torch.custom_mm"

    def test_unknown_kernel_returns_none(self):
        api = resolve_microbench_api({}, "TotallyUnknownKernel")
        assert api is None


class TestSupportedApis:
    def test_common_apis_present(self):
        assert "torch.mm" in SUPPORTED_MICROBENCH_APIS
        assert "torch.add" in SUPPORTED_MICROBENCH_APIS

    def test_fallback_apis_are_supported(self):
        for kernel, api in FALLBACK_API_BY_KERNEL.items():
            assert api in SUPPORTED_MICROBENCH_APIS, (
                f"Fallback API {api!r} for {kernel} not in SUPPORTED_MICROBENCH_APIS"
            )
