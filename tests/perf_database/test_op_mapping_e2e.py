"""End-to-end test: verify op_mapping.yaml + shape matching for all 49 dispatched ops.

Uses stub CSVs generated from TC trace data (generate_stub_csvs.py) to verify
that ProfilingDataSource.lookup() can find shape matches for every non-zero-cost,
non-communication, non-attention-special op that appears in the 4 simulation configs:
  - Qwen3-32B Prefill/Decode (BF16, TP=16)
  - DeepSeek-V3 Prefill/Decode (W8A8, TP=4/DP=8/EP)

This tests the full pipeline: op_name → op_mapping.yaml → kernel_type → CSV shape match.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest
import torch
import yaml

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
TRACE_DIR = Path(__file__).parent / "fixtures" / "traces"

OP_MAPPING_PATH = (
    PROJECT_ROOT
    / "tensor_cast/performance_model/perf_database/data"
    / "ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5"
    / "op_mapping.yaml"
)


@pytest.fixture(scope="module")
def stub_data_dir():
    """Generate stub CSVs in a temp directory for testing."""
    if not OP_MAPPING_PATH.exists():
        pytest.skip("CANN 8.5 op_mapping.yaml not found")

    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "tools" / "perf_data_collection"))
    from generate_stub_csvs import generate_stub_csvs

    with tempfile.TemporaryDirectory(prefix="stub_csvs_") as tmpdir:
        report = generate_stub_csvs(TRACE_DIR, OP_MAPPING_PATH, Path(tmpdir))
        yield Path(tmpdir), report


@pytest.fixture(scope="module")
def data_source(stub_data_dir):
    """Create ProfilingDataSource with stub CSVs."""
    from tensor_cast.performance_model.perf_database.profiling_data_source import (
        ProfilingDataSource,
    )

    data_dir, _ = stub_data_dir
    return ProfilingDataSource(data_dir=data_dir)


@pytest.fixture(scope="module")
def all_trace_ops() -> Dict[str, List[dict]]:
    """Load all ops from all 4 trace files."""
    ops = {}
    for trace_file in sorted(TRACE_DIR.glob("*.json")):
        with open(trace_file) as f:
            data = json.load(f)
        for op in data["ops"]:
            op_name = op["op_name"]
            if op_name not in ops:
                ops[op_name] = {"variants": [], "traces": []}
            ops[op_name]["traces"].append(trace_file.stem)
            for v in op.get("shape_variants", []):
                ops[op_name]["variants"].append(v)
    return ops


@pytest.fixture(scope="module")
def op_mapping():
    """Load op_mapping.yaml."""
    if not OP_MAPPING_PATH.exists():
        pytest.skip("CANN 8.5 op_mapping.yaml not found")
    with open(OP_MAPPING_PATH) as f:
        return yaml.safe_load(f)


class _MockFunc:
    """Mock torch op func that returns the right string for _normalize_func_name."""

    def __init__(self, op_name: str):
        self._name = f"torch.ops.{op_name}"

    def __str__(self):
        return self._name

    def __repr__(self):
        return self._name


def _make_mock_op_invoke(
    op_name: str, input_shapes: List[str], input_dtypes: List[str]
) -> MagicMock:
    """Create a mock OpInvokeInfo from trace data."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "int8": torch.int8,
        "int32": torch.int32,
        "int64": torch.int64,
        "float32": torch.float32,
        "bool": torch.bool,
    }

    tensors = []
    for i, shape_str in enumerate(input_shapes):
        shape_str = shape_str.strip().strip("()")
        if not shape_str or shape_str == "()":
            # Scalar — skip (profiling _parse_shape_str also drops empty shapes)
            continue
        dims = tuple(int(x.strip()) for x in shape_str.split(",") if x.strip())
        # Get dtype, defaulting based on op semantics
        dt_str = input_dtypes[i] if i < len(input_dtypes) and input_dtypes[i] else ""
        if dt_str:
            dt = dtype_map.get(dt_str, torch.bfloat16)
        elif "where" in op_name and i == 0:
            dt = torch.bool
        elif "bitwise" in op_name:
            dt = torch.bool
        else:
            dt = torch.bfloat16
        t = torch.empty(dims, dtype=dt, device="meta")
        tensors.append(t)

    mock = MagicMock()
    mock.func = _MockFunc(op_name)
    mock.args = tuple(tensors)
    mock.kwargs = {}
    return mock


def _is_skippable(config: Optional[dict]) -> Optional[str]:
    """Return skip reason if op doesn't need CSV match, None otherwise."""
    if config is None:
        return "unmapped"
    if config.get("zero_cost"):
        return "zero_cost"
    if config.get("category") == "communication":
        return "communication"
    if config.get("query_mode") == "attention_special":
        return "attention_special"
    # Composite ops containing hcom sub-kernels can't be fully validated
    # with stub CSVs (comm lookup requires message_bytes format, not shapes)
    if config.get("composite"):
        sub_kernels = config.get("sub_kernels", [])
        if any(sk.startswith("hcom_") for sk in sub_kernels):
            return "composite_with_comm"
    return None


class TestOpMappingE2E:
    """End-to-end tests: every non-skip op should get a lookup hit."""

    def test_stub_csvs_generated(self, stub_data_dir):
        """Verify stub CSVs were generated."""
        data_dir, report = stub_data_dir
        assert len(report["generated"]) > 0
        csv_files = list(data_dir.glob("*.csv"))
        assert len(csv_files) >= 25, f"Expected ≥25 CSV stubs, got {len(csv_files)}"

    def test_all_kernel_types_have_csvs(self, stub_data_dir, op_mapping, all_trace_ops):
        """Every kernel_type referenced by dispatched ops should have a stub CSV."""
        data_dir, _ = stub_data_dir
        entries = op_mapping.get("operator_mappings", {})

        missing = []
        for op_name in all_trace_ops:
            config = entries.get(op_name)
            if _is_skippable(config):
                continue
            if config.get("composite"):
                for sk in config.get("sub_kernels", []):
                    if sk.startswith("hcom_"):
                        continue
                    if not (data_dir / f"{sk}.csv").exists():
                        missing.append(f"{op_name} -> {sk}")
            else:
                # csv_file overrides kernel_type for CSV filename
                kt = config.get("csv_file", config["kernel_type"])
                if not (data_dir / f"{kt}.csv").exists():
                    missing.append(f"{op_name} -> {kt}")

        assert missing == [], f"Missing CSV stubs: {missing}"

    def test_zero_cost_ops_return_zero(self, data_source, op_mapping, all_trace_ops):
        """Zero-cost ops should return latency=0."""
        entries = op_mapping.get("operator_mappings", {})
        for op_name, info in all_trace_ops.items():
            config = entries.get(op_name)
            if not config or not config.get("zero_cost"):
                continue
            variant = info["variants"][0]
            mock = _make_mock_op_invoke(
                op_name, variant["input_shapes"], variant["input_dtypes"]
            )
            result = data_source.lookup(mock)
            assert result is not None, f"zero_cost op {op_name} returned None"
            assert result.latency_us == 0.0, f"zero_cost op {op_name} latency != 0"

    def test_communication_ops_return_none(
        self, data_source, op_mapping, all_trace_ops
    ):
        """Communication ops should return None (fallback to analytic)."""
        entries = op_mapping.get("operator_mappings", {})
        for op_name, info in all_trace_ops.items():
            config = entries.get(op_name)
            if not config or config.get("category") != "communication":
                continue
            variant = info["variants"][0]
            mock = _make_mock_op_invoke(
                op_name, variant["input_shapes"], variant["input_dtypes"]
            )
            # Comm ops need rank + rank_group as last args for _lookup_comm
            mock.args = (*mock.args, 0, list(range(16)))
            result = data_source.lookup(mock)
            assert result is None, f"communication op {op_name} should return None"

    def test_compute_ops_shape_match(self, data_source, op_mapping, all_trace_ops):
        """Every compute op variant should find a shape match in stub CSVs."""
        entries = op_mapping.get("operator_mappings", {})
        hits = []
        misses = []

        for op_name, info in sorted(all_trace_ops.items()):
            config = entries.get(op_name)
            skip = _is_skippable(config)
            if skip:
                continue

            # Deduplicate variants by shape
            seen = set()
            for variant in info["variants"]:
                key = (tuple(variant["input_shapes"]), tuple(variant["input_dtypes"]))
                if key in seen:
                    continue
                seen.add(key)

                mock = _make_mock_op_invoke(
                    op_name, variant["input_shapes"], variant["input_dtypes"]
                )
                result = data_source.lookup(mock)
                if result is not None:
                    hits.append(op_name)
                else:
                    misses.append(
                        {
                            "op": op_name,
                            "shapes": variant["input_shapes"],
                            "dtypes": variant["input_dtypes"],
                            "kernel_type": config.get("kernel_type", "composite"),
                        }
                    )

        # Report
        total = len(hits) + len(misses)
        hit_rate = len(hits) / total * 100 if total > 0 else 0
        print(f"\n  Shape matching: {len(hits)}/{total} hits ({hit_rate:.1f}%)")
        if misses:
            print("  MISSES:")
            for m in misses:
                print(f"    {m['op']} -> {m['kernel_type']}: shapes={m['shapes']}")

        assert misses == [], f"{len(misses)} ops failed shape matching: " + ", ".join(
            m["op"] for m in misses
        )


class TestPerConfigCoverage:
    """Verify coverage per simulation config."""

    @pytest.mark.parametrize(
        "trace_name",
        ["qwen3_32b_prefill", "qwen3_32b_decode", "dsv3_prefill", "dsv3_decode"],
    )
    def test_config_full_coverage(self, trace_name, data_source, op_mapping):
        """Each config should have 100% coverage for non-skip ops."""
        entries = op_mapping.get("operator_mappings", {})
        trace_file = TRACE_DIR / f"{trace_name}.json"
        assert trace_file.exists(), (
            f"Fixture trace {trace_name}.json missing from "
            f"tests/perf_database/fixtures/traces/ — regenerate with TC analytic+compile"
        )

        with open(trace_file) as f:
            data = json.load(f)

        hits = 0
        misses = 0
        skip = 0

        for op in data["ops"]:
            op_name = op["op_name"]
            config = entries.get(op_name)
            if _is_skippable(config):
                skip += 1
                continue

            variant = op["shape_variants"][0]
            mock = _make_mock_op_invoke(
                op_name, variant["input_shapes"], variant["input_dtypes"]
            )
            result = data_source.lookup(mock)
            if result is not None:
                hits += 1
            else:
                misses += 1
                print(f"  MISS in {trace_name}: {op_name}")

        total = hits + misses
        print(f"\n  {trace_name}: {hits}/{total} compute hits, {skip} skipped")
        assert misses == 0, f"{trace_name}: {misses} misses out of {total} compute ops"
