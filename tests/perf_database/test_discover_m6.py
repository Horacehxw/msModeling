"""Test M6 key in discover_operators output."""

import sys
from pathlib import Path

import yaml

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "tools" / "perf_data_collection")
)
from discover_operators import discover_operators


def test_m6_key_in_coverage_dict(tmp_path):
    """discover_operators should include m6_empirical_prediction_coverage in coverage."""
    csv_path = tmp_path / "kernel_details.csv"
    csv_path.write_text(
        "Type,Duration(us)\n"
        "MatMulV2,50.0\n"
        "MatMulV2,50.0\n"
        "SwiGlu,20.0\n"
        "UnknownKernel,30.0\n"
    )

    op_mapping = {
        "operator_mappings": {
            "aten.mm.default": {"kernel_type": "MatMulV2"},
            "tensor_cast.swiglu.default": {"kernel_type": "SwiGlu"},
        }
    }
    yaml_path = tmp_path / "op_mapping.yaml"
    yaml_path.write_text(yaml.dump(op_mapping))

    result = discover_operators(csv_path, yaml_path)
    cov = result["coverage"]

    assert "m6_empirical_prediction_coverage" in cov
    # MatMulV2 (100us) + SwiGlu (20us) = 120, total = 150 → 80%
    assert abs(cov["m6_empirical_prediction_coverage"] - 0.8) < 1e-9
    assert abs(cov["duration_coverage_pct"] - 80.0) < 1e-9
