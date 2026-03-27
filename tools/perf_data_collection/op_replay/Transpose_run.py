"""
Replay Transpose cases from the performance database on Ascend NPU.

Purpose:
  Read Transpose rows, infer the transposed dimensions from the recorded
  input/output shapes, then execute torch.transpose().contiguous().
"""

from __future__ import annotations

try:
    from .common import parse_list_field, parse_shape
    from .replay_framework import OpReplay
except ImportError:
    from common import parse_list_field, parse_shape
    from replay_framework import OpReplay


def infer_transpose_dims(input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> tuple[int, int]:
    diff_dims = [index for index, dims in enumerate(zip(input_shape, output_shape)) if dims[0] != dims[1]]
    if len(diff_dims) >= 2:
        return diff_dims[0], diff_dims[1]
    if len(input_shape) >= 2:
        return 0, 1
    return 0, 0


def build_case(row: dict[str, str]):
    inputs = op.build_inputs(row)
    if not inputs:
        raise ValueError("Transpose requires at least one recorded input tensor")

    output_shapes = [parse_shape(item) for item in parse_list_field(row["Output Shapes"])]
    if not output_shapes:
        raise ValueError("Transpose requires at least one recorded output shape")

    dims = infer_transpose_dims(tuple(inputs[0].shape), output_shapes[0])
    return {
        "inputs": [inputs[0]],
        "kwargs": {"dim0": dims[0], "dim1": dims[1]},
        "api": op.resolve_api(),
    }


def run_case(case):
    tensor = case["inputs"][0]
    return case["api"](tensor, case["kwargs"]["dim0"], case["kwargs"]["dim1"]).contiguous()


def format_success(csv_path, row_index: int, row: dict[str, str], case, result) -> str:
    return (
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} output={tuple(result.shape)} "
        f"dims=({case['kwargs']['dim0']}, {case['kwargs']['dim1']})"
    )


op = OpReplay(
    kernel_type="Transpose",
    api_path="torch.transpose",
    description=(
        "Run Transpose workload replay on Ascend NPU.\n"
        "The script reads Transpose.csv under the selected device and\n"
        "vllm_ascend version directory, infers the transpose dimensions\n"
        "from the recorded input/output shapes, then runs\n"
        "torch.transpose(input, dim0, dim1).contiguous()."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/Transpose_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.16.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.16.0.",
    input_count=1,
    build_case=build_case,
    run_case=run_case,
    format_success=format_success,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
