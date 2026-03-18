from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch_npu

FRACTAL_NZ_FORMAT_ID = 29
# common.py → op_replay/ [0] → perf_data_collection/ [1] → tools/ [2] → repo_root [3]
DATA_DIR = (
    Path(__file__).resolve().parents[3]
    / "tensor_cast"
    / "performance_model"
    / "perf_database"
    / "data"
)
SUPPORTED_DEVICES = [
    "TEST_DEVICE",
    "ATLAS_800_A2_376T_64G",
    "ATLAS_800_A2_313T_64G",
    "ATLAS_800_A2_280T_64G",
    "ATLAS_800_A2_280T_64G_PCIE",
    "ATLAS_800_A2_280T_32G_PCIE",
    "ATLAS_800_A3_752T_128G_DIE",
    "ATLAS_800_A3_560T_128G_DIE",
]

torch = None
torch_npu = None
DTYPE_MAP = {}


def init_runtime() -> None:
    global torch
    global torch_npu
    global DTYPE_MAP

    if torch is not None and torch_npu is not None and DTYPE_MAP:
        return

    try:
        import torch as torch_module
        import torch_npu as torch_npu_module
    except ImportError as exc:
        raise RuntimeError("未找到NPU") from exc

    try:
        from vllm_ascend.utils import enable_custom_op
        enable_custom_op()
    except Exception as exc:
        print(f"Warning: 未找到自定义算子依赖 ({exc})。如果脚本中包含自定义算子，可能会运行失败。")

    torch = torch_module
    torch_npu = torch_npu_module
    torch_npu.npu.config.allow_internal_format = True
    DTYPE_MAP = {
        "DT_FLOAT": torch.float32,
        "DT_FLOAT16": torch.float16,
        "DT_BF16": torch.bfloat16,
        "DT_DOUBLE": torch.float64,
        "DT_INT8": torch.int8,
        "DT_UINT8": torch.uint8,
        "DT_INT16": torch.int16,
        "DT_INT32": torch.int32,
        "DT_INT64": torch.int64,
        "DT_BOOL": torch.bool,
    }


def get_runtime_modules():
    init_runtime()
    return torch, torch_npu


def check_version(value: str) -> str:
    version = value.strip()
    if not re.fullmatch(r"[0-9]+(?:\.[0-9A-Za-z_-]+)*", version):
        raise argparse.ArgumentTypeError(
            f"Invalid --vllm-ascend-version: {value!r}. Expected value like 0.9.2"
        )
    return version


def normalize_device_name(device: str) -> str:
    return device.strip()


def normalize_vllm_ascend_version(version: str) -> str:
    normalized = version.strip()
    if not normalized.startswith("v"):
        normalized = f"v{normalized}"
    return normalized


def ensure_npu_available() -> None:
    runtime_torch, _ = get_runtime_modules()
    has_npu = hasattr(runtime_torch, "npu") and runtime_torch.npu.is_available()
    if not has_npu:
        raise RuntimeError("未找到NPU")


def parse_list_field(raw_value: str) -> list[str]:
    cleaned = raw_value.strip().strip('"')
    return [item.strip() for item in cleaned.split(";") if item.strip()]


def parse_shape(raw_shape: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw_shape.split(",") if part.strip())


def expand_fractal_nz_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) != 4:
        raise ValueError(f"Unsupported FRACTAL_NZ shape: {shape}")
    a_dim, b_dim, c_dim, d_dim = shape
    return b_dim * c_dim, a_dim * d_dim


def normalize_shape(shape: tuple[int, ...], input_format: str) -> tuple[int, ...]:
    if input_format == "FRACTAL_NZ":
        return expand_fractal_nz_shape(shape)
    return shape


def build_host_tensor(shape: tuple[int, ...], dtype):
    runtime_torch, _ = get_runtime_modules()

    if dtype == runtime_torch.bool:
        return runtime_torch.randint(0, 2, shape, dtype=runtime_torch.int32).to(runtime_torch.bool)
    if dtype in {
        runtime_torch.float16,
        runtime_torch.bfloat16,
        runtime_torch.float32,
        runtime_torch.float64,
    }:
        return runtime_torch.randn(shape).to(dtype)
    return runtime_torch.randint(0, 8, shape, dtype=dtype)


def maybe_cast_internal_format(tensor, input_format: str):
    _, runtime_torch_npu = get_runtime_modules()
    if input_format == "FRACTAL_NZ":
        return runtime_torch_npu.npu_format_cast(tensor, FRACTAL_NZ_FORMAT_ID)
    return tensor


def build_input_tensor(
    shape: tuple[int, ...],
    input_format: str,
    dtype_name: str,
    transpose: bool = False,
):
    dtype = DTYPE_MAP.get(dtype_name)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype_name}")

    normalized_shape = normalize_shape(shape, input_format)
    tensor = build_host_tensor(normalized_shape, dtype)
    tensor = tensor.npu()
    if transpose:
        tensor = tensor.t()
    return maybe_cast_internal_format(tensor, input_format)


def get_target_data_dir(device: str, vllm_ascend_version: str) -> Path:
    return (
        DATA_DIR
        / normalize_device_name(device)
        / "vllm_ascend"
        / normalize_vllm_ascend_version(vllm_ascend_version)
    )


def iter_csv_rows(target_data_dir: Path, csv_name: str):
    for csv_path in sorted(target_data_dir.rglob(csv_name)):
        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row_index, row in enumerate(reader, start=2):
                yield csv_path, row_index, row


def build_standard_argparser(
    *,
    description: str,
    usage_examples: list[str],
    version_help: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=description,
        epilog=(
            "Usage examples:\n"
            + "\n".join(f"  {item}" for item in usage_examples)
            + "\n\nParameter notes:\n"
            + "  --device                Selects the device folder under perf_database/data.\n"
            + "  --vllm-ascend-version   Selects the version folder under {device}/vllm_ascend/.\n"
            + "  -h, --help              Show this help message and exit."
        ),
    )
    parser.add_argument(
        "--device",
        required=True,
        choices=SUPPORTED_DEVICES,
        help=(
            "Target device folder under "
            "tensor_cast/performance_model/perf_database/data/{device}/"
        ),
    )
    parser.add_argument(
        "--vllm-ascend-version",
        required=True,
        type=check_version,
        help=version_help,
    )
    return parser
