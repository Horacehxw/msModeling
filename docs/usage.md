# Usage Guide

This guide covers installation, running simulations, and interpreting results for both TensorCast and ServingCast.

---

## 1. Prerequisites

- **Python**: 3.10 or later (project requires >= 3.9, but 3.10+ is recommended).
- **PyTorch**: 2.7 – 2.8 (installed via `requirements.txt`).
- **HuggingFace access**: some models require authentication. Set `HF_TOKEN` or run `huggingface-cli login` if needed.

## 2. Installation

```bash
git clone https://gitcode.com/Ascend/msmodeling.git -b develop
cd msmodeling

# TensorCast dependencies
pip install -r ./tensor_cast/requirements.txt

# ServingCast dependencies (also install TensorCast deps first)
pip install -r ./serving_cast/requirements.txt

# Set PYTHONPATH so modules can find each other
export PYTHONPATH=$(pwd):$PYTHONPATH
```

---

## 3. TensorCast — Single-Request Simulation

### 3.1 Text Generation (`text_generate`)

Simulate one or more LLM inference passes (prefill or decode) and view an operator-level performance breakdown.

**Basic prefill simulation:**

```bash
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 2 \
  --query-length 3500 \
  --device TEST_DEVICE
```

**Decode simulation with context and quantization:**

```bash
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 10 \
  --query-length 1 \
  --context-length 4500 \
  --device TEST_DEVICE \
  --decode \
  --quantize-linear-action W8A8_DYNAMIC
```

**With tensor parallelism and Chrome Trace output:**

```bash
python -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 4 \
  --query-length 3500 \
  --device ATLAS_800_A2_280T_64G \
  --world-size 8 \
  --tp-size 8 \
  --quantize-linear-action W8A8_DYNAMIC \
  --chrome-trace trace_output.json
```

Open `trace_output.json` in `chrome://tracing` (Chromium-based browsers) for a visual timeline.

#### Key CLI Arguments

| Argument | Description | Default |
|---|---|---|
| `model_id` | HuggingFace model ID (positional) | Required |
| `--num-queries` | Number of requests in the batch | Required |
| `--query-length` | New input tokens per request | Required |
| `--context-length` | Total context length (prefix + query) | None |
| `--device` | Target device profile name | `TEST_DEVICE` |
| `--decode` | Run in decode mode (vs. prefill) | False |
| `--compile` | Apply `torch.compile()` graph optimization | False |
| `--quantize-linear-action` | Quantization scheme for linear layers | `DISABLED` |
| `--quantize-attention-action` | KV cache quantization | `DISABLED` |
| `--world-size` | Total number of devices | 1 |
| `--tp-size` | Tensor parallelism degree | 1 |
| `--dp-size` | Data parallelism degree | None (auto) |
| `--ep` | Enable expert parallelism (for MoE) | False |
| `--num-mtp-tokens` | Multi-token prediction tokens (0 = disabled) | 0 |
| `--chrome-trace` | Path for Chrome Trace JSON output | None |
| `--reserved-memory-gb` | Memory reserved for system use | 0 |
| `--remote-source` | Model source: `huggingface` or `modelscope` | `huggingface` |

Run `python -m tensor_cast.scripts.text_generate --help` for the full argument list.

### 3.2 Benchmark (`benchmark`)

Search for optimal throughput under TTFT and TPOT service-level constraints.

**Single model/device benchmark:**

```bash
python -m tensor_cast.scripts.benchmark \
  --model-id Qwen/Qwen3-235B-A22B \
  --device ATLAS_800_A2_280T_64G \
  --num-devices 16 \
  --input-length 3500 \
  --output-length 1500 \
  --num-mtp-tokens 1 \
  --ttft-limits 3 \
  --tpot-limits 0.1 0.05
```

**Multi-model/device benchmark via YAML config:**

Create a config file (`benchmark_config.yaml`):

```yaml
devices:
  - "ATLAS_800_A2_280T_64G"
  - "H20"

models:
  "Qwen/Qwen3-32B": [1, 2, 4]       # device counts to test
  "zai-org/GLM-4.5": [4, 8]
```

```bash
python -m tensor_cast.scripts.benchmark \
  -c benchmark_config.yaml \
  --input-length 3500 \
  --output-length 1500 \
  --tpot-limits 0.05
```

#### Key Benchmark Arguments

| Argument | Description | Default |
|---|---|---|
| `--input-length` | Prompt token length | Required |
| `--output-length` | Expected output token length | Required |
| `--model-id` | HuggingFace model ID | None |
| `--device` | Target device | None |
| `--num-devices` | Number of devices | 1 |
| `-c` / `--config` | YAML config for batch benchmarking | None |
| `--ttft-limits` | TTFT upper bounds (seconds) | inf |
| `--tpot-limits` | TPOT upper bounds (seconds) | [50.0] |
| `--tp-sizes` | TP sizes to search over | Auto |
| `--concurrency-range` | Concurrency range to search | Auto |
| `--backend` | Serving backend name | MindIE |
| `--disaggregation` | Use P/D disaggregation mode | False |

### 3.3 Video Generation (`video_generate`)

Simulate video generation model performance (e.g., HunyuanVideo):

```bash
python -m tensor_cast.scripts.video_generate <model_id> [options]
```

Run with `--help` for available options.

---

## 4. ServingCast — End-to-End Serving Simulation

### 4.1 Configuration Files

ServingCast requires two YAML configuration files:

#### Instance Configuration (`instances.yaml`)

Defines the compute infrastructure — how many instances, what devices, parallelism, and communication properties:

```yaml
instance_groups:
  - num_instances: 1
    num_devices_per_instance: 1
    device_type: TEST_DEVICE
    pd_role: both                  # "prefill", "decode", or "both"
    parallel_config:
      world_size: 1
      tp_size: 1
      dp_size: 1
      ep: false
    communication_config:
      host2device_bandwidth: 10000000000    # 10 GB/s
      host2device_rate: 1.0
      device2device_bandwidth: 4000000000   # 4 GB/s
      device2device_rate: 1.0
```

For P/D disaggregation, define separate instance groups with `pd_role: prefill` and `pd_role: decode`.

#### Common Configuration (`common.yaml`)

Defines the model, workload, and serving parameters:

```yaml
model_config:
  name: Qwen/Qwen3-32B
  num_mtp_tokens: 0
  quantize_linear_action: W8A8_DYNAMIC
  quantize_lmhead: false
  do_compile: false
  enable_kv_transfer_modeling: true

load_gen:
  load_gen_type: fixed_length
  num_requests: 500
  num_input_tokens: 3500
  num_output_tokens: 1500
  request_rate: 2.0               # requests per second

serving_config:
  max_concurrency: 100
  block_size: 128                  # KV cache block size in tokens
  max_tokens_budget: 8192          # max tokens per prefill batch
```

### 4.2 Running the Simulation

**Basic run:**

```bash
python serving_cast/main.py \
  --instance_config_path=./serving_cast/example/instances.yaml \
  --common_config_path=./serving_cast/example/common.yaml
```

**With profiling enabled:**

```bash
python serving_cast/main.py \
  --instance_config_path=./serving_cast/example/instances.yaml \
  --common_config_path=./serving_cast/example/common.yaml \
  --enable_profiling \
  --profiling_output_path=./profiling_results
```

Profiling produces:
- `chrome_tracing.json` — viewable in `chrome://tracing`.
- `profiler.db` — viewable in MindStudio Insight.

### 4.3 Interpreting Results

After simulation, a summary is printed:

```
         E2E_TIME(s)  TTFT(s)  TPOT(s)  INPUT_TOKENS  OUTPUT_TOKENS  OUTPUT_TOKEN_THROUGHPUT(tok/s)
AVERAGE     1052.591    0.378    0.301        1500.0         3500.0                           3.327
MIN         1050.000    0.300    0.300        1500.0         3500.0                           2.978
MAX         1175.500    0.600    0.336        1500.0         3500.0                           3.334
MEDIAN      1050.100    0.400    0.300        1500.0         3500.0                           3.334
P75         1050.125    0.400    0.300        1500.0         3500.0                           3.334
P90         1050.200    0.500    0.300        1500.0         3500.0                           3.334
P99         1175.500    0.600    0.336        1500.0         3500.0                           3.334
======== Overall Summary ========
benchmark_duration(s)          1225.500
total_requests                 100.000
request_throughput(req/s)      0.082
total_input_tokens             150000.000
input_token_throughput(tok/s)  122.399
total_output_tokens            350000.000
output_token_throughput(tok/s) 285.598
```

| Metric | Meaning |
|---|---|
| **E2E_TIME** | Wall-clock time from request arrival to final output token |
| **TTFT** | Time from request arrival to first output token (includes queuing + prefill) |
| **TPOT** | Average time between successive output tokens |
| **OUTPUT_TOKEN_THROUGHPUT** | Per-request output token generation rate |
| **request_throughput** | System-wide completed requests per second |
| **input_token_throughput** | Total input tokens processed per second |
| **output_token_throughput** | Total output tokens generated per second |

### 4.4 Performance Analysis Script

For quick P/D aggregation analysis without writing YAML configs:

```bash
python -m serving_cast.scripts.performance_analyze \
  --model-id Qwen/Qwen3-32B \
  --device TEST_DEVICE \
  --num-devices 8 \
  --input-length 3500 \
  --output-length 1500 \
  --tpot-limits 50
```

This searches across TP/DP/batch-size configurations and reports the top throughput-optimal configurations.

---

## 5. Custom Device Profiles

To add a custom hardware profile, create a Python file in `tensor_cast/device_profiles/`:

```python
# tensor_cast/device_profiles/my_device.py
import torch
from tensor_cast.device import (
    CommGrid,
    DeviceProfile,
    InterconnectTopology,
    InterconnectType,
    StaticCost,
)

MY_DEVICE = DeviceProfile(
    name="MY_DEVICE",
    vendor="MyVendor",
    comm_grid=CommGrid(
        grid=torch.tensor([8, 2]),
        topologies={
            0: InterconnectTopology(
                bandwidth_bytes_ps=50e9,
                latency_s=5e-6,
                comm_efficiency=0.9,
                type=InterconnectType.CLOS,
            ),
            1: InterconnectTopology(
                bandwidth_bytes_ps=200e9,
                latency_s=1e-6,
                comm_efficiency=0.95,
                type=InterconnectType.FULL_MESH,
            ),
        },
    ),
    mma_ops={
        torch.float32: 50e12,
        torch.half: 200e12,
        torch.bfloat16: 200e12,
        torch.int8: 400e12,
    },
    gp_ops={
        torch.float32: 50e12,
        torch.half: 100e12,
        torch.bfloat16: 100e12,
    },
    compute_efficiency=0.85,
    memory_size_bytes=80 * (1024 ** 3),         # 80 GB
    memory_bandwidth_bytes_ps=2.0e12,           # 2 TB/s
    memory_efficiency=0.9,
    static_cost=StaticCost(
        mma_op_cost_s=3e-6,
        gp_op_cost_s=3e-6,
        comm_op_cost_s=10e-6,
    ),
)
```

The profile is automatically discovered by name (e.g., `--device MY_DEVICE`).

---

## 6. Running Tests

### TensorCast Tests

```bash
pip install pytest pytest-xdist
pytest ./tensor_cast/tests -n auto
```

### ServingCast Tests

```bash
cd /path/to/msmodeling
./serving_cast/tests/run_test.sh all   # Both unit and system tests
./serving_cast/tests/run_test.sh ut    # Unit tests only
./serving_cast/tests/run_test.sh st    # System tests only
```

### Root Tests

```bash
pytest ./tests
```

---

## 7. Code Quality

The project uses `lintrunner` for consistent code style:

```bash
pip install lintrunner
cd /path/to/msmodeling
lintrunner init       # one-time setup
lintrunner -a         # check and auto-fix style issues
```

Linting tools: Flake8, Black (formatting), isort (imports), Ruff (unified linter).
