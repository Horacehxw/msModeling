# MindStudio-Modeling (msModeling) — Project Overview

## What Is msModeling?

MindStudio-Modeling (msModeling) is a lightweight, full-system simulation framework for evaluating the theoretical performance of Large Language Model (LLM) inference workloads. Rather than executing models on physical accelerators, msModeling predicts operator-level execution time, memory consumption, and end-to-end serving throughput on user-defined hardware profiles. This enables engineers and researchers to compare deployment strategies — parallelism configurations, quantization schemes, batch sizes, and device selections — without needing access to the target hardware.

The project is developed by Huawei Technologies and licensed under the Mulan Permissive Software License, Version 2 (MulanPSL2).

## Two Core Modules

msModeling is composed of two complementary subsystems:

### TensorCast — Single-Request Performance Simulation

TensorCast operates as a "virtual machine" runtime simulator for PyTorch programs. It intercepts a model's computational graph via `torch.compile` and `TorchDispatchMode`, then simulates execution on a user-defined `DeviceProfile` rather than running on real hardware.

Key capabilities:

- **Out-of-the-box HuggingFace support.** Load any `transformers` or `diffusers` model by its Hub ID.
- **Operator-level performance breakdown.** Estimates execution time per op using analytic roofline models or empirical data.
- **Memory footprint tracking.** Reports total and peak memory, including KV cache, model weights, and activations.
- **Graph optimization pipeline.** Automatic quantization, tensor-parallel sharding, FX-graph fusion passes (RMSNorm, RoPE, linear merge), and constant folding.
- **Chrome Trace output.** Produces `chrome://tracing`-compatible JSON files for visual timeline analysis.
- **Configurable hardware profiles.** Built-in Ascend ATLAS family (A2, A3) profiles; drop-in custom device definitions.

### ServingCast — End-to-End Serving Simulation

ServingCast builds on top of TensorCast to simulate LLM inference serving at the system level. It uses discrete-event simulation (via the `salabim` library) to model request arrival, batching, scheduling, KV cache management, and multi-instance coordination.

Key capabilities:

- **Prefill/Decode (P/D) modes.** Supports aggregation (both on one instance) and disaggregation (separate prefill and decode pools).
- **Batch scheduling.** Continuous batching with configurable token budgets and concurrency limits.
- **KV cache lifecycle management.** Block-based paged attention with configurable block sizes.
- **Inter-instance communication modeling.** Simulates host-to-device and device-to-device transfer with configurable bandwidth and latency.
- **Rich output metrics.** E2E latency, TTFT, TPOT, request throughput, and token throughput at multiple percentiles (P75, P90, P99).
- **Profiling support.** Generates Chrome Trace and `profiler.db` files viewable in MindStudio Insight.

## Repository Structure

```
msModeling/
├── tensor_cast/                  # Performance simulation framework
│   ├── core/                     #   Model runner, input generator, config resolver
│   │   └── quantization/         #   Quantization datatypes and config
│   ├── compilation/              #   torch.compile backend and FX graph passes
│   │   ├── passes/               #   Optimization passes (merge_linear, sink_split, etc.)
│   │   └── patterns/             #   Fusion pattern matchers (RMSNorm, RoPE)
│   ├── layers/                   #   NN layers (attention, MLA, MoE, MTP, parallel linear)
│   ├── ops/                      #   Custom op implementations for simulation
│   ├── performance_model/        #   Analytic roofline model, empirical model, memory tracker
│   ├── transformers/             #   HuggingFace transformer model integration
│   ├── diffusers/                #   Diffusers library integration
│   ├── device_profiles/          #   Drop-in custom hardware profiles
│   ├── scripts/                  #   CLI entry points (text_generate, benchmark, video_generate)
│   └── tests/                    #   Unit and integration tests
├── serving_cast/                 # Inference serving simulation
│   ├── service/                  #   Backend implementations (MindIE), task scheduling, reporting
│   ├── profiler/                 #   Profiling utilities
│   ├── scripts/                  #   Performance analysis scripts
│   ├── example/                  #   Example YAML configuration files
│   └── tests/                    #   Unit tests (ut/) and system tests (st/)
├── docs/                         # Documentation and RFCs
│   └── RFC/                      #   Request-for-comments design documents
├── tests/                        # Root-level tests (simulation time)
├── tools/                        # Linting and developer tooling
├── stime.py                      # Simulation time management (salabim-based)
├── pyproject.toml                # Project metadata and linter configuration
├── requirements.txt              # Root-level dependencies
└── README.md                     # Project description
```

## Supported Models

msModeling supports HuggingFace transformer models out of the box. Models that have been specifically validated include:

| Model Family | Example Variants |
|---|---|
| Qwen3 | Qwen3-32B, Qwen3-235B-A22B |
| DeepSeek | DeepSeek-V3, DeepSeek-V3.1 (671B, MoE) |
| GLM | GLM-4.5 and variants |
| Kimi | Kimi-K2 |
| Vision-Language | Qwen3-VL, Qwen2-VL |
| Video Generation | HunyuanVideo |

Any model loadable via `transformers.AutoModelForCausalLM` or the `diffusers` library can be used, though models with exotic architectures may need adapter layers.

## Supported Hardware

Built-in device profiles ship for Ascend ATLAS accelerators:

| Device | Compute | Memory |
|---|---|---|
| ATLAS 800 A2 376T 64G | 376 TFLOPS (FP16) | 64 GB HBM |
| ATLAS 800 A2 313T 64G | 313 TFLOPS (FP16) | 64 GB HBM |
| ATLAS 800 A2 280T 64G | 280 TFLOPS (FP16) | 64 GB HBM |
| ATLAS 800 A2 280T 32G PCIe | 280 TFLOPS (FP16) | 32 GB HBM |
| ATLAS 800 A3 752T 128G | 752 TFLOPS (FP16) | 128 GB HBM |
| ATLAS 800 A3 560T 128G | 560 TFLOPS (FP16) | 128 GB HBM |

Example profiles for H20 are provided under `device_profiles/`. Custom devices can be defined in Python files and dropped into the `tensor_cast/device_profiles/` directory for automatic loading.

## Parallelism Support

msModeling models the following parallelism strategies:

| Strategy | Status | Description |
|---|---|---|
| Tensor Parallelism (TP) | Supported | Column/row-wise sharding of linear layers |
| Data Parallelism (DP) | Supported | Replicates model across devices, splits batches |
| Expert Parallelism (EP) | Supported | Distributes MoE experts across devices |
| Pipeline Parallelism (PP) | Partial | Supported in serving benchmark scripts |
| Context Parallelism (CP) | Planned | Not yet implemented |
| Sequence Parallelism (SP) | Planned | Not yet implemented |

Fine-grained parallelism controls are available for MLP, LM head, and output projection layers independently of the global TP/DP setting.

## Quantization Support

| Scheme | Weight | Activation | Status |
|---|---|---|---|
| W8A16 | INT8 | FP16/BF16 | Supported (static) |
| W8A8 | INT8 | INT8 | Supported (static & dynamic) |
| W4A8 | INT4 | INT8 | Supported (static & dynamic) |
| FP8 | FP8 | FP8 | Supported |
| MXFP4 | FP4 (mixed) | FP4 (mixed) | Supported |
| KV Cache INT8 | — | INT8 | Supported |

## License

This project is licensed under the Mulan Permissive Software License, Version 2 (Mulan PSL v2). See the `LICENSE` file for full terms.
