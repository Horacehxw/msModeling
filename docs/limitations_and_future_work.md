# Limitations and Future Work

This document summarizes the current limitations of msModeling and the planned or potential directions for future development. Items marked with checkboxes reflect the project's own TODO tracking (from `tensor_cast/README.md` and code TODOs).

---

## 1. Current Limitations

### 1.1 Simulation Fidelity

- **Analytic roofline approximation.** The primary performance model uses a peak-throughput roofline analysis. Real hardware exhibits additional overheads — cache effects, memory bank conflicts, warp scheduling inefficiencies, instruction-level bottlenecks — that are not captured. The gap between predicted and actual latency varies by operator and workload.
- **No backward pass modeling.** The framework simulates inference only. Training workloads (forward + backward + optimizer step) are out of scope.
- **Static scheduling model.** Operator execution is modeled as sequential within a single stream. Overlapping compute and communication (e.g., pipelining AllReduce with the next layer's matmul) is not fully modeled at the operator level.
- **No memory fragmentation.** Memory tracking computes allocation sizes and peak usage, but does not model memory fragmentation, allocator overhead, or memory pool behavior.

### 1.2 Model Coverage

- **HuggingFace-centric.** Models must be loadable via `transformers.AutoModelForCausalLM`, `AutoModelForVision2Seq`, or the `diffusers` library. Custom model architectures outside these frameworks require manual adapter code.
- **Compilation coverage.** The `torch.compile` fusion passes (RMSNorm, RoPE, linear merge) are tuned for specific model families. Models with non-standard layer structures may not benefit from or may break during compilation. Fusion support for Qwen3 Dense and DeepSeek is still incomplete.
- **Vision-Language models.** VL model support (e.g., Qwen3-VL) is available but limited to specific adapter implementations. Arbitrary multimodal architectures are not generically supported.

### 1.3 Parallelism

- **No Context Parallelism (CP).** Splitting long sequences across devices along the sequence dimension is not implemented.
- **No Sequence Parallelism (SP).** Distributing non-tensor-parallel operations (LayerNorm, Dropout) across the sequence dimension is not supported.
- **Pipeline Parallelism is partial.** PP is supported in serving benchmark scripts but not fully integrated into TensorCast's single-pass simulation with inter-stage communication modeling.
- **Auto-parallelism search is manual.** The benchmark script searches over TP/DP/batch-size configurations, but does not automatically determine the optimal parallelism strategy given hardware constraints. The user must specify or provide a search space.

### 1.4 Quantization

- **Symmetric quantization only** for most schemes. Asymmetric quantization support is defined in the configuration but not fully validated across all model families.
- **No calibration-based quantization.** Static quantization scales are configured by the user or derived from model metadata. The framework does not perform activation calibration passes.
- **FP8 and MXFP4 are early-stage.** These quantization modes are implemented but have less validation coverage than W8A8 and W4A8.

### 1.5 Device Profiles

- **Built-in profiles are Ascend-focused.** The primary validated profiles are for Ascend ATLAS A2/A3 accelerators. Profiles for other vendors (NVIDIA, AMD, Intel) exist only as examples and are not guaranteed to be accurate.
- **Interconnect modeling is simplified.** The `CommGrid` model supports hierarchical topologies (Full Mesh, Clos), but does not model congestion, routing conflicts, or adaptive routing behavior.
- **No thermal or power modeling.** Device profiles do not account for thermal throttling or power limits that could reduce sustained performance.

### 1.6 ServingCast

- **Fixed-length load generation only.** The current `FixedLengthLoadGen` generates requests with uniform input/output lengths. Real workloads have diverse length distributions (e.g., log-normal).
- **Single backend.** Only the MindIE backend is implemented. Other serving frameworks (vLLM, TensorRT-LLM, etc.) would need new backend implementations.
- **No request priority or fairness.** The scheduler does not model request priorities, fair queuing, or SLO-aware preemption.
- **Simplified TTFT/TPOT calculation.** The performance analysis script uses an analytical formula for TTFT and TPOT that assumes uniform request processing and ignores queuing dynamics under bursty load.

---

## 2. Future Work

### 2.1 From the Project TODO List

#### Parallelism
- [ ] Context Parallelism (CP)
- [ ] Sequence Parallelism (SP)
- [ ] Full Pipeline Parallelism with inter-stage communication

#### Quantization
- [ ] FP8 validation and optimization
- [ ] FP4 (beyond MXFP4)
- [ ] C8 (INT8 communication quantization)
- [ ] Calibration-based static quantization

#### Compilation
- [ ] Complete fusion support for Qwen3 Dense models
- [ ] Complete fusion support for DeepSeek models
- [ ] Additional fusion patterns (SwiGLU, fused attention variants)

#### Performance Modeling
- [ ] Empirical performance model with collected benchmark data
- [ ] Analytic models for additional PyTorch and Ascend-specific operators
- [ ] ML-based performance predictors (e.g., learned cost models)

### 2.2 Potential Enhancements

#### Simulation Accuracy
- **Overlap modeling.** Model computation-communication overlap for pipeline parallelism and asynchronous AllReduce.
- **Memory allocator simulation.** Model real allocator behavior (caching allocator, memory pools) to predict OOM more accurately.
- **Multi-stream execution.** Simulate concurrent CUDA/Ascend streams for operations that can execute in parallel.

#### Model Support
- **Encoder-decoder architectures.** Extend beyond decoder-only causal LMs to support T5-style encoder-decoder models.
- **Speech and audio models.** Adapt the framework for Whisper-style or other audio models.
- **Broader diffuser support.** Extend the diffusers integration to cover more image/video generation architectures.

#### Serving Simulation
- **Variable-length load generation.** Support log-normal or trace-driven request length distributions.
- **Multiple backend implementations.** Add vLLM, TensorRT-LLM, or other serving framework backends.
- **SLO-aware scheduling.** Model priority queues and deadline-driven preemption.
- **Autoscaling simulation.** Model elastic scaling of instance pools based on load.

#### Device Ecosystem
- **Validated NVIDIA profiles.** Provide well-calibrated profiles for A100, H100, H200, B200.
- **Multi-vendor comparison workflows.** Streamline cross-vendor performance comparison in a single benchmark run.
- **Interconnect congestion modeling.** Model network contention under heavy all-to-all communication patterns.

#### Developer Experience
- **Web-based result viewer.** Interactive dashboard for exploring performance breakdowns beyond Chrome Trace.
- **CI-integrated regression testing.** Automated performance regression detection on model updates.
- **Python API.** Expose TensorCast and ServingCast as importable libraries with programmatic interfaces, in addition to CLI scripts.

---

## 3. Contributing

Contributions to address any of the above limitations are welcome. The project uses an RFC process for significant design changes — see `docs/RFC/` for existing proposals and the RFC template. Before submitting code:

1. Run `lintrunner -a` to ensure code style compliance.
2. Add or update unit tests for new functionality.
3. Follow the PR template in `.gitcode/PULL_REQUEST_TEMPLATE.md`.
