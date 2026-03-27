# Phase 1 E2E Baseline (Before op_mapping Update)

Date: 2026-03-15
Branch: integration/phase1-e2e-v2 (after C1-C6 code changes, before op_mapping SKILL update)

## Baseline Results

| Scenario | Op-Count HR | Fused Op HR (w/ zc) | Fused Op HR (w/o zc) |
|----------|------------|--------------------|-----------------------|
| Qwen3 Prefill (nq=1, ql=4096) | 58.7% (27/46) | 35.0% (7/20) | 0.0% (0/13) |
| Qwen3 Decode (nq=16, ql=1, cl=4096) | 75.6% (34/45) | 55.0% (11/20) | 30.8% (4/13) |
| DSv3 Prefill (nq=1, ql=2048, tp=8, dp=2, ep=2) | 40.5% (34/84) | 28.2% (11/39) | 6.7% (2/30) |
| DSv3 Decode (nq=8, ql=1, cl=4096, tp=8, dp=2, ep=2) | 40.2% (35/87) | 25.6% (10/39) | 3.3% (1/30) |

## Qwen3 MISS Breakdown

### Qwen3 Prefill
- [csv_format_raw]: attention.default (FIA needs microbench)
- [input_count_mismatch]: index.Tensor(x2), embedding, apply_rope, reshape_and_cache, swiglu, add.Tensor, copy_
- [shape_mismatch]: rms_norm(x3), mm(x3), matmul_all_reduce(x2), add_rms_norm2, all_gather

### Qwen3 Decode
- [comm_sub_kernel_miss]: matmul_all_reduce(x2)
- [csv_format_raw]: attention.default
- [input_count_mismatch]: index.Tensor(x2), embedding, apply_rope, reshape_and_cache, add.Tensor, copy_
- [shape_mismatch]: all_gather

## DSv3 MISS Breakdown

### DSv3 Decode
- [csv_not_found]: topk(x3), sum(x3), add_rms_norm_quant2, sigmoid, bitwise_not, where, div, init_routing_v2, cat, grouped_matmul_quant_swiglu, grouped_matmul_quant
- [input_count_mismatch]: quantize(x6), add.Tensor(x4), static_quant_linear(x3), index.Tensor(x2), swiglu(x2), embedding, concat_and_cache_mla, scatter, gather, mul.Tensor, copy_
- [shape_mismatch]: static_quant_linear_all_reduce(x2), all_to_all(x2), rms_norm, mlapo_quant, multihead_latent_attention, add_rms_norm2, mm, clone, add.Tensor, mul.Tensor, all_reduce, all_gather
