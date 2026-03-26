import unittest

import torch

import tensor_cast.ops  # noqa: F401 — triggers op registration


class KvRmsNormRopeCacheOpTest(unittest.TestCase):
    def test_op_registered(self):
        """kv_rmsnorm_rope_cache op should be registered in tensor_cast namespace."""
        self.assertTrue(hasattr(torch.ops.tensor_cast, "kv_rmsnorm_rope_cache"))

    def test_op_returns_correct_shapes(self):
        """Op should return (k_pe, kv_c_normed) with correct shapes."""
        num_tokens = 8
        kv_lora_rank = 512
        qk_rope_head_dim = 64
        with torch.device("meta"):
            kv = torch.empty(num_tokens, kv_lora_rank + qk_rope_head_dim)
            gamma = torch.empty(kv_lora_rank)
            cos = torch.empty(1, num_tokens, qk_rope_head_dim)
            sin = torch.empty(1, num_tokens, qk_rope_head_dim)
            kv_cache = torch.empty(256, 1, kv_lora_rank + qk_rope_head_dim)
            slot_mapping = torch.empty(num_tokens, dtype=torch.long)

        k_pe, kv_c_normed = torch.ops.tensor_cast.kv_rmsnorm_rope_cache(
            kv,
            gamma,
            cos,
            sin,
            kv_cache,
            slot_mapping,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            epsilon=1e-6,
        )
        self.assertEqual(k_pe.shape, (num_tokens, qk_rope_head_dim))
        self.assertEqual(kv_c_normed.shape, (num_tokens, kv_lora_rank))


class MC2PassVerificationTest(unittest.TestCase):
    """Verify MC2 (MatMul+AllReduce) pass produces fused ops in dispatch trace."""

    def setUp(self):
        torch.compiler.reset()

    def _run_model(self, quant_action):
        from dataclasses import asdict

        from tensor_cast.core.input_generator import generate_inputs
        from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
        from tensor_cast.core.user_config import UserInputConfig

        user_input = UserInputConfig(
            model_id="Qwen/Qwen3-32B",
            num_queries=1,
            query_len=100,
            context_length=1000,
            do_compile=True,
            quantize_linear_action=quant_action,
            num_mtp_tokens=0,
            num_hidden_layers_override=1,
            world_size=2,
            tp_size=2,
        )
        runner = ModelRunner(user_input)
        result = runner.run_inference(generate_inputs_func=generate_inputs)
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        return result["table_result"]

    def test_mc2_bf16_fused(self):
        """BF16 MC2: matmul_all_reduce should appear in dispatch trace."""
        from tensor_cast.core.quantization.datatypes import QuantizeLinearAction

        table = self._run_model(QuantizeLinearAction.DISABLED)
        self.assertIn("tensor_cast.matmul_all_reduce.default", table)

    def test_mc2_w8a8_fused(self):
        """W8A8 MC2: static_quant_linear_all_reduce should appear."""
        from tensor_cast.core.quantization.datatypes import QuantizeLinearAction

        table = self._run_model(QuantizeLinearAction.W8A8_STATIC)
        self.assertIn(
            "tensor_cast.static_quant_linear_all_reduce.default", table
        )

    def test_mc2_op_mapping_entries_exist(self):
        """All MC2 fused op variants should have op_mapping entries."""
        from pathlib import Path

        import yaml

        mapping_path = (
            Path(__file__).resolve().parents[2]
            / "tensor_cast/performance_model/profiling_database/data"
            / "ATLAS_800_A3_752T_128G_DIE/vllm_ascend/v0.13.0/op_mapping.yaml"
        )
        with open(mapping_path) as f:
            full = yaml.safe_load(f)
        mapping = full.get("operator_mappings", {})

        # All 5 MC2 variants from matmul_allreduce.py should have entries
        mc2_ops = [
            "tensor_cast.matmul_all_reduce.default",
            "tensor_cast.static_quant_linear_all_reduce.default",
            "tensor_cast.static_quant_linear_int4_all_reduce.default",
            "tensor_cast.fp8_linear_all_reduce.default",
            "tensor_cast.mxfp4_linear_all_reduce.default",
        ]
        for op in mc2_ops:
            self.assertIn(op, mapping, f"MC2 op '{op}' missing from op_mapping")
            entry = mapping[op]
            self.assertTrue(
                entry.get("composite", False),
                f"MC2 op '{op}' should be composite",
            )


class ConfigFlagTest(unittest.TestCase):
    def test_enable_kv_rmsnorm_rope_cache_flag(self):
        """Config should have enable_kv_rmsnorm_rope_cache flag (default False)."""
        from tensor_cast import config

        self.assertFalse(
            config.compilation.fusion_patterns.enable_kv_rmsnorm_rope_cache
        )


if __name__ == "__main__":
    unittest.main()
