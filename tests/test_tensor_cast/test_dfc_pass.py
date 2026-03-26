# tests/test_tensor_cast/test_dfc_pass.py
import os
import unittest
from dataclasses import asdict

import torch
from parameterized import parameterized

from tensor_cast.compilation import get_backend
from tensor_cast.core.config_resolver import ConfigResolver
from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction, QuantizeAttentionAction
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import TEST_DEVICE
from tensor_cast.layers.attention import AttentionTensorCast
from tensor_cast.layers.quant_linear import TensorCastQuantLinear
from tensor_cast.model_config import ModelConfig, ParallelConfig
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.performance_model.memory_tracker import MemoryTracker
from tensor_cast.quantize_utils import LinearQuantType, QuantGranularity
from tensor_cast.runtime import Runtime
from tensor_cast.transformers.model import TransformerModel
from tensor_cast.transformers.custom_model_registry import get_moe_config
from tensor_cast.transformers.utils import AutoModelConfigLoader
from .test_common import count_events, get_quant_config


class DfcPassTestCase(unittest.TestCase):
    """DispatchFFNCombine fusion pass tests.

    DSv3 configs from Phase 1 E2E (5525b21):
      Prefill: W8A8, TP=8/DP=2/EP=16, nq=1, ql=256
      Decode:  W8A8, TP=8/DP=2/EP=16, nq=16, ql=1, cl=4096
    DSv3 first_k_dense_replace=3, so num_hidden_layers_override≥4 to include MoE.
    """

    def setUp(self):
        torch.compiler.reset()

    @parameterized.expand(
        [
            # (scenario, num_queries, query_len, context_length)
            ("prefill", 1, 256, 0),
            ("decode", 16, 1, 4096),
        ]
    )

    def test_dfc_dsv3_ep(self, scenario, num_queries, query_len, context_length):
        """Verify that DFC is effective for DSv3 large EP configuration (Phase 1)"""
        model_id = "deepseek-ai/DeepSeek-V3"
        user_input = UserInputConfig(
            device="ATLAS_800_A2_280T_64G",
            model_id=model_id,
            num_queries=num_queries,
            query_len=query_len,
            context_length=context_length,
            do_compile=True,
            allow_graph_break=True,
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)
        self.assertIn("tensor_cast.dispatch_ffn_combine.default", result["table_result"])
        self.assertNotIn("tensor_cast.permute_tokens.default",result["table_result"])
        self.assertNotIn("tensor_cast.unpermute_tokens.default", result["table_result"])

    def test_dfc_output_shape_matches_baseline(self):
        """Verify that the DFC output shape is consistent with the unpermute_tokens baseline (Qwen3 non-EP)"""
        from tensor_cast import config

        model_id = "Qwen/Qwen3-235B-A22B"

        def run_model(enable_dfc):
            with unittest.mock.patch.object(
                    config.compilation.fusion_patterns,
                    'enable_dispatch_ffn_combine',
                    new_callable=unittest.mock.PropertyMock,
                    return_value=enable_dfc
            ):
                torch.compiler.reset()
                user_input = UserInputConfig(
                    model_id=model_id,
                    do_compile=True,
                    num_hidden_layers_override=1,
                    quantize_linear_action=QuantizeLinearAction.DISABLED,
                )
                config_resolver = ConfigResolver(user_input=user_input)
                model_config = config_resolver.resolve()
                model = TransformerModel(model_id, model_config)
                model = torch.compile(model, backend=get_backend(), fullgraph=True)
                inputs = torch.empty([1, 100], dtype=torch.long, device="meta")
                pos = torch.empty([1, 100], dtype=torch.long, device="meta")
                perf_model = AnalyticPerformanceModel(TEST_DEVICE)
                with (
                    Runtime(
                        perf_model, TEST_DEVICE, memory_tracker=MemoryTracker(TEST_DEVICE)
                    ) as rt,
                    torch.no_grad(),
                ):
                    out = model.forward(inputs, pos)
                return out.shape

        baseline_shape = run_model(enable_dfc=False)
        dfc_shape = run_model(enable_dfc=True)
        self.assertEqual(baseline_shape, dfc_shape)

    def test_dfc_dsv3_0324_w8a8_local_model(self):
        """Verify DFC on local DeepSeek-V3-0324 W8A8 model."""
        if os.getenv("TC_ENABLE_LOCAL_MODEL_TESTS") != "1":
            self.skipTest(
                "Local model test disabled. Set TC_ENABLE_LOCAL_MODEL_TESTS=1 to enable."
            )

        # model_id is from modelscope, download to local and replace model_id with local path
        user_input = UserInputConfig(
            device="ATLAS_800_A3_752T_128G_DIE",
            model_id="Eco-Tech/DeepSeek-V3-0324-w8a8-mtp-QuaRot",
            num_queries=2,
            query_len=4096,
            do_compile=True,
            allow_graph_break=True,
            world_size=16,
            tp_size=8,
            dp_size=2,
            ep_size=16,
            quantize_linear_action=QuantizeLinearAction.W8A8_STATIC,
            performance_model="profiling",
            profiling_database=(
                "tensor_cast/performance_model/profiling_database/data/"
                "ATLAS_800_A3_752T_128G_DIE/vllm_ascend/"
                "vllm0.15.0_torch2.9.0_cann8.5"
            ),
        )
        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)

        self.assertIn("tensor_cast.dispatch_ffn_combine.default", result["table_result"])
        self.assertNotIn("tensor_cast.permute_tokens.default", result["table_result"])
        self.assertNotIn("tensor_cast.unpermute_tokens.default", result["table_result"])
