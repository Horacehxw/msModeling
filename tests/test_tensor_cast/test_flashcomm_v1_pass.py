"""Test for FlashCommV1 Pass.

Transforms: all_reduce → rms_norm  =>  reduce_scatter → rms_norm(local) → all_gather
This matches VLLM_ASCEND_ENABLE_FLASHCOMM1=1 behavior on NPU.
"""

import unittest
from dataclasses import asdict

import torch
from parameterized import parameterized

from tensor_cast import config, ops  # noqa: F401
from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.model_config import WordEmbeddingTPMode


class FlashCommV1PassTestCase(unittest.TestCase):
    """Test FlashCommV1 pass transforms all_reduce+norm to FlashCommV1 pattern."""

    def setUp(self):
        torch.compiler.reset()
        self._orig_enable_flashcomm_v1 = config.compilation.passes.enable_flashcomm_v1

    def tearDown(self):
        config.compilation.passes.enable_flashcomm_v1 = self._orig_enable_flashcomm_v1

    @parameterized.expand(
        [
            # (tp_size, expected_local_seq)
            # (1, 128),   # No FlashCommV1: rms_norm sees full seq
            (2, 64),  # FlashCommV1: rms_norm sees seq/2
            # (4, 32),    # FlashCommV1: rms_norm sees seq/4
            # (8, 16),    # FlashCommV1: rms_norm sees seq/8
        ]
    )
    def test_sp_reduces_rms_norm_seq_dim(self, tp_size: int, expected_local_seq: int):
        """Verify rms_norm operates on reduced seq length with FlashCommV1 enabled."""
        config.compilation.passes.enable_flashcomm_v1 = True
        user_input = UserInputConfig(
            model_id="Qwen/Qwen3-32B",
            num_queries=1,
            query_len=128,
            context_length=0,
            do_compile=True,
            dump_input_shapes=True,
            enable_flashcomm_v1=True,
            num_mtp_tokens=0,
            num_hidden_layers_override=1,
            world_size=tp_size,
            tp_size=tp_size,
            word_embedding_tp=True,
            word_embedding_tp_mode=WordEmbeddingTPMode.row.value,
            quantize_linear_action=QuantizeLinearAction.DISABLED,
        )

        model_runner = ModelRunner(user_input)
        result = model_runner.run_inference(generate_inputs_func=generate_inputs)
        if isinstance(result, ModelRunnerMetrics):
            result = asdict(result)

        table = result["table_result"]

        # Verify rms_norm is present
        self.assertIn("tensor_cast.rms_norm.default", table)
        self.assertIn(
            f"[1, {expected_local_seq}, 5120], [5120]",
            table,
            "FlashCommV1 should shard the entry rms_norm sequence dimension",
        )

        # Verify FlashCommV1 pattern presence
        if tp_size > 1:
            # With FlashCommV1: should have reduce_scatter and all_gather
            self.assertIn(
                "tensor_cast.reduce_scatter.default",
                table,
                "FlashCommV1 mode should have reduce_scatter",
            )
            self.assertIn(
                "tensor_cast.all_gather.default",
                table,
                "FlashCommV1 mode should have all_gather",
            )
            # Should NOT have all_reduce (replaced by FlashCommV1 pattern)
            self.assertNotIn(
                "tensor_cast.all_reduce.default",
                table,
                "FlashCommV1 mode should replace all_reduce",
            )
        else:
            # Without FlashCommV1: should have all_reduce
            self.assertIn(
                "tensor_cast.all_reduce.default",
                table,
                "Non-FlashCommV1 mode should have all_reduce",
            )


if __name__ == "__main__":
    # PYTHONPATH=/pathto/msmodeling:$PYTHONPATH pytest -v \
    #   tests/test_tensor_cast/test_flashcomm_v1_pass.py \
    #   --log-cli-level=DEBUG > test.log
    unittest.main()
