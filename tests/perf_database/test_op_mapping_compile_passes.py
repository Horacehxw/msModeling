import unittest
from pathlib import Path

import yaml

CANN85_OP_MAPPING = (
    Path(__file__).resolve().parents[2]
    / "tensor_cast/performance_model/perf_database/data"
    / "ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5"
    / "op_mapping.yaml"
)


@unittest.skipIf(
    not CANN85_OP_MAPPING.exists(),
    "CANN 8.5 op_mapping.yaml not found",
)
class CompilePassOpMappingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CANN85_OP_MAPPING, encoding="utf-8") as f:
            full = yaml.safe_load(f)
        cls.mapping = full.get("operator_mappings", {})
        cls.torch_npu_ref = full.get("torch_npu_reference", {})

    def test_kv_rmsnorm_rope_cache_entry(self):
        """kv_rmsnorm_rope_cache should map to NPU KvRmsNormRopeCache kernel."""
        entry = self.mapping.get("tensor_cast.kv_rmsnorm_rope_cache.default")
        self.assertIsNotNone(entry, "Missing kv_rmsnorm_rope_cache in op_mapping")
        self.assertEqual(entry["kernel_type"], "KvRmsNormRopeCache")

    def test_matmul_all_reduce_entry(self):
        """matmul_all_reduce should be composite with MatMulV2 + hcom sub-kernels."""
        entry = self.mapping.get("tensor_cast.matmul_all_reduce.default")
        self.assertIsNotNone(entry, "Missing matmul_all_reduce in op_mapping")
        self.assertTrue(entry.get("composite", False))
        self.assertIn("MatMulV2", entry["sub_kernels"])
        self.assertIn("hcom_allReduce_", entry["sub_kernels"])

    def test_static_quant_linear_all_reduce_entry(self):
        """static_quant_linear_all_reduce should be composite."""
        entry = self.mapping.get(
            "tensor_cast.static_quant_linear_all_reduce.default"
        )
        self.assertIsNotNone(entry)
        self.assertTrue(entry.get("composite", False))

    def test_mlapo_is_composite(self):
        """mlapo should be composite with sub_kernels including KvRmsNormRopeCache."""
        entry = self.mapping.get("tensor_cast.mlapo.default")
        self.assertIsNotNone(entry, "Missing mlapo in op_mapping")
        self.assertTrue(entry.get("composite", False))
        self.assertIn("KvRmsNormRopeCache", entry.get("sub_kernels", []))

    def test_torch_npu_reference_kv_rmsnorm(self):
        """torch_npu_reference should have KvRmsNormRopeCache entry."""
        self.assertIn("KvRmsNormRopeCache", self.torch_npu_ref)


if __name__ == "__main__":
    unittest.main()
