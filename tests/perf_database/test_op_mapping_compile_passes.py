import unittest
from pathlib import Path

import yaml

OP_MAPPING_PATH = (
    Path(__file__).resolve().parents[2]
    / "tensor_cast/performance_model/perf_database/data"
    / "ATLAS_800_A3_752T_128G_DIE/vllm_ascend/v0.13.0/op_mapping.yaml"
)


class CompilePassOpMappingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(OP_MAPPING_PATH) as f:
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

    def test_mlapo_includes_kv_rmsnorm_rope_cache(self):
        """mlapo should be composite with KvRmsNormRopeCache in sub_kernels."""
        entry = self.mapping.get("tensor_cast.mlapo.default")
        self.assertIsNotNone(entry, "Missing mlapo in op_mapping")
        self.assertTrue(entry.get("composite", False))
        self.assertIn("KvRmsNormRopeCache", entry["sub_kernels"])

    def test_torch_npu_reference_kv_rmsnorm(self):
        """torch_npu_reference should have KvRmsNormRopeCache entry."""
        self.assertIn("KvRmsNormRopeCache", self.torch_npu_ref)


if __name__ == "__main__":
    unittest.main()
