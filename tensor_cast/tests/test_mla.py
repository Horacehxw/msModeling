import unittest
import torch
from unittest.mock import MagicMock
from ..performance_model import _multihead_latent_attention_properties_helper

class TestMultiheadLatentAttentionHelper(unittest.TestCase):
    """Unit tests for mla_properties_helper in performance_model.py."""
    def setUp(self):
        """Set up test fixtures."""
        self.softmax_dtype = torch.float16  
        self.op_invoke_info = MagicMock()

    def test_mla_properties_helper_prefill_without_context(self):
        """Test mla properties calculation for prefill mode without context"""
        q = torch.empty(500, 8, 64, dtype=torch.float16)  
        kv_c_normed = torch.empty(500, 32)  
        k_rot = torch.empty(500, 32)    
        kv_cache = torch.empty(4, 128, 64)
        query_start_loc = 0  
        seq_lens = torch.tensor([500])  
        query_lens = torch.tensor([500])  
        W_UK_T = torch.empty(8, 32, 32)
        W_UV = torch.empty(8, 32, 32)
        kv_b_proj = torch.empty(32, 96)
        v_head_dim = 64 

        self.op_invoke_info.args = [
            q, kv_c_normed, k_rot, kv_cache, None, query_start_loc,
            seq_lens, query_lens, W_UK_T, W_UV, kv_b_proj, v_head_dim
        ]
        
        self.op_invoke_info.get_memory_access_properties.return_value = MagicMock(
            compute_ops={self.softmax_dtype: MagicMock()},
            memory_read_bytes=0 
        )

        properties = _multihead_latent_attention_properties_helper(self.op_invoke_info, self.softmax_dtype)
        
        total_fma_ops_expected = ((8 * (32 + 64)) * 500 * 32 * 2 + 
                                  500 * 500 * 8 * 64 * 2 + 500 * 500 * 8 * 64 * 2)
        total_gp_ops_expected = (500 * 500 * 8 * 4) 
        compute_ops_mma_ops = properties.compute_ops[q.dtype].mma_ops
        compute_ops_gp_ops = properties.compute_ops[self.softmax_dtype].gp_ops
        self.assertEqual(compute_ops_mma_ops, total_fma_ops_expected, "FMA operations do not match the expected value.")
        self.assertEqual(compute_ops_gp_ops, total_gp_ops_expected, "GP operations do not match the expected value.")

    def test_mla_properties_helper_prefill_with_context(self):
        """Test mla properties calculation for prefill mode with context"""
        q = torch.empty(500, 8, 64, dtype=torch.float16)  
        kv_c_normed = torch.empty(500, 32)  
        k_rot = torch.empty(500, 32)    
        kv_cache = torch.empty(8, 128, 64)
        query_start_loc = 0  
        seq_lens = torch.tensor([1000])  
        query_lens = torch.tensor([500])  
        W_UK_T = torch.empty(8, 32, 32)
        W_UV = torch.empty(8, 32, 32)
        kv_b_proj = torch.empty(32, 96)
        v_head_dim = 64 

        self.op_invoke_info.args = [
            q, kv_c_normed, k_rot, kv_cache, None, query_start_loc,
            seq_lens, query_lens, W_UK_T, W_UV, kv_b_proj, v_head_dim
        ]
        
        self.op_invoke_info.get_memory_access_properties.return_value = MagicMock(
            compute_ops={self.softmax_dtype: MagicMock()},
            memory_read_bytes=0 
        )

        properties = _multihead_latent_attention_properties_helper(self.op_invoke_info, self.softmax_dtype)
        
        # total_fma_ops_expected = (prefill_op1_ops + prefill_op2_ops + prefill_op4_ops)
        total_fma_ops_expected = ((8 * (32 + 64)) * 1000 * 32 * 2 + 
                                  500 * 1000 * 8 * 64 * 2 + 500 * 1000 * 8 * 64 * 2)
        
        # total_gp_ops_expected = prefill_op3_ops
        total_gp_ops_expected = (500 * 1000 * 8 * 4) 

        compute_ops_mma_ops = properties.compute_ops[q.dtype].mma_ops
        compute_ops_gp_ops = properties.compute_ops[self.softmax_dtype].gp_ops

        self.assertEqual(compute_ops_mma_ops, total_fma_ops_expected, "FMA operations do not match the expected value.")
        self.assertEqual(compute_ops_gp_ops, total_gp_ops_expected, "GP operations do not match the expected value.")

if __name__ == '__main__':
    unittest.main()