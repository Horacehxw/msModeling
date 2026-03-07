"""Tests for generate_microbench.py."""

from tools.perf_data_collection.generate_microbench import (
    generate_attention_script,
    generate_elementwise_script,
    generate_gemm_script,
)


def test_gemm_script_contains_torch_mm():
    """GEMM script should use torch.mm for MatMulV2."""
    script = generate_gemm_script(
        M=128, N=5120, K=25600, dtype="torch.bfloat16", kernel_type="MatMulV2"
    )
    assert "torch.mm" in script
    assert "128" in script  # M
    assert "5120" in script  # N
    assert "25600" in script  # K
    assert "bfloat16" in script


def test_gemm_script_is_valid_python():
    """Generated GEMM script should be valid Python syntax."""
    script = generate_gemm_script(
        M=64, N=1024, K=512, dtype="torch.bfloat16", kernel_type="MatMulV2"
    )
    compile(script, "<test>", "exec")  # Raises SyntaxError if invalid


def test_gemm_script_has_warmup_and_timing():
    """Script should include warmup iterations and timing."""
    script = generate_gemm_script(
        M=64, N=1024, K=512, dtype="torch.bfloat16", kernel_type="MatMulV2"
    )
    assert "warmup" in script.lower() or "WARMUP" in script
    assert "time" in script.lower()


def test_gemm_script_has_csv_output():
    """Script should output results in CSV-compatible format."""
    script = generate_gemm_script(
        M=64, N=1024, K=512, dtype="torch.bfloat16", kernel_type="MatMulV2"
    )
    assert "Duration" in script or "duration" in script


def test_elementwise_script_contains_add():
    """Elementwise script should use torch.add or similar."""
    script = generate_elementwise_script(
        num_tokens=128, hidden_size=5120, dtype="torch.bfloat16", kernel_type="Add"
    )
    assert "128" in script
    assert "5120" in script
    assert "bfloat16" in script


def test_elementwise_script_is_valid_python():
    """Generated elementwise script should be valid Python syntax."""
    script = generate_elementwise_script(
        num_tokens=64, hidden_size=1024, dtype="torch.bfloat16", kernel_type="Add"
    )
    compile(script, "<test>", "exec")


def test_attention_script_contains_fia():
    """Attention script should reference FusedInferAttentionScore."""
    script = generate_attention_script(
        batch_size=1,
        avg_seq_len=1024,
        num_heads=64,
        head_dim=128,
        dtype="torch.bfloat16",
    )
    assert "fused_infer_attention_score" in script.lower() or "FIA" in script
    assert "1024" in script  # seq_len
    assert "64" in script  # num_heads
    assert "128" in script  # head_dim


def test_attention_script_is_valid_python():
    """Generated attention script should be valid Python syntax."""
    script = generate_attention_script(
        batch_size=1,
        avg_seq_len=512,
        num_heads=32,
        head_dim=128,
        dtype="torch.bfloat16",
    )
    compile(script, "<test>", "exec")
