"""Tests for generate_comm_microbench.py."""

from tools.perf_data_collection.generate_comm_microbench import generate_comm_script


def test_allreduce_script_contains_dist_call():
    """AllReduce script should use torch.distributed.all_reduce."""
    script = generate_comm_script(
        op_type="all_reduce", message_bytes=1048576, num_devices=8,
        topology_tier=1, group_ranks=list(range(8)),
    )
    assert "all_reduce" in script
    assert "1048576" in script


def test_allgather_script():
    """AllGather script should use torch.distributed.all_gather."""
    script = generate_comm_script(
        op_type="all_gather", message_bytes=65536, num_devices=16,
        topology_tier=1, group_ranks=list(range(16)),
    )
    assert "all_gather" in script
    assert "65536" in script


def test_alltoall_script():
    """AllToAll script should use torch.distributed.all_to_all."""
    script = generate_comm_script(
        op_type="all_to_all", message_bytes=262144, num_devices=8,
        topology_tier=1, group_ranks=list(range(8)),
    )
    assert "all_to_all" in script


def test_script_is_valid_python():
    """Generated script should be valid Python syntax."""
    script = generate_comm_script(
        op_type="all_reduce", message_bytes=1048576, num_devices=8,
        topology_tier=1, group_ranks=list(range(8)),
    )
    compile(script, "<test>", "exec")


def test_script_has_warmup_and_timing():
    """Script should include warmup and timing."""
    script = generate_comm_script(
        op_type="all_reduce", message_bytes=1048576, num_devices=8,
        topology_tier=1, group_ranks=list(range(8)),
    )
    assert "warmup" in script.lower() or "WARMUP" in script
    assert "Duration" in script or "duration" in script


def test_script_has_hccl_init():
    """Script should initialize HCCL backend."""
    script = generate_comm_script(
        op_type="all_reduce", message_bytes=1048576, num_devices=8,
        topology_tier=1, group_ranks=list(range(8)),
    )
    assert (
        "hccl" in script.lower()
        or "nccl" in script.lower()
        or "init_process_group" in script
    )


def test_script_has_rank_group():
    """Script should handle rank groups."""
    script = generate_comm_script(
        op_type="all_reduce", message_bytes=1048576, num_devices=8,
        topology_tier=1, group_ranks=list(range(8)),
    )
    assert "group" in script.lower() or "rank" in script.lower()
