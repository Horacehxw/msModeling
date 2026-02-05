"""Stage 1: Analyze VLLM profiling data.

This stage parses VLLM profiling output, auto-detects the phase (prefill
or decode), resolves TensorCast execution parameters from the profile,
and prints the exact text_generate.py command the user would run.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config.loader import load_profile
from ..config.schema import ModelProfile, PhaseType, TensorCastConfig
from ..parsers.kernel_details_parser import KernelDetailsParser
from ..parsers.phase_detector import PhaseDetector


@dataclass
class AnalyzeResult:
    """Result of the analyze stage.

    Attributes:
        phase: Detected or specified phase
        tc_config: Resolved TensorCast configuration
        num_steps: Number of complete forward passes detected
        num_layers: Number of decoder layers used for step detection
        step_index: Recommended step index for comparison
        command_string: Full text_generate.py command for the user
    """

    phase: PhaseType
    tc_config: TensorCastConfig
    num_steps: int
    num_layers: Optional[int]
    step_index: int
    command_string: str


def run_analyze(
    vllm_dir: Path,
    profile_name: str,
    phase: PhaseType = PhaseType.AUTO,
    num_layers: Optional[int] = None,
    step_index: int = 0,
) -> AnalyzeResult:
    """Run Stage 1: Analyze VLLM profiling data.

    Args:
        vllm_dir: Path to ASCEND_PROFILER_OUTPUT directory containing kernel_details.csv
        profile_name: Model profile name (e.g., "qwen3-32b")
        phase: Phase type override (default: auto-detect)
        num_layers: Number of decoder layers override
        step_index: Step index for comparison

    Returns:
        AnalyzeResult with detected parameters and TC command

    Raises:
        FileNotFoundError: If profiling data or profile not found
    """
    # Load model profile
    profile = load_profile(profile_name)
    print(f"Loaded profile: {profile.name}")
    print(f"  Description: {profile.description}")

    # Resolve num_layers
    if num_layers is None:
        num_layers = profile.num_layers
    if num_layers:
        print(f"  num_layers: {num_layers}")

    # Find kernel_details.csv
    kernel_csv = vllm_dir / "kernel_details.csv"
    if not kernel_csv.exists():
        raise FileNotFoundError(f"kernel_details.csv not found at {kernel_csv}")

    # Auto-detect phase if needed
    detected_phase = phase
    if phase == PhaseType.AUTO:
        detector = PhaseDetector(kernel_csv)
        phase_info = detector.detect_phase()
        detected_phase = phase_info.phase
        print(f"\n  Detected phase: {detected_phase.value} "
              f"(confidence: {phase_info.confidence:.0%})")
        if phase_info.query_length:
            print(f"  Detected query length: {phase_info.query_length}")
    else:
        print(f"\n  Phase: {phase.value} (specified)")

    # Parse VLLM data and get step statistics
    parser = KernelDetailsParser(kernel_csv)
    stats = parser.get_step_statistics(num_layers)
    num_steps = stats["num_steps"]
    print(f"  Complete forward passes: {num_steps}")
    if num_steps > 0:
        print(f"  Avg step duration: {stats['avg_step_duration_us']:.2f} us")

    # Build TensorCast config from profile + detected phase
    tc_config = _resolve_tc_config(profile, detected_phase)

    # Build command string
    command_str = tc_config.to_command_string()
    print(f"\n=== TensorCast Command ===")
    print(command_str)

    return AnalyzeResult(
        phase=detected_phase,
        tc_config=tc_config,
        num_steps=num_steps,
        num_layers=num_layers,
        step_index=step_index,
        command_string=command_str,
    )


def _resolve_tc_config(profile: ModelProfile, phase: PhaseType) -> TensorCastConfig:
    """Resolve TensorCast config from profile and phase.

    Args:
        profile: Model profile
        phase: Resolved phase type

    Returns:
        TensorCastConfig with phase-appropriate defaults applied
    """
    tc = profile.tensorcast
    if phase == PhaseType.PREFILL:
        defaults = profile.prefill_defaults
    else:
        defaults = profile.decode_defaults

    return TensorCastConfig(
        model_id=tc.model_id,
        device=tc.device,
        world_size=tc.world_size,
        tp_size=tc.tp_size,
        dp_size=tc.dp_size,
        ep=tc.ep,
        quantize_linear_action=tc.quantize_linear_action,
        num_queries=defaults.num_queries,
        query_length=defaults.query_length,
        context_length=defaults.context_length,
        word_embedding_tp=tc.word_embedding_tp,
        lmhead_tp_size=tc.lmhead_tp_size,
        enable_external_shared_experts=tc.enable_external_shared_experts,
    )
