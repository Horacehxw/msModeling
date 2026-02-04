"""Configuration dataclasses for profiling comparison tool.

This module defines the configuration schema for comparing VLLM profiling
data with TensorCast simulation results.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class PhaseType(str, Enum):
    """Phase type for PD aggregation/disaggregation."""

    AUTO = "auto"       # Auto-detect from profiling data
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class TensorCastConfig:
    """Configuration for TensorCast simulation.

    Attributes:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-32B")
        device: Device profile name (e.g., "ATLAS_800_A3_752T_128G_DIE")
        world_size: Total number of devices
        tp_size: Tensor parallelism size
        dp_size: Data parallelism size
        ep: Enable expert parallelism (for MoE models)
        quantize_linear_action: Quantization scheme
        num_queries: Number of queries / batch size
        query_length: Input sequence length
        context_length: Context length for decode phase
        word_embedding_tp: Enable word embedding tensor parallelism
        lmhead_tp_size: LM head tensor parallelism size
        enable_external_shared_experts: Enable external shared experts for MoE
    """

    model_id: str
    device: str
    world_size: int = 1
    tp_size: int = 1
    dp_size: int = 1
    ep: bool = False
    quantize_linear_action: str = "DISABLED"
    num_queries: int = 1
    query_length: int = 1
    context_length: int = 0
    word_embedding_tp: bool = False
    lmhead_tp_size: int = 1
    enable_external_shared_experts: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary for TensorCastParser."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "world_size": self.world_size,
            "tp_size": self.tp_size,
            "dp_size": self.dp_size,
            "ep": self.ep,
            "quantize_linear_action": self.quantize_linear_action,
            "num_queries": self.num_queries,
            "query_length": self.query_length,
            "context_length": self.context_length,
            "word_embedding_tp": self.word_embedding_tp,
            "lmhead_tp_size": self.lmhead_tp_size,
            "enable_external_shared_experts": self.enable_external_shared_experts,
        }


@dataclass
class VLLMConfig:
    """Configuration for VLLM profiling data.

    Attributes:
        profiling_dir: Path to ASCEND_PROFILER_OUTPUT directory
        num_output_tokens: Number of output tokens generated
        step_index: Decode step index to extract (for single-step mode)
        phase: Phase type (auto, prefill, decode)
    """

    profiling_dir: Optional[Path] = None
    num_output_tokens: int = 1
    step_index: int = 100  # Default to step 100 to avoid warmup
    phase: PhaseType = PhaseType.AUTO


@dataclass
class OutputConfig:
    """Configuration for output generation.

    Attributes:
        output_path: Path for output file
        format: Output format ("excel", "json", "console")
        include_unmatched: Include unmatched operations in output
        include_shapes: Include input/output shapes in output
    """

    output_path: Optional[Path] = None
    format: str = "excel"
    include_unmatched: bool = True
    include_shapes: bool = True


@dataclass
class ProfileDefaults:
    """Default values for prefill and decode phases.

    Attributes:
        num_queries: Default batch size
        query_length: Default query length
        context_length: Default context length
    """

    num_queries: int = 1
    query_length: int = 1
    context_length: int = 0


@dataclass
class ModelProfile:
    """Complete model profile for profiling comparison.

    Attributes:
        name: Profile name (e.g., "qwen3-32b")
        description: Human-readable description
        tensorcast: TensorCast configuration
        prefill_defaults: Default values for prefill phase
        decode_defaults: Default values for decode phase
        mapping_file: Optional custom mapping file name
    """

    name: str
    tensorcast: TensorCastConfig
    prefill_defaults: ProfileDefaults = field(default_factory=ProfileDefaults)
    decode_defaults: ProfileDefaults = field(default_factory=ProfileDefaults)
    description: str = ""
    mapping_file: Optional[str] = None


@dataclass
class ComparisonConfig:
    """Complete configuration for a profiling comparison run.

    Attributes:
        tensorcast: TensorCast simulation configuration
        vllm: VLLM profiling configuration
        output: Output configuration
        mode: Comparison mode ("single-step", "full")
        mapping_file: Optional custom mapping file
    """

    tensorcast: TensorCastConfig
    vllm: VLLMConfig
    output: OutputConfig
    mode: str = "single-step"
    mapping_file: Optional[str] = None

    @classmethod
    def from_profile(
        cls,
        profile: ModelProfile,
        vllm_dir: Path,
        output_path: Optional[Path] = None,
        phase: PhaseType = PhaseType.AUTO,
        step_index: int = 100,
        output_format: str = "excel",
        mode: str = "single-step",
    ) -> "ComparisonConfig":
        """Create ComparisonConfig from a ModelProfile.

        Args:
            profile: Model profile to use
            vllm_dir: Path to VLLM profiling directory
            output_path: Output file path (optional)
            phase: Phase type override
            step_index: Decode step index
            output_format: Output format
            mode: Comparison mode

        Returns:
            ComparisonConfig instance
        """
        # Start with base tensorcast config from profile
        tc_config = TensorCastConfig(
            model_id=profile.tensorcast.model_id,
            device=profile.tensorcast.device,
            world_size=profile.tensorcast.world_size,
            tp_size=profile.tensorcast.tp_size,
            dp_size=profile.tensorcast.dp_size,
            ep=profile.tensorcast.ep,
            quantize_linear_action=profile.tensorcast.quantize_linear_action,
            word_embedding_tp=profile.tensorcast.word_embedding_tp,
            lmhead_tp_size=profile.tensorcast.lmhead_tp_size,
            enable_external_shared_experts=profile.tensorcast.enable_external_shared_experts,
        )

        # Apply phase-specific defaults
        if phase == PhaseType.DECODE or (phase == PhaseType.AUTO):
            # Default to decode for single-step mode
            defaults = profile.decode_defaults
            tc_config.num_queries = defaults.num_queries
            tc_config.query_length = defaults.query_length
            tc_config.context_length = defaults.context_length
        else:
            defaults = profile.prefill_defaults
            tc_config.num_queries = defaults.num_queries
            tc_config.query_length = defaults.query_length
            tc_config.context_length = defaults.context_length

        # Create VLLM config
        vllm_config = VLLMConfig(
            profiling_dir=vllm_dir,
            step_index=step_index,
            phase=phase,
        )

        # Create output config
        if output_path is None:
            output_path = Path(f"{profile.name}_comparison.{output_format}")

        output_config = OutputConfig(
            output_path=output_path,
            format=output_format,
        )

        return cls(
            tensorcast=tc_config,
            vllm=vllm_config,
            output=output_config,
            mode=mode,
            mapping_file=profile.mapping_file,
        )
