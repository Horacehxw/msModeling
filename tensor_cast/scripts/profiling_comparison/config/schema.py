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

    AUTO = "auto"  # Auto-detect from profiling data
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

    def to_cli_args(self, chrome_trace_path: Optional[Path] = None) -> List[str]:
        """Build CLI arguments for text_generate.py subprocess.

        Args:
            chrome_trace_path: Optional path for --chrome-trace output

        Returns:
            List of command-line argument strings
        """
        args = [
            self.model_id,
            "--device", self.device,
            "--world-size", str(self.world_size),
            "--tp-size", str(self.tp_size),
            "--dp-size", str(self.dp_size),
            "--num-queries", str(self.num_queries),
            "--query-length", str(self.query_length),
            "--context-length", str(self.context_length),
        ]
        if self.ep:
            args.append("--ep")
        if self.quantize_linear_action != "DISABLED":
            args.extend(["--quantize-linear-action", self.quantize_linear_action])
        if self.word_embedding_tp:
            args.extend(["--word-embedding-tp", str(self.word_embedding_tp)])
        if self.lmhead_tp_size > 1:
            args.extend(["--lmhead-tp-size", str(self.lmhead_tp_size)])
        if self.enable_external_shared_experts:
            args.append("--enable-external-shared-experts")
        if chrome_trace_path is not None:
            args.extend(["--chrome-trace", str(chrome_trace_path)])
        return args

    def to_command_string(self, chrome_trace_path: Optional[Path] = None) -> str:
        """Build a human-readable command string for text_generate.py.

        Args:
            chrome_trace_path: Optional path for --chrome-trace output

        Returns:
            Full command string including python -m prefix
        """
        args = self.to_cli_args(chrome_trace_path)
        return "python -m tensor_cast.scripts.text_generate " + " ".join(args)


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
        num_layers: Number of decoder layers (for step detection)
    """

    name: str
    tensorcast: TensorCastConfig
    prefill_defaults: ProfileDefaults = field(default_factory=ProfileDefaults)
    decode_defaults: ProfileDefaults = field(default_factory=ProfileDefaults)
    description: str = ""
    mapping_file: Optional[str] = None
    num_layers: Optional[int] = None
