"""Stages for the 3-stage profiling comparison pipeline.

Stage 1 (analyze): Parse VLLM profiling, detect phase, print TC command
Stage 2 (simulate): Run TensorCast simulation as subprocess, produce chrome trace
Stage 3 (compare): Sequence-match VLLM ops vs TC trace, produce Excel report
"""

from .analyze import run_analyze
from .compare import run_compare
from .simulate import run_simulate

__all__ = [
    "run_analyze",
    "run_simulate",
    "run_compare",
]
