"""Phase detection module for PD Aggregation scenarios.

This module provides automatic detection of prefill/decode phases from
profiling data, which is essential for PD Aggregation where both phases
run on the same device.

Detection Algorithm:
1. Primary Method: Query length in ReshapeAndCacheNdKernel Input Shapes
   - query_len == 1 -> DECODE (99% confidence)
   - query_len > 100 -> PREFILL (98% confidence)

2. Secondary Method: MatMul batch dimension
   - M <= 256 -> DECODE (85% confidence)
   - M >= 512 -> PREFILL (90% confidence)

3. Warmup Detection: Batch size variation
   - batch_size = 1 -> WARMUP (skip)
   - batch_size >= 128 -> PRODUCTION
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..config.schema import PhaseType

from .kernel_details_parser import KernelDetailsParser, KernelOp


@dataclass
class PhaseInfo:
    """Information about detected phase.

    Attributes:
        phase: Detected phase type
        confidence: Detection confidence (0.0 to 1.0)
        method: Detection method used
        query_length: Detected query length (if available)
        batch_size: Detected batch size (if available)
        is_warmup: Whether this appears to be a warmup iteration
    """

    phase: PhaseType
    confidence: float
    method: str
    query_length: Optional[int] = None
    batch_size: Optional[int] = None
    is_warmup: bool = False


@dataclass
class StepBoundary:
    """Boundary information for a single step.

    Attributes:
        step_index: Index of this step
        start_time_us: Start time in microseconds
        end_time_us: End time in microseconds
        phase: Detected phase for this step
        is_warmup: Whether this is a warmup step
    """

    step_index: int
    start_time_us: float
    end_time_us: float
    phase: PhaseType
    is_warmup: bool


class PhaseDetector:
    """Automatic phase detector for profiling data.

    This class detects whether profiling data represents prefill or decode
    phase, which is essential for PD Aggregation scenarios where both phases
    run on the same device.
    """

    # Anchor operations for phase detection
    CACHE_OPS = ["ReshapeAndCacheNdKernel", "ScatterPaKvCache", "KvRmsNormRopeCache"]
    ATTENTION_OPS = ["FusedInferAttentionScore", "PagedAttention"]
    MATMUL_OPS = ["MatMulV2", "MatMul", "QuantBatchMatmulV3", "GroupedMatmul"]

    # Thresholds for phase classification
    DECODE_QUERY_LEN_THRESHOLD = 10  # <= this is decode
    PREFILL_QUERY_LEN_THRESHOLD = 100  # >= this is prefill
    WARMUP_BATCH_SIZE_THRESHOLD = 2  # <= this is warmup

    def __init__(self, kernel_details_path: Path):
        """Initialize phase detector.

        Args:
            kernel_details_path: Path to kernel_details.csv
        """
        self.kernel_details_path = Path(kernel_details_path)
        self._parser: Optional[KernelDetailsParser] = None
        self._operations: Optional[List[KernelOp]] = None

    @property
    def parser(self) -> KernelDetailsParser:
        """Get or create kernel details parser."""
        if self._parser is None:
            self._parser = KernelDetailsParser(self.kernel_details_path)
        return self._parser

    @property
    def operations(self) -> List[KernelOp]:
        """Get all parsed operations."""
        if self._operations is None:
            self._operations = self.parser.parse_all_operations()
        return self._operations

    def detect_phase(self) -> PhaseInfo:
        """Detect the primary phase type from profiling data.

        Returns:
            PhaseInfo with detected phase and confidence
        """
        # Try primary method: ReshapeAndCacheNdKernel query length
        phase_info = self._detect_from_cache_ops()
        if phase_info and phase_info.confidence >= 0.9:
            return phase_info

        # Try secondary method: MatMul batch dimension
        secondary_info = self._detect_from_matmul_ops()
        if secondary_info:
            # If primary had lower confidence, combine
            if phase_info and phase_info.phase == secondary_info.phase:
                # Same result, boost confidence
                return PhaseInfo(
                    phase=phase_info.phase,
                    confidence=min(0.99, phase_info.confidence + 0.1),
                    method=f"{phase_info.method} + {secondary_info.method}",
                    query_length=phase_info.query_length,
                    batch_size=secondary_info.batch_size,
                )
            elif secondary_info.confidence > (
                phase_info.confidence if phase_info else 0
            ):
                return secondary_info

        # Return best result or default
        if phase_info:
            return phase_info

        # Default to decode with low confidence
        return PhaseInfo(
            phase=PhaseType.DECODE,
            confidence=0.5,
            method="default",
        )

    def _detect_from_cache_ops(self) -> Optional[PhaseInfo]:
        """Detect phase from cache operation input shapes.

        The query_len field in ReshapeAndCacheNdKernel Input Shapes is the
        most reliable indicator of phase.

        Returns:
            PhaseInfo if detection successful, None otherwise
        """
        for op in self.operations:
            if any(cache_op in op.op_type for cache_op in self.CACHE_OPS):
                query_len = self._extract_query_length(op.input_shapes)
                batch_size = self._extract_batch_size(op.input_shapes)

                if query_len is not None:
                    is_warmup = (
                        batch_size is not None
                        and batch_size <= self.WARMUP_BATCH_SIZE_THRESHOLD
                    )

                    if query_len <= self.DECODE_QUERY_LEN_THRESHOLD:
                        return PhaseInfo(
                            phase=PhaseType.DECODE,
                            confidence=0.99,
                            method="cache_op_query_len",
                            query_length=query_len,
                            batch_size=batch_size,
                            is_warmup=is_warmup,
                        )
                    elif query_len >= self.PREFILL_QUERY_LEN_THRESHOLD:
                        return PhaseInfo(
                            phase=PhaseType.PREFILL,
                            confidence=0.98,
                            method="cache_op_query_len",
                            query_length=query_len,
                            batch_size=batch_size,
                            is_warmup=is_warmup,
                        )
                    else:
                        # Uncertain range (10 < query_len < 100)
                        # Could be chunked prefill or multi-token decode
                        return PhaseInfo(
                            phase=PhaseType.DECODE
                            if query_len < 50
                            else PhaseType.PREFILL,
                            confidence=0.7,
                            method="cache_op_query_len_uncertain",
                            query_length=query_len,
                            batch_size=batch_size,
                            is_warmup=is_warmup,
                        )

        return None

    def _detect_from_matmul_ops(self) -> Optional[PhaseInfo]:
        """Detect phase from MatMul batch dimension.

        Returns:
            PhaseInfo if detection successful, None otherwise
        """
        for op in self.operations:
            if any(mm_op in op.op_type for mm_op in self.MATMUL_OPS):
                m_dim = self._extract_m_dimension(op.input_shapes)
                if m_dim is not None:
                    if m_dim <= 256:
                        return PhaseInfo(
                            phase=PhaseType.DECODE,
                            confidence=0.85,
                            method="matmul_m_dim",
                            batch_size=m_dim,
                        )
                    elif m_dim >= 512:
                        return PhaseInfo(
                            phase=PhaseType.PREFILL,
                            confidence=0.90,
                            method="matmul_m_dim",
                            batch_size=m_dim,
                        )

        return None

    def _extract_query_length(self, input_shapes: str) -> Optional[int]:
        """Extract query length from input shapes string.

        Expected format: "batch_size,query_len,head_dim;..." or similar

        Args:
            input_shapes: Input shapes string from profiling

        Returns:
            Query length if found, None otherwise
        """
        if not input_shapes:
            return None

        # Try to parse shape strings
        # Format 1: "136,1,128;..." -> batch=136, query_len=1, head_dim=128
        # Format 2: "136,4096,5120;..." -> batch=136, query_len=4096, hidden=5120

        # Split by semicolon to get individual tensor shapes
        shapes = input_shapes.split(";")
        for shape in shapes:
            shape = shape.strip()
            if not shape:
                continue

            # Parse comma-separated dimensions
            dims = [d.strip() for d in shape.split(",")]
            if len(dims) >= 2:
                try:
                    # Second dimension is often query_len
                    query_len = int(dims[1])
                    # Validate: query_len should be reasonable (1 to 100000)
                    if 1 <= query_len <= 100000:
                        return query_len
                except (ValueError, IndexError):
                    continue

        return None

    def _extract_batch_size(self, input_shapes: str) -> Optional[int]:
        """Extract batch size from input shapes string.

        Args:
            input_shapes: Input shapes string from profiling

        Returns:
            Batch size if found, None otherwise
        """
        if not input_shapes:
            return None

        shapes = input_shapes.split(";")
        for shape in shapes:
            shape = shape.strip()
            if not shape:
                continue

            dims = [d.strip() for d in shape.split(",")]
            if dims:
                try:
                    # First dimension is usually batch size
                    batch_size = int(dims[0])
                    if 1 <= batch_size <= 10000:
                        return batch_size
                except (ValueError, IndexError):
                    continue

        return None

    def _extract_m_dimension(self, input_shapes: str) -> Optional[int]:
        """Extract M dimension from MatMul input shapes.

        For MatMul(M,K) x (K,N) -> (M,N), M is the batch*seq dimension.

        Args:
            input_shapes: Input shapes string from profiling

        Returns:
            M dimension if found, None otherwise
        """
        if not input_shapes:
            return None

        shapes = input_shapes.split(";")
        for shape in shapes:
            shape = shape.strip()
            if not shape:
                continue

            dims = [d.strip() for d in shape.split(",")]
            if len(dims) >= 2:
                try:
                    m_dim = int(dims[0])
                    if 1 <= m_dim <= 1000000:
                        return m_dim
                except (ValueError, IndexError):
                    continue

        return None

    def find_step_boundaries_by_phase(self) -> List[StepBoundary]:
        """Find step boundaries with phase information.

        This method identifies step boundaries using anchor operations
        and annotates each step with its detected phase.

        Returns:
            List of StepBoundary objects
        """
        # Use attention ops as anchors (one per step)
        anchor_times = []
        for op in self.operations:
            if any(att_op in op.op_type for att_op in self.ATTENTION_OPS):
                anchor_times.append(op.start_time_us)

        anchor_times = sorted(anchor_times)

        if not anchor_times:
            return []

        boundaries = []
        for i, start_time in enumerate(anchor_times):
            end_time = (
                anchor_times[i + 1] if i + 1 < len(anchor_times) else start_time + 10000
            )

            # Detect phase for this step
            step_ops = [
                op
                for op in self.operations
                if start_time <= op.start_time_us < end_time
            ]
            phase_info = self._detect_phase_for_ops(step_ops)

            boundaries.append(
                StepBoundary(
                    step_index=i,
                    start_time_us=start_time,
                    end_time_us=end_time,
                    phase=phase_info.phase if phase_info else PhaseType.DECODE,
                    is_warmup=phase_info.is_warmup if phase_info else False,
                )
            )

        return boundaries

    def _detect_phase_for_ops(self, ops: List[KernelOp]) -> Optional[PhaseInfo]:
        """Detect phase for a subset of operations.

        Args:
            ops: List of operations to analyze

        Returns:
            PhaseInfo if detection successful
        """
        for op in ops:
            if any(cache_op in op.op_type for cache_op in self.CACHE_OPS):
                query_len = self._extract_query_length(op.input_shapes)
                batch_size = self._extract_batch_size(op.input_shapes)

                if query_len is not None:
                    is_warmup = (
                        batch_size is not None
                        and batch_size <= self.WARMUP_BATCH_SIZE_THRESHOLD
                    )

                    if query_len <= self.DECODE_QUERY_LEN_THRESHOLD:
                        return PhaseInfo(
                            phase=PhaseType.DECODE,
                            confidence=0.99,
                            method="step_cache_op",
                            query_length=query_len,
                            batch_size=batch_size,
                            is_warmup=is_warmup,
                        )
                    elif query_len >= self.PREFILL_QUERY_LEN_THRESHOLD:
                        return PhaseInfo(
                            phase=PhaseType.PREFILL,
                            confidence=0.98,
                            method="step_cache_op",
                            query_length=query_len,
                            batch_size=batch_size,
                            is_warmup=is_warmup,
                        )

        return None

    def get_production_decode_steps(self) -> List[int]:
        """Get indices of production (non-warmup) decode steps.

        Returns:
            List of step indices suitable for comparison
        """
        boundaries = self.find_step_boundaries_by_phase()
        return [
            b.step_index
            for b in boundaries
            if b.phase == PhaseType.DECODE and not b.is_warmup
        ]

    def get_recommended_step_index(self) -> int:
        """Get recommended step index for comparison.

        This returns a step index that is:
        1. Not a warmup step
        2. In the stable region (not first 10 steps)
        3. Of the detected phase type

        Returns:
            Recommended step index (default 100 if detection fails)
        """
        production_steps = self.get_production_decode_steps()

        if not production_steps:
            return 100  # Default fallback

        # Skip first 10 production steps to avoid startup effects
        if len(production_steps) > 20:
            return production_steps[10]
        elif len(production_steps) > 5:
            return production_steps[len(production_steps) // 2]
        else:
            return production_steps[-1]
