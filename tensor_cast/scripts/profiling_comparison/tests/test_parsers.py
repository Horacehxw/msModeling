"""Tests for parser modules."""

import csv
import tempfile
from pathlib import Path

import pytest

from tensor_cast.scripts.profiling_comparison.config import PhaseType

from tensor_cast.scripts.profiling_comparison.parsers import (
    KernelDetailsParser,
    KernelOp,
    SingleStepData,
)
from tensor_cast.scripts.profiling_comparison.parsers.phase_detector import (
    PhaseDetector,
    PhaseInfo,
)


class TestKernelOp:
    """Tests for KernelOp dataclass."""

    def test_is_attention_anchor(self):
        """Test attention anchor detection."""
        op1 = KernelOp(
            name="kernel1",
            op_type="FusedInferAttentionScore",
            start_time_us=1000.0,
            duration_us=500.0,
            end_time_us=1500.0,
            core_type="AI_CORE",
            input_shapes="136,1,128",
            output_shapes="136,1,128",
        )
        assert op1.is_attention_anchor is True

        op2 = KernelOp(
            name="kernel2",
            op_type="MatMulV2",
            start_time_us=2000.0,
            duration_us=300.0,
            end_time_us=2300.0,
            core_type="AI_CORE",
            input_shapes="136,4096",
            output_shapes="136,5120",
        )
        assert op2.is_attention_anchor is False


class TestSingleStepData:
    """Tests for SingleStepData dataclass."""

    def test_total_duration_calculation(self):
        """Test that total duration is calculated from operations."""
        ops = [
            KernelOp("k1", "Op1", 0, 100, 100, "", "", ""),
            KernelOp("k2", "Op2", 100, 200, 300, "", "", ""),
        ]
        step = SingleStepData(
            step_index=0,
            operations=ops,
            start_time_us=0,
            end_time_us=300,
            total_duration_us=0,  # Will be recalculated
        )
        assert step.total_duration_us == 300  # 100 + 200

    def test_get_ops_by_type(self):
        """Test grouping operations by type."""
        ops = [
            KernelOp("k1", "MatMul", 0, 100, 100, "", "", ""),
            KernelOp("k2", "MatMul", 100, 100, 200, "", "", ""),
            KernelOp("k3", "Add", 200, 50, 250, "", "", ""),
        ]
        step = SingleStepData(0, ops, 0, 250, 0)

        by_type = step.get_ops_by_type()
        assert len(by_type["MatMul"]) == 2
        assert len(by_type["Add"]) == 1

    def test_get_aggregated_stats(self):
        """Test aggregated statistics calculation."""
        ops = [
            KernelOp("k1", "MatMul", 0, 100, 100, "CORE", "in1", "out1"),
            KernelOp("k2", "MatMul", 100, 150, 250, "CORE", "in2", "out2"),
        ]
        step = SingleStepData(0, ops, 0, 250, 0)

        stats = step.get_aggregated_stats()
        assert stats["MatMul"]["count"] == 2
        assert stats["MatMul"]["total_duration_us"] == 250
        assert stats["MatMul"]["avg_duration_us"] == 125


class TestKernelDetailsParser:
    """Tests for KernelDetailsParser class."""

    def _create_test_csv(self, rows):
        """Create a temporary CSV file for testing."""
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        )
        writer = csv.writer(f)
        writer.writerow(
            [
                "Name",
                "Type",
                "Start Time(us)",
                "Duration(us)",
                "Accelerator Core",
                "Input Shapes",
                "Output Shapes",
            ]
        )
        for row in rows:
            writer.writerow(row)
        f.flush()
        return Path(f.name)

    def test_parse_all_operations(self):
        """Test parsing all operations from CSV."""
        csv_path = self._create_test_csv(
            [
                [
                    "kernel1",
                    "MatMulV2",
                    "1000",
                    "500",
                    "AI_CORE",
                    "136,4096",
                    "136,5120",
                ],
                ["kernel2", "Add", "1500", "100", "AI_CORE", "136,5120", "136,5120"],
            ]
        )

        parser = KernelDetailsParser(csv_path)
        ops = parser.parse_all_operations()

        assert len(ops) == 2
        assert ops[0].op_type == "MatMulV2"
        assert ops[0].start_time_us == 1000
        assert ops[1].op_type == "Add"

    def test_find_step_boundaries(self):
        """Test finding step boundaries using attention anchor."""
        csv_path = self._create_test_csv(
            [
                ["k1", "MatMulV2", "100", "50", "", "", ""],
                ["k2", "FusedInferAttentionScore", "200", "100", "", "", ""],
                ["k3", "MatMulV2", "400", "50", "", "", ""],
                ["k4", "FusedInferAttentionScore", "500", "100", "", "", ""],
            ]
        )

        parser = KernelDetailsParser(csv_path)
        boundaries = parser.find_step_boundaries()

        assert len(boundaries) == 2
        assert boundaries[0] == 200
        assert boundaries[1] == 500

    def test_extract_single_step(self):
        """Test extracting a single step."""
        csv_path = self._create_test_csv(
            [
                ["k1", "FusedInferAttentionScore", "100", "50", "", "", ""],
                ["k2", "MatMulV2", "150", "100", "", "", ""],
                ["k3", "FusedInferAttentionScore", "300", "50", "", "", ""],
                ["k4", "MatMulV2", "350", "100", "", "", ""],
            ]
        )

        parser = KernelDetailsParser(csv_path)
        step = parser.extract_single_step(0)

        assert step.step_index == 0
        assert step.start_time_us == 100
        assert step.end_time_us == 300
        assert len(step.operations) == 2

    def test_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            KernelDetailsParser(Path("/nonexistent/path.csv"))


class TestPhaseDetector:
    """Tests for PhaseDetector class."""

    def _create_test_csv(self, rows):
        """Create a temporary CSV file for testing."""
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        )
        writer = csv.writer(f)
        writer.writerow(
            [
                "Name",
                "Type",
                "Start Time(us)",
                "Duration(us)",
                "Accelerator Core",
                "Input Shapes",
                "Output Shapes",
            ]
        )
        for row in rows:
            writer.writerow(row)
        f.flush()
        return Path(f.name)

    def test_detect_decode_phase(self):
        """Test detecting decode phase from query_len=1."""
        csv_path = self._create_test_csv(
            [
                [
                    "k1",
                    "ReshapeAndCacheNdKernel",
                    "100",
                    "50",
                    "",
                    "136,1,128;136,64,128",
                    "",
                ],
                ["k2", "FusedInferAttentionScore", "200", "100", "", "", ""],
            ]
        )

        detector = PhaseDetector(csv_path)
        phase_info = detector.detect_phase()

        assert phase_info.phase == PhaseType.DECODE
        assert phase_info.confidence >= 0.9

    def test_detect_prefill_phase(self):
        """Test detecting prefill phase from query_len>100."""
        csv_path = self._create_test_csv(
            [
                [
                    "k1",
                    "ReshapeAndCacheNdKernel",
                    "100",
                    "50",
                    "",
                    "136,4096,128;136,64,128",
                    "",
                ],
                ["k2", "FusedInferAttentionScore", "200", "100", "", "", ""],
            ]
        )

        detector = PhaseDetector(csv_path)
        phase_info = detector.detect_phase()

        assert phase_info.phase == PhaseType.PREFILL
        assert phase_info.confidence >= 0.9

    def test_extract_query_length(self):
        """Test extracting query length from input shapes."""
        csv_path = self._create_test_csv(
            [
                ["k1", "MatMulV2", "100", "50", "", "136,4096", ""],
            ]
        )

        detector = PhaseDetector(csv_path)
        query_len = detector._extract_query_length("136,4096,128")
        assert query_len == 4096

    def test_extract_batch_size(self):
        """Test extracting batch size from input shapes."""
        csv_path = self._create_test_csv(
            [
                ["k1", "MatMulV2", "100", "50", "", "136,4096", ""],
            ]
        )

        detector = PhaseDetector(csv_path)
        batch_size = detector._extract_batch_size("136,4096,128")
        assert batch_size == 136


class TestPhaseInfo:
    """Tests for PhaseInfo dataclass."""

    def test_basic_creation(self):
        """Test basic PhaseInfo creation."""
        info = PhaseInfo(
            phase=PhaseType.DECODE,
            confidence=0.99,
            method="cache_op_query_len",
            query_length=1,
            batch_size=136,
        )
        assert info.phase == PhaseType.DECODE
        assert info.confidence == 0.99
        assert info.query_length == 1

    def test_warmup_detection(self):
        """Test warmup detection field."""
        info = PhaseInfo(
            phase=PhaseType.DECODE,
            confidence=0.9,
            method="test",
            batch_size=1,
            is_warmup=True,
        )
        assert info.is_warmup is True
