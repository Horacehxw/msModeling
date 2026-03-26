import csv
from tools.perf_data_collection.fill_fia_runtime_metadata import (
    build_csv_signature,
    build_json_signature,
)
from tools.perf_data_collection.parse_kernel_details import (
    EXTRA_NUMERIC_COLUMNS,
    KernelDetailsParser,
)


def _base_kernel_row(input_shapes: str, output_shapes: str) -> dict[str, str]:
    row = {
        "Type": "FusedInferAttentionScore",
        "OP State": "dynamic",
        "Accelerator Core": "MIX_AIC",
        "Input Shapes": input_shapes,
        "Input Data Types": "DT_BF16;DT_BF16;DT_BF16",
        "Input Formats": "ND;ND;ND",
        "Output Shapes": output_shapes,
        "Output Data Types": "DT_BF16;FLOAT",
        "Output Formats": "ND;ND",
        "Duration(us)": "1.0",
    }
    for column in EXTRA_NUMERIC_COLUMNS:
        row[column] = "0"
    return row


class TestFiaOperatorDetailsEnrichment:
    def test_parse_kernel_details_prefers_operator_raw_shapes_for_mla(self, tmp_path):
        profile_dir = tmp_path / "profile_a"
        profile_dir.mkdir()

        kernel_csv = profile_dir / "kernel_details.csv"
        operator_csv = profile_dir / "operator_details.csv"

        kernel_row = _base_kernel_row(
            "8192,16,128;8192,16,128;8192,16,128;;2048,2048;2;2;;;;;;;;;;;;;;;;;;8192,16,64;8192,16,64;;;;;",
            "8192,16,128;8192,16,1",
        )

        with kernel_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(kernel_row.keys()))
            writer.writeheader()
            writer.writerow(kernel_row)

        operator_row = {
            "Type": "aclnnFusedInferAttentionScoreV2",
            "Name": "npu_fused_infer_attention_score_v2",
            "Input Shapes": (
                "5,16,1,512;1171,1,128,512;1171,1,128,512;"
                ";;;;;;;;;;;;5,512;;;;;;;;;;5,16,1,64;1171,1,128,64"
            ),
            "Output Shapes": "5,16,1,512;5,16,1,1",
        }
        with operator_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(operator_row.keys()))
            writer.writeheader()
            writer.writerow(operator_row)

        parser = KernelDetailsParser(
            device="TEST_DEVICE",
            kernel_details_path=str(tmp_path),
            vllm_ascend_version="0.9.2",
        )
        parser.output_dir = tmp_path / "out"
        output_files = parser.parse_and_export()

        output_csv = next(
            path
            for path in output_files
            if path.name == "FusedInferAttentionScore.csv"
        )
        with output_csv.open("r", encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))

        assert "Runtime operator_input_shapes_raw" in row
        assert row["Runtime operator_input_shapes_raw"].startswith(
            "5,16,1,512;1171,1,128,512"
        )
        assert row["Runtime input_layout"] == "BNSD_NBSD"
        assert row["Runtime num_key_value_heads"] == "1"
        assert row["Runtime attn_state"] == "mla_paged_runtime"


class TestFiaBackfillSignature:
    def test_backfill_signature_prefers_operator_raw_shapes(self):
        csv_row = {
            "Input Shapes": "8192,16,128;8192,16,128;8192,16,128",
            "Runtime operator_input_shapes_raw": "5,16,1,512;1171,1,128,512;1171,1,128,512",
            "Runtime block_table_shape": "5,512",
            "Runtime num_heads": "16",
            "Runtime num_key_value_heads": "1",
            "Runtime input_layout": "BNSD_NBSD",
            "Runtime sparse_mode": "0",
            "Runtime block_size": "128",
        }
        json_record = {
            "query_shape": [5, 16, 1, 512],
            "key_shape": [1171, 1, 128, 512],
            "value_shape": [1171, 1, 128, 512],
            "block_table_shape": [5, 512],
            "num_heads": 16,
            "num_key_value_heads": 1,
            "input_layout": "BNSD_NBSD",
            "sparse_mode": 0,
            "block_size": 128,
        }

        assert build_csv_signature(csv_row) == build_json_signature(json_record)
