"""
Validate all paper-ready tables, figures, and key numerical results.

Outputs:
    results/validation/paper_output_validation_report.txt
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

TABLE_DIRECTORY = (
    ROOT
    / "results"
    / "tables"
)

FIGURE_DIRECTORY = (
    ROOT
    / "results"
    / "figures"
)

CSV_DIRECTORY = (
    ROOT
    / "results"
    / "csv"
)

REPORT_PATH = (
    ROOT
    / "results"
    / "validation"
    / "paper_output_validation_report.txt"
)

TABLE_PATHS = (
    TABLE_DIRECTORY
    / "Table1_dataset_statistics.csv",
    TABLE_DIRECTORY
    / "Table2_occlusion_validation.csv",
    TABLE_DIRECTORY
    / "Table3_noise_validation.csv",
    TABLE_DIRECTORY
    / "Table4_main_results.csv",
    TABLE_DIRECTORY
    / "Table5_ablation_results.csv",
    TABLE_DIRECTORY
    / "Table6_cn2_distance_results.csv",
    TABLE_DIRECTORY
    / "Table7_statistical_significance.csv",
)

FIGURE_PATHS = (
    FIGURE_DIRECTORY
    / "fig_method_overall_accuracy.png",
    FIGURE_DIRECTORY
    / "fig_method_overall_accuracy.pdf",
    FIGURE_DIRECTORY
    / "fig_accuracy_vs_occlusion.png",
    FIGURE_DIRECTORY
    / "fig_accuracy_vs_occlusion.pdf",
    FIGURE_DIRECTORY
    / "fig_accuracy_vs_snr.png",
    FIGURE_DIRECTORY
    / "fig_accuracy_vs_snr.pdf",
    FIGURE_DIRECTORY
    / "fig_mc_vwls_cn2_distance_heatmap.png",
    FIGURE_DIRECTORY
    / "fig_mc_vwls_cn2_distance_heatmap.pdf",
)

ABLATION_SUMMARY_PATH = (
    CSV_DIRECTORY
    / "mc_vwls_ablation_summary.csv"
)

PAIRWISE_PATH = (
    CSV_DIRECTORY
    / "ablation_pairwise_statistics.csv"
)

CN2_DISTANCE_PATH = (
    CSV_DIRECTORY
    / "ablation_by_cn2_distance.csv"
)

EXPECTED_OVERALL_RESULTS = {
    "A0_DAF": {
        "label_accuracy": 0.84565476,
        "order_accuracy": 0.89205357,
        "phase_accuracy": 0.85589286,
    },
    "A1_MASK_ULS": {
        "label_accuracy": 0.84857143,
        "order_accuracy": 0.89348214,
        "phase_accuracy": 0.85830357,
    },
    "A2_RAW_VWLS": {
        "label_accuracy": 0.84571429,
        "order_accuracy": 0.89318452,
        "phase_accuracy": 0.85565476,
    },
    "A3_MC_VWLS": {
        "label_accuracy": 0.84812500,
        "order_accuracy": 0.89345238,
        "phase_accuracy": 0.85791667,
    },
}

METHOD_LABEL_TO_INTERNAL = {
    "DAF": "A0_DAF",
    "Mask-ULS": "A1_MASK_ULS",
    "Raw-VWLS": "A2_RAW_VWLS",
    "MC-VWLS": "A3_MC_VWLS",
}

FLOAT_TOLERANCE = 1.0e-8


def load_csv(
    path: Path,
) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"CSV file does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        rows = list(
            csv.DictReader(
                csv_file
            )
        )

    if not rows:
        raise ValueError(
            f"CSV contains no data rows: {path}"
        )

    return rows


def require_file(
    path: Path,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required output does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Required output is not a file: {path}"
        )

    if path.stat().st_size <= 0:
        raise ValueError(
            f"Required output is empty: {path}"
        )


def find_overall_ablation_row(
    rows: List[Dict[str, str]],
    method: str,
) -> Dict[str, str]:
    matches = [
        row
        for row in rows
        if (
            row["method"] == method
            and row["scope"] == "overall"
        )
    ]

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one overall ablation row, "
            f"found {len(matches)} for method={method}"
        )

    return matches[0]


def find_main_table_row(
    rows: List[Dict[str, str]],
    method_label: str,
) -> Dict[str, str]:
    matches = [
        row
        for row in rows
        if row["Method"] == method_label
    ]

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one main-table row, "
            f"found {len(matches)} for method={method_label}"
        )

    return matches[0]


def assert_close(
    actual: float,
    expected: float,
    *,
    name: str,
) -> None:
    if not np.isclose(
        actual,
        expected,
        rtol=0.0,
        atol=FLOAT_TOLERANCE,
    ):
        raise ValueError(
            f"Numerical mismatch for {name}: "
            f"actual={actual:.12f}, "
            f"expected={expected:.12f}"
        )


def validate_files(
    report_lines: List[str],
) -> None:
    report_lines.append(
        "[FILE EXISTENCE]"
    )

    for path in TABLE_PATHS:
        require_file(
            path
        )

        report_lines.append(
            f"PASS table: {path.name}"
        )

    for path in FIGURE_PATHS:
        require_file(
            path
        )

        report_lines.append(
            f"PASS figure: {path.name}"
        )

    report_lines.append("")


def validate_table_row_counts(
    report_lines: List[str],
) -> None:
    expected_counts = {
        "Table1_dataset_statistics.csv": 15,
        "Table2_occlusion_validation.csv": 5,
        "Table3_noise_validation.csv": 5,
        "Table4_main_results.csv": 4,
        "Table5_ablation_results.csv": 4,
        "Table6_cn2_distance_results.csv": 28,
        "Table7_statistical_significance.csv": 6,
    }

    report_lines.append(
        "[TABLE ROW COUNTS]"
    )

    for path in TABLE_PATHS:
        rows = load_csv(
            path
        )

        expected_count = expected_counts[
            path.name
        ]

        actual_count = len(
            rows
        )

        if actual_count != expected_count:
            raise ValueError(
                f"Unexpected row count for {path.name}: "
                f"expected={expected_count}, "
                f"actual={actual_count}"
            )

        report_lines.append(
            f"PASS {path.name}: rows={actual_count}"
        )

    report_lines.append("")


def validate_main_results(
    report_lines: List[str],
) -> None:
    ablation_rows = load_csv(
        ABLATION_SUMMARY_PATH
    )

    main_table_rows = load_csv(
        TABLE_DIRECTORY
        / "Table4_main_results.csv"
    )

    report_lines.append(
        "[MAIN RESULT CONSISTENCY]"
    )

    for method, expected_values in (
        EXPECTED_OVERALL_RESULTS.items()
    ):
        raw_row = find_overall_ablation_row(
            ablation_rows,
            method,
        )

        raw_label_accuracy = float(
            raw_row[
                "label_accuracy"
            ]
        )

        raw_order_accuracy = float(
            raw_row[
                "order_accuracy"
            ]
        )

        raw_phase_accuracy = float(
            raw_row[
                "phase_accuracy"
            ]
        )

        assert_close(
            raw_label_accuracy,
            expected_values[
                "label_accuracy"
            ],
            name=(
                f"{method} raw label accuracy"
            ),
        )

        assert_close(
            raw_order_accuracy,
            expected_values[
                "order_accuracy"
            ],
            name=(
                f"{method} raw order accuracy"
            ),
        )

        assert_close(
            raw_phase_accuracy,
            expected_values[
                "phase_accuracy"
            ],
            name=(
                f"{method} raw phase accuracy"
            ),
        )

        method_label = next(
            label
            for label, internal_name
            in METHOD_LABEL_TO_INTERNAL.items()
            if internal_name == method
        )

        table_row = find_main_table_row(
            main_table_rows,
            method_label,
        )

        table_label_accuracy = float(
            table_row[
                "Label accuracy"
            ]
        )

        table_order_accuracy = float(
            table_row[
                "Order accuracy"
            ]
        )

        table_phase_accuracy = float(
            table_row[
                "Phase accuracy"
            ]
        )

        assert_close(
            table_label_accuracy,
            raw_label_accuracy,
            name=(
                f"{method} table label accuracy"
            ),
        )

        assert_close(
            table_order_accuracy,
            raw_order_accuracy,
            name=(
                f"{method} table order accuracy"
            ),
        )

        assert_close(
            table_phase_accuracy,
            raw_phase_accuracy,
            name=(
                f"{method} table phase accuracy"
            ),
        )

        report_lines.append(
            (
                f"PASS {method}: "
                f"label={raw_label_accuracy:.8f}, "
                f"order={raw_order_accuracy:.8f}, "
                f"phase={raw_phase_accuracy:.8f}"
            )
        )

    report_lines.append("")


def validate_pairwise_statistics(
    report_lines: List[str],
) -> None:
    pairwise_rows = load_csv(
        PAIRWISE_PATH
    )

    label_rows = [
        row
        for row in pairwise_rows
        if row["metric"] == "label_correct"
    ]

    if len(label_rows) != 6:
        raise ValueError(
            "Unexpected number of label-accuracy "
            f"pairwise comparisons: {len(label_rows)}"
        )

    seen_pairs = set()

    for row in label_rows:
        pair = (
            row["first_method"],
            row["second_method"],
        )

        if pair in seen_pairs:
            raise ValueError(
                f"Duplicate pairwise comparison: {pair}"
            )

        seen_pairs.add(
            pair
        )

    report_lines.append(
        "[PAIRWISE STATISTICS]"
    )

    report_lines.append(
        "PASS label-accuracy comparisons: 6"
    )

    target_matches = [
        row
        for row in label_rows
        if (
            row["first_method"]
            == "A1_MASK_ULS"
            and row["second_method"]
            == "A3_MC_VWLS"
        )
    ]

    if len(target_matches) != 1:
        raise ValueError(
            "A1_MASK_ULS vs A3_MC_VWLS comparison "
            "was not found exactly once."
        )

    target = target_matches[0]

    adjusted_pvalue = float(
        target[
            "bh_adjusted_exact_pvalue"
        ]
    )

    significant = int(
        target[
            "significant_at_0_05"
        ]
    )

    if adjusted_pvalue < 0.05:
        raise ValueError(
            "A1_MASK_ULS vs A3_MC_VWLS should not "
            "be significant at the 0.05 level."
        )

    if significant != 0:
        raise ValueError(
            "A1_MASK_ULS vs A3_MC_VWLS significant flag "
            "should be zero."
        )

    report_lines.append(
        (
            "PASS A1_MASK_ULS vs A3_MC_VWLS: "
            f"BH-adjusted p={adjusted_pvalue:.10e}, "
            "significant=0"
        )
    )

    report_lines.append("")


def validate_cn2_distance(
    report_lines: List[str],
) -> None:
    rows = load_csv(
        CN2_DISTANCE_PATH
    )

    if len(rows) != 112:
        raise ValueError(
            "Unexpected raw Cn2-distance row count: "
            f"expected=112, actual={len(rows)}"
        )

    methods = sorted(
        {
            row["method"]
            for row in rows
        }
    )

    if len(methods) != 4:
        raise ValueError(
            "Unexpected number of methods in "
            f"Cn2-distance summary: {len(methods)}"
        )

    for method in methods:
        method_rows = [
            row
            for row in rows
            if row["method"] == method
        ]

        if len(method_rows) != 28:
            raise ValueError(
                f"Unexpected Cn2-distance count for {method}: "
                f"expected=28, actual={len(method_rows)}"
            )

        condition_pairs = {
            (
                float(
                    row["cn2"]
                ),
                float(
                    row["distance"]
                ),
            )
            for row in method_rows
        }

        if len(condition_pairs) != 28:
            raise ValueError(
                f"Duplicate or missing Cn2-distance conditions "
                f"for method={method}"
            )

    table_rows = load_csv(
        TABLE_DIRECTORY
        / "Table6_cn2_distance_results.csv"
    )

    if len(table_rows) != 28:
        raise ValueError(
            "Unexpected Table6 row count: "
            f"expected=28, actual={len(table_rows)}"
        )

    report_lines.append(
        "[CN2-DISTANCE RESULTS]"
    )

    report_lines.append(
        "PASS raw summary rows: 112"
    )

    report_lines.append(
        "PASS conditions per method: 28"
    )

    report_lines.append(
        "PASS Table6 rows: 28"
    )

    report_lines.append("")


def main() -> None:
    print("=" * 78)
    print("VALIDATE PAPER OUTPUTS")
    print("=" * 78)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_lines = [
        "Paper output validation report",
        "",
    ]

    try:
        validate_files(
            report_lines
        )

        validate_table_row_counts(
            report_lines
        )

        validate_main_results(
            report_lines
        )

        validate_pairwise_statistics(
            report_lines
        )

        validate_cn2_distance(
            report_lines
        )

        status = "PASS"

    except Exception as error:
        status = "FAIL"

        report_lines.extend(
            [
                "",
                "[ERROR]",
                f"{type(error).__name__}: {error}",
            ]
        )

    report_lines.extend(
        [
            "",
            "=" * 78,
            f"PAPER OUTPUT VALIDATION STATUS: {status}",
            "=" * 78,
        ]
    )

    report_text = "\n".join(
        report_lines
    )

    REPORT_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    print(
        report_text
    )

    print("")
    print(
        "Report:",
        REPORT_PATH,
    )

    if status != "PASS":
        raise RuntimeError(
            "Paper output validation failed. "
            f"See report: {REPORT_PATH}"
        )


if __name__ == "__main__":
    main()