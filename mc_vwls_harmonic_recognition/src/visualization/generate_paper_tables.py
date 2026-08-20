"""
Generate paper-ready CSV tables from completed experiment outputs.

Inputs:
    results/csv/occlusion_level_summary.csv
    results/csv/receiver_noise_qc.csv
    results/csv/mc_vwls_ablation_summary.csv
    results/csv/ablation_pairwise_statistics.csv
    results/csv/ablation_by_cn2_distance.csv
    results/csv/ablation_accuracy_confidence_intervals.csv

Outputs:
    results/tables/Table1_dataset_statistics.csv
    results/tables/Table2_occlusion_validation.csv
    results/tables/Table3_noise_validation.csv
    results/tables/Table4_main_results.csv
    results/tables/Table5_ablation_results.csv
    results/tables/Table6_cn2_distance_results.csv
    results/tables/Table7_statistical_significance.csv
    results/validation/paper_tables_report.txt
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

CSV_DIRECTORY = (
    ROOT
    / "results"
    / "csv"
)

TABLE_DIRECTORY = (
    ROOT
    / "results"
    / "tables"
)

REPORT_PATH = (
    ROOT
    / "results"
    / "validation"
    / "paper_tables_report.txt"
)

OCCLUSION_PATH = (
    CSV_DIRECTORY
    / "occlusion_level_summary.csv"
)

NOISE_PATH = (
    CSV_DIRECTORY
    / "receiver_noise_qc.csv"
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

CONFIDENCE_INTERVAL_PATH = (
    CSV_DIRECTORY
    / "ablation_accuracy_confidence_intervals.csv"
)

TABLE1_PATH = (
    TABLE_DIRECTORY
    / "Table1_dataset_statistics.csv"
)

TABLE2_PATH = (
    TABLE_DIRECTORY
    / "Table2_occlusion_validation.csv"
)

TABLE3_PATH = (
    TABLE_DIRECTORY
    / "Table3_noise_validation.csv"
)

TABLE4_PATH = (
    TABLE_DIRECTORY
    / "Table4_main_results.csv"
)

TABLE5_PATH = (
    TABLE_DIRECTORY
    / "Table5_ablation_results.csv"
)

TABLE6_PATH = (
    TABLE_DIRECTORY
    / "Table6_cn2_distance_results.csv"
)

TABLE7_PATH = (
    TABLE_DIRECTORY
    / "Table7_statistical_significance.csv"
)

METHOD_ORDER = (
    "A0_DAF",
    "A1_MASK_ULS",
    "A2_RAW_VWLS",
    "A3_MC_VWLS",
)

METHOD_LABELS = {
    "A0_DAF": "DAF",
    "A1_MASK_ULS": "Mask-ULS",
    "A2_RAW_VWLS": "Raw-VWLS",
    "A3_MC_VWLS": "MC-VWLS",
}

METHOD_DESCRIPTIONS = {
    "A0_DAF": (
        "Raw zero-filled angular profile; "
        "no mask normalization; no visibility weighting"
    ),
    "A1_MASK_ULS": (
        "Mask-normalized angular profile; "
        "unweighted least squares"
    ),
    "A2_RAW_VWLS": (
        "Raw zero-filled angular profile; "
        "visibility-weighted least squares"
    ),
    "A3_MC_VWLS": (
        "Mask-normalized angular profile; "
        "visibility-weighted least squares"
    ),
}

CN2_VALUES = (
    1.0e-15,
    2.5e-15,
    5.0e-15,
    1.0e-14,
    2.5e-14,
    5.0e-14,
    1.0e-13,
)

DISTANCES = (
    250.0,
    500.0,
    750.0,
    1000.0,
)


def load_csv(
    path: Path,
) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required CSV does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file
        )

        rows = list(
            reader
        )

    if not rows:
        raise ValueError(
            f"CSV contains no data rows: {path}"
        )

    return rows


def require_columns(
    rows: Sequence[Dict[str, str]],
    required_columns: Iterable[str],
    *,
    source_name: str,
) -> None:
    if not rows:
        raise ValueError(
            f"No rows found in {source_name}."
        )

    available_columns = set(
        rows[0].keys()
    )

    missing_columns = (
        set(required_columns)
        - available_columns
    )

    if missing_columns:
        raise KeyError(
            f"{source_name} is missing columns: "
            f"{sorted(missing_columns)}"
        )


def save_csv(
    path: Path,
    rows: Sequence[Dict[str, object]],
) -> None:
    if not rows:
        raise ValueError(
            f"No rows available for {path}."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def as_float(
    row: Dict[str, str],
    key: str,
) -> float:
    value = row[key]

    if value is None:
        return float("nan")

    stripped = str(value).strip()

    if stripped == "":
        return float("nan")

    return float(
        stripped
    )


def as_int(
    row: Dict[str, str],
    key: str,
) -> int:
    return int(
        round(
            as_float(
                row,
                key,
            )
        )
    )


def format_scientific(
    value: float,
) -> str:
    return f"{float(value):.3e}"


def format_decimal(
    value: float,
    digits: int = 6,
) -> str:
    if not np.isfinite(
        value
    ):
        return ""

    return f"{float(value):.{digits}f}"


def find_ablation_row(
    rows: Sequence[Dict[str, str]],
    *,
    method: str,
    scope: str,
    target_snr_db: float | None = None,
    target_occlusion: float | None = None,
    true_order: int | None = None,
) -> Dict[str, str]:
    matches = []

    for row in rows:
        if row["method"] != method:
            continue

        if row["scope"] != scope:
            continue

        if target_snr_db is not None:
            value = as_float(
                row,
                "target_snr_db",
            )

            if not np.isclose(
                value,
                target_snr_db,
                rtol=0.0,
                atol=1.0e-8,
            ):
                continue

        if target_occlusion is not None:
            value = as_float(
                row,
                "target_occlusion",
            )

            if not np.isclose(
                value,
                target_occlusion,
                rtol=0.0,
                atol=1.0e-8,
            ):
                continue

        if true_order is not None:
            value = as_float(
                row,
                "true_order",
            )

            if not np.isclose(
                value,
                float(true_order),
                rtol=0.0,
                atol=1.0e-8,
            ):
                continue

        matches.append(
            row
        )

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one ablation summary row, "
            f"found {len(matches)}: "
            f"method={method}, scope={scope}, "
            f"SNR={target_snr_db}, "
            f"occlusion={target_occlusion}, "
            f"order={true_order}"
        )

    return matches[0]


def find_ci_row(
    rows: Sequence[Dict[str, str]],
    *,
    method: str,
    metric: str,
) -> Dict[str, str]:
    matches = [
        row
        for row in rows
        if (
            row["method"] == method
            and row["metric"] == metric
        )
    ]

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one confidence-interval row, "
            f"found {len(matches)}: "
            f"method={method}, metric={metric}"
        )

    return matches[0]


def build_table1_dataset_statistics() -> List[Dict[str, object]]:
    return [
        {
            "Item": "State classes",
            "Value": 32,
            "Description": (
                "Four OAM orders multiplied by eight phase bins"
            ),
        },
        {
            "Item": "OAM orders",
            "Value": "1, 2, 3, 4",
            "Description": (
                "Candidate superposition-state OAM orders"
            ),
        },
        {
            "Item": "Phase bins",
            "Value": 8,
            "Description": (
                "Uniform discrete relative-phase categories"
            ),
        },
        {
            "Item": "Turbulence strengths Cn2",
            "Value": 7,
            "Description": (
                "1e-15, 2.5e-15, 5e-15, 1e-14, "
                "2.5e-14, 5e-14, 1e-13"
            ),
        },
        {
            "Item": "Propagation distances",
            "Value": 4,
            "Description": "250, 500, 750, 1000",
        },
        {
            "Item": "Propagation seeds per condition",
            "Value": 10,
            "Description": "Seeds 0 through 9",
        },
        {
            "Item": "Occlusion levels",
            "Value": 5,
            "Description": "0.0, 0.1, 0.2, 0.3, 0.4",
        },
        {
            "Item": "Clean samples",
            "Value": 44800,
            "Description": (
                "Complete corrected local-occlusion dataset"
            ),
        },
        {
            "Item": "Training samples",
            "Value": 31360,
            "Description": (
                "Grouped split with no propagation-group leakage"
            ),
        },
        {
            "Item": "Validation samples",
            "Value": 6720,
            "Description": (
                "Grouped split with no propagation-group leakage"
            ),
        },
        {
            "Item": "Test samples",
            "Value": 6720,
            "Description": (
                "Grouped split with no propagation-group leakage"
            ),
        },
        {
            "Item": "Nominal SNR levels",
            "Value": 5,
            "Description": "20, 15, 10, 5, 0 dB",
        },
        {
            "Item": "Noisy test observations per method",
            "Value": 33600,
            "Description": "6720 test samples multiplied by five SNR levels",
        },
        {
            "Item": "Angular samples",
            "Value": 180,
            "Description": "Uniform polar angular samples",
        },
        {
            "Item": "Radial samples",
            "Value": 64,
            "Description": "Polar radial samples",
        },
    ]


def build_table2_occlusion_validation(
    rows: Sequence[Dict[str, str]],
) -> List[Dict[str, object]]:
    require_columns(
        rows,
        (
            "target_occlusion_ratio",
            "sample_count",
            "achieved_mean",
            "achieved_std",
            "achieved_min",
            "achieved_max",
            "absolute_error_mean",
            "absolute_error_max",
            "radius_mean_pixels",
            "radius_std_pixels",
            "visible_area_fraction_mean",
            "visible_area_fraction_std",
        ),
        source_name=OCCLUSION_PATH.name,
    )

    output_rows = []

    sorted_rows = sorted(
        rows,
        key=lambda row: as_float(
            row,
            "target_occlusion_ratio",
        ),
    )

    for row in sorted_rows:
        output_rows.append(
            {
                "Target energy occlusion": format_decimal(
                    as_float(
                        row,
                        "target_occlusion_ratio",
                    ),
                    1,
                ),
                "Samples": as_int(
                    row,
                    "sample_count",
                ),
                "Achieved mean": format_decimal(
                    as_float(
                        row,
                        "achieved_mean",
                    ),
                    6,
                ),
                "Achieved standard deviation": format_decimal(
                    as_float(
                        row,
                        "achieved_std",
                    ),
                    6,
                ),
                "Achieved minimum": format_decimal(
                    as_float(
                        row,
                        "achieved_min",
                    ),
                    6,
                ),
                "Achieved maximum": format_decimal(
                    as_float(
                        row,
                        "achieved_max",
                    ),
                    6,
                ),
                "Mean absolute error": format_decimal(
                    as_float(
                        row,
                        "absolute_error_mean",
                    ),
                    6,
                ),
                "Maximum absolute error": format_decimal(
                    as_float(
                        row,
                        "absolute_error_max",
                    ),
                    6,
                ),
                "Mean radius (pixels)": format_decimal(
                    as_float(
                        row,
                        "radius_mean_pixels",
                    ),
                    3,
                ),
                "Radius standard deviation (pixels)": format_decimal(
                    as_float(
                        row,
                        "radius_std_pixels",
                    ),
                    3,
                ),
                "Mean visible area fraction": format_decimal(
                    as_float(
                        row,
                        "visible_area_fraction_mean",
                    ),
                    6,
                ),
                "Visible area fraction standard deviation": format_decimal(
                    as_float(
                        row,
                        "visible_area_fraction_std",
                    ),
                    6,
                ),
            }
        )

    return output_rows


def build_table3_noise_validation(
    rows: Sequence[Dict[str, str]],
) -> List[Dict[str, object]]:
    require_columns(
        rows,
        (
            "target_snr_db",
            "sample_count",
            "preclip_snr_mean_db",
            "preclip_snr_std_db",
            "postclip_snr_mean_db",
            "postclip_snr_std_db",
            "postclip_snr_shift_mean_db",
            "clipped_fraction_mean",
            "clipped_fraction_std",
        ),
        source_name=NOISE_PATH.name,
    )

    output_rows = []

    sorted_rows = sorted(
        rows,
        key=lambda row: as_float(
            row,
            "target_snr_db",
        ),
        reverse=True,
    )

    for row in sorted_rows:
        output_rows.append(
            {
                "Target SNR (dB)": format_decimal(
                    as_float(
                        row,
                        "target_snr_db",
                    ),
                    1,
                ),
                "Samples": as_int(
                    row,
                    "sample_count",
                ),
                "Pre-clipping SNR mean (dB)": format_decimal(
                    as_float(
                        row,
                        "preclip_snr_mean_db",
                    ),
                    6,
                ),
                "Pre-clipping SNR standard deviation (dB)": (
                    format_scientific(
                        as_float(
                            row,
                            "preclip_snr_std_db",
                        )
                    )
                ),
                "Post-clipping SNR mean (dB)": format_decimal(
                    as_float(
                        row,
                        "postclip_snr_mean_db",
                    ),
                    6,
                ),
                "Post-clipping SNR standard deviation (dB)": (
                    format_decimal(
                        as_float(
                            row,
                            "postclip_snr_std_db",
                        ),
                        6,
                    )
                ),
                "Mean SNR shift (dB)": format_decimal(
                    as_float(
                        row,
                        "postclip_snr_shift_mean_db",
                    ),
                    6,
                ),
                "Mean clipped-pixel fraction": format_decimal(
                    as_float(
                        row,
                        "clipped_fraction_mean",
                    ),
                    6,
                ),
                "Clipped-pixel fraction standard deviation": (
                    format_decimal(
                        as_float(
                            row,
                            "clipped_fraction_std",
                        ),
                        6,
                    )
                ),
            }
        )

    return output_rows


def build_table4_main_results(
    ablation_rows: Sequence[Dict[str, str]],
    ci_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, object]]:
    output_rows = []

    for method in METHOD_ORDER:
        summary = find_ablation_row(
            ablation_rows,
            method=method,
            scope="overall",
        )

        label_ci = find_ci_row(
            ci_rows,
            method=method,
            metric="label_correct",
        )

        output_rows.append(
            {
                "Method": METHOD_LABELS[
                    method
                ],
                "Samples": as_int(
                    summary,
                    "sample_count",
                ),
                "Label accuracy": format_decimal(
                    as_float(
                        summary,
                        "label_accuracy",
                    ),
                    8,
                ),
                "Label accuracy 95% CI lower": format_decimal(
                    as_float(
                        label_ci,
                        "wilson_95_ci_lower",
                    ),
                    8,
                ),
                "Label accuracy 95% CI upper": format_decimal(
                    as_float(
                        label_ci,
                        "wilson_95_ci_upper",
                    ),
                    8,
                ),
                "Order accuracy": format_decimal(
                    as_float(
                        summary,
                        "order_accuracy",
                    ),
                    8,
                ),
                "Phase accuracy": format_decimal(
                    as_float(
                        summary,
                        "phase_accuracy",
                    ),
                    8,
                ),
                "Mean circular phase-bin error": format_decimal(
                    as_float(
                        summary,
                        "mean_phase_bin_error",
                    ),
                    8,
                ),
                "Mean confidence": format_decimal(
                    as_float(
                        summary,
                        "mean_confidence",
                    ),
                    8,
                ),
            }
        )

    return output_rows


def build_table5_ablation_results(
    rows: Sequence[Dict[str, str]],
) -> List[Dict[str, object]]:
    output_rows = []

    for method in METHOD_ORDER:
        overall = find_ablation_row(
            rows,
            method=method,
            scope="overall",
        )

        occ_00 = find_ablation_row(
            rows,
            method=method,
            scope="occlusion",
            target_occlusion=0.0,
        )

        occ_02 = find_ablation_row(
            rows,
            method=method,
            scope="occlusion",
            target_occlusion=0.2,
        )

        occ_04 = find_ablation_row(
            rows,
            method=method,
            scope="occlusion",
            target_occlusion=0.4,
        )

        output_rows.append(
            {
                "Method": METHOD_LABELS[
                    method
                ],
                "Configuration": METHOD_DESCRIPTIONS[
                    method
                ],
                "Overall label accuracy": format_decimal(
                    as_float(
                        overall,
                        "label_accuracy",
                    ),
                    8,
                ),
                "Occlusion 0.0 accuracy": format_decimal(
                    as_float(
                        occ_00,
                        "label_accuracy",
                    ),
                    8,
                ),
                "Occlusion 0.2 accuracy": format_decimal(
                    as_float(
                        occ_02,
                        "label_accuracy",
                    ),
                    8,
                ),
                "Occlusion 0.4 accuracy": format_decimal(
                    as_float(
                        occ_04,
                        "label_accuracy",
                    ),
                    8,
                ),
                "Accuracy loss from 0.0 to 0.4": format_decimal(
                    (
                        as_float(
                            occ_00,
                            "label_accuracy",
                        )
                        - as_float(
                            occ_04,
                            "label_accuracy",
                        )
                    ),
                    8,
                ),
                "Overall order accuracy": format_decimal(
                    as_float(
                        overall,
                        "order_accuracy",
                    ),
                    8,
                ),
                "Overall phase accuracy": format_decimal(
                    as_float(
                        overall,
                        "phase_accuracy",
                    ),
                    8,
                ),
            }
        )

    return output_rows


def build_table6_cn2_distance_results(
    rows: Sequence[Dict[str, str]],
) -> List[Dict[str, object]]:
    require_columns(
        rows,
        (
            "method",
            "cn2",
            "distance",
            "sample_count",
            "label_accuracy",
            "order_accuracy",
            "phase_accuracy",
        ),
        source_name=CN2_DISTANCE_PATH.name,
    )

    output_rows = []

    for method in METHOD_ORDER:
        for cn2 in CN2_VALUES:
            output_row: Dict[str, object] = {
                "Method": METHOD_LABELS[
                    method
                ],
                "Cn2": format_scientific(
                    cn2
                ),
            }

            for distance in DISTANCES:
                matches = [
                    row
                    for row in rows
                    if (
                        row["method"]
                        == method
                        and np.isclose(
                            as_float(
                                row,
                                "cn2",
                            ),
                            cn2,
                            rtol=1.0e-5,
                            atol=0.0,
                        )
                        and np.isclose(
                            as_float(
                                row,
                                "distance",
                            ),
                            distance,
                            rtol=0.0,
                            atol=1.0e-8,
                        )
                    )
                ]

                if len(matches) != 1:
                    raise ValueError(
                        "Expected exactly one Cn2-distance row, "
                        f"found {len(matches)}: "
                        f"method={method}, "
                        f"Cn2={cn2:.3e}, "
                        f"distance={distance:.0f}"
                    )

                output_row[
                    f"Accuracy at distance {int(distance)}"
                ] = format_decimal(
                    as_float(
                        matches[0],
                        "label_accuracy",
                    ),
                    8,
                )

            output_rows.append(
                output_row
            )

    return output_rows


def build_table7_statistical_significance(
    rows: Sequence[Dict[str, str]],
) -> List[Dict[str, object]]:
    require_columns(
        rows,
        (
            "metric",
            "first_method",
            "second_method",
            "sample_count",
            "first_accuracy",
            "second_accuracy",
            "accuracy_difference_second_minus_first",
            "bootstrap_ci_lower",
            "bootstrap_ci_upper",
            "first_only_correct",
            "second_only_correct",
            "exact_pvalue",
            "bh_adjusted_exact_pvalue",
            "significant_at_0_05",
        ),
        source_name=PAIRWISE_PATH.name,
    )

    label_rows = [
        row
        for row in rows
        if row["metric"] == "label_correct"
    ]

    method_rank = {
        method: index
        for index, method in enumerate(
            METHOD_ORDER
        )
    }

    label_rows.sort(
        key=lambda row: (
            method_rank[
                row["first_method"]
            ],
            method_rank[
                row["second_method"]
            ],
        )
    )

    output_rows = []

    for row in label_rows:
        first_method = row[
            "first_method"
        ]

        second_method = row[
            "second_method"
        ]

        output_rows.append(
            {
                "Comparison": (
                    f"{METHOD_LABELS[first_method]} vs "
                    f"{METHOD_LABELS[second_method]}"
                ),
                "Paired observations": as_int(
                    row,
                    "sample_count",
                ),
                "First accuracy": format_decimal(
                    as_float(
                        row,
                        "first_accuracy",
                    ),
                    8,
                ),
                "Second accuracy": format_decimal(
                    as_float(
                        row,
                        "second_accuracy",
                    ),
                    8,
                ),
                "Difference second minus first": format_decimal(
                    as_float(
                        row,
                        "accuracy_difference_second_minus_first",
                    ),
                    8,
                ),
                "Bootstrap 95% CI lower": format_decimal(
                    as_float(
                        row,
                        "bootstrap_ci_lower",
                    ),
                    8,
                ),
                "Bootstrap 95% CI upper": format_decimal(
                    as_float(
                        row,
                        "bootstrap_ci_upper",
                    ),
                    8,
                ),
                "First-only correct": as_int(
                    row,
                    "first_only_correct",
                ),
                "Second-only correct": as_int(
                    row,
                    "second_only_correct",
                ),
                "Exact McNemar p-value": format_scientific(
                    as_float(
                        row,
                        "exact_pvalue",
                    )
                ),
                "BH-adjusted p-value": format_scientific(
                    as_float(
                        row,
                        "bh_adjusted_exact_pvalue",
                    )
                ),
                "Significant at 0.05": (
                    "Yes"
                    if as_int(
                        row,
                        "significant_at_0_05",
                    ) == 1
                    else "No"
                ),
            }
        )

    return output_rows


def main() -> None:
    print("=" * 78)
    print("GENERATE PAPER TABLES")
    print("=" * 78)

    TABLE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    occlusion_rows = load_csv(
        OCCLUSION_PATH
    )

    noise_rows = load_csv(
        NOISE_PATH
    )

    ablation_rows = load_csv(
        ABLATION_SUMMARY_PATH
    )

    pairwise_rows = load_csv(
        PAIRWISE_PATH
    )

    cn2_distance_rows = load_csv(
        CN2_DISTANCE_PATH
    )

    ci_rows = load_csv(
        CONFIDENCE_INTERVAL_PATH
    )

    table1 = (
        build_table1_dataset_statistics()
    )

    table2 = (
        build_table2_occlusion_validation(
            occlusion_rows
        )
    )

    table3 = (
        build_table3_noise_validation(
            noise_rows
        )
    )

    table4 = (
        build_table4_main_results(
            ablation_rows,
            ci_rows,
        )
    )

    table5 = (
        build_table5_ablation_results(
            ablation_rows
        )
    )

    table6 = (
        build_table6_cn2_distance_results(
            cn2_distance_rows
        )
    )

    table7 = (
        build_table7_statistical_significance(
            pairwise_rows
        )
    )

    table_specs = (
        (
            TABLE1_PATH,
            table1,
            "Dataset statistics",
        ),
        (
            TABLE2_PATH,
            table2,
            "Occlusion validation",
        ),
        (
            TABLE3_PATH,
            table3,
            "Receiver-noise validation",
        ),
        (
            TABLE4_PATH,
            table4,
            "Main recognition results",
        ),
        (
            TABLE5_PATH,
            table5,
            "Ablation results",
        ),
        (
            TABLE6_PATH,
            table6,
            "Cn2-distance results",
        ),
        (
            TABLE7_PATH,
            table7,
            "Paired statistical significance",
        ),
    )

    report_lines = [
        "Paper table generation report",
        "",
    ]

    for path, rows, description in table_specs:
        save_csv(
            path,
            rows,
        )

        print(
            "Saved:",
            path,
        )

        report_lines.append(
            (
                f"{path.name}: "
                f"rows={len(rows)}, "
                f"description={description}"
            )
        )

    report_lines.extend(
        [
            "",
            "[INPUT FILES]",
            str(OCCLUSION_PATH),
            str(NOISE_PATH),
            str(ABLATION_SUMMARY_PATH),
            str(PAIRWISE_PATH),
            str(CN2_DISTANCE_PATH),
            str(CONFIDENCE_INTERVAL_PATH),
            "",
            "[OUTPUT DIRECTORY]",
            str(TABLE_DIRECTORY),
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    print("")
    print("=" * 78)
    print("PAPER TABLE GENERATION COMPLETE")
    print("=" * 78)

    print(
        "Report:",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()