"""
Summarize full-test recognition results by turbulence strength and distance.

Input:
    results/csv/mc_vwls_ablation_full_test_predictions.csv
    data/generated/occlusion_clean_v2.h5

Outputs:
    results/csv/ablation_by_cn2_distance.csv
    results/csv/ablation_by_cn2_distance_occlusion.csv
    results/validation/ablation_by_cn2_distance_report.txt
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

PREDICTION_PATH = (
    ROOT
    / "results"
    / "csv"
    / "mc_vwls_ablation_full_test_predictions.csv"
)

H5_PATH = (
    ROOT
    / "data"
    / "generated"
    / "occlusion_clean_v2.h5"
)

SUMMARY_PATH = (
    ROOT
    / "results"
    / "csv"
    / "ablation_by_cn2_distance.csv"
)

SUMMARY_OCCLUSION_PATH = (
    ROOT
    / "results"
    / "csv"
    / "ablation_by_cn2_distance_occlusion.csv"
)

REPORT_PATH = (
    ROOT
    / "results"
    / "validation"
    / "ablation_by_cn2_distance_report.txt"
)

METHOD_NAMES = (
    "A0_DAF",
    "A1_MASK_ULS",
    "A2_RAW_VWLS",
    "A3_MC_VWLS",
)

EXPECTED_CN2_VALUES = (
    1.0e-15,
    2.5e-15,
    5.0e-15,
    1.0e-14,
    2.5e-14,
    5.0e-14,
    1.0e-13,
)

EXPECTED_DISTANCES = (
    250.0,
    500.0,
    750.0,
    1000.0,
)

EXPECTED_OCCLUSIONS = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
)


def load_conditions() -> np.ndarray:
    if not H5_PATH.exists():
        raise FileNotFoundError(
            f"HDF5 file does not exist: {H5_PATH}"
        )

    with h5py.File(
        H5_PATH,
        "r",
    ) as h5:
        if "conditions" not in h5:
            raise KeyError(
                "conditions dataset was not found."
            )

        conditions = np.asarray(
            h5["conditions"][:],
            dtype=np.float64,
        )

    if conditions.ndim != 2:
        raise ValueError(
            "conditions must be two-dimensional."
        )

    if conditions.shape[1] != 9:
        raise ValueError(
            "conditions must contain exactly nine columns."
        )

    if not np.all(
        np.isfinite(conditions)
    ):
        raise ValueError(
            "conditions contains NaN or Inf."
        )

    return conditions


def load_predictions(
    conditions: np.ndarray,
) -> List[Dict[str, object]]:
    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(
            f"Prediction CSV does not exist: {PREDICTION_PATH}"
        )

    rows: List[
        Dict[str, object]
    ] = []

    with PREDICTION_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file
        )

        required_columns = {
            "method",
            "sample_index",
            "target_snr_db",
            "target_occlusion",
            "true_label",
            "predicted_label",
            "true_order",
            "predicted_order",
            "true_phase_bin",
            "predicted_phase_bin",
            "phase_bin_error",
            "label_correct",
            "order_correct",
            "phase_correct",
            "confidence",
        }

        missing_columns = (
            required_columns
            - set(
                reader.fieldnames or []
            )
        )

        if missing_columns:
            raise KeyError(
                "Prediction CSV is missing columns: "
                f"{sorted(missing_columns)}"
            )

        for raw_row in reader:
            sample_index = int(
                raw_row["sample_index"]
            )

            if not (
                0
                <= sample_index
                < len(conditions)
            ):
                raise IndexError(
                    "Prediction sample index is outside "
                    f"conditions array: {sample_index}"
                )

            condition = conditions[
                sample_index
            ]

            rows.append(
                {
                    "method": raw_row["method"],
                    "sample_index": sample_index,
                    "target_snr_db": float(
                        raw_row[
                            "target_snr_db"
                        ]
                    ),
                    "target_occlusion": float(
                        raw_row[
                            "target_occlusion"
                        ]
                    ),
                    "true_label": int(
                        raw_row[
                            "true_label"
                        ]
                    ),
                    "predicted_label": int(
                        raw_row[
                            "predicted_label"
                        ]
                    ),
                    "true_order": int(
                        raw_row[
                            "true_order"
                        ]
                    ),
                    "predicted_order": int(
                        raw_row[
                            "predicted_order"
                        ]
                    ),
                    "true_phase_bin": int(
                        raw_row[
                            "true_phase_bin"
                        ]
                    ),
                    "predicted_phase_bin": int(
                        raw_row[
                            "predicted_phase_bin"
                        ]
                    ),
                    "phase_bin_error": int(
                        raw_row[
                            "phase_bin_error"
                        ]
                    ),
                    "label_correct": int(
                        raw_row[
                            "label_correct"
                        ]
                    ),
                    "order_correct": int(
                        raw_row[
                            "order_correct"
                        ]
                    ),
                    "phase_correct": int(
                        raw_row[
                            "phase_correct"
                        ]
                    ),
                    "confidence": float(
                        raw_row[
                            "confidence"
                        ]
                    ),
                    "cn2": float(
                        condition[0]
                    ),
                    "distance": float(
                        condition[1]
                    ),
                    "propagation_seed": int(
                        round(
                            condition[2]
                        )
                    ),
                    "achieved_occlusion": float(
                        condition[4]
                    ),
                }
            )

    if not rows:
        raise ValueError(
            "Prediction CSV contains no data rows."
        )

    return rows


def calculate_mean(
    rows: List[Dict[str, object]],
    key: str,
) -> float:
    if not rows:
        return float("nan")

    return float(
        np.mean(
            [
                float(
                    row[key]
                )
                for row in rows
            ],
            dtype=np.float64,
        )
    )


def make_summary_row(
    *,
    method: str,
    cn2: float,
    distance: float,
    rows: List[Dict[str, object]],
    target_occlusion: object = "",
) -> Dict[str, object]:
    return {
        "method": method,
        "cn2": float(
            cn2
        ),
        "distance": float(
            distance
        ),
        "target_occlusion": target_occlusion,
        "sample_count": len(
            rows
        ),
        "label_accuracy": calculate_mean(
            rows,
            "label_correct",
        ),
        "order_accuracy": calculate_mean(
            rows,
            "order_correct",
        ),
        "phase_accuracy": calculate_mean(
            rows,
            "phase_correct",
        ),
        "mean_phase_bin_error": calculate_mean(
            rows,
            "phase_bin_error",
        ),
        "mean_confidence": calculate_mean(
            rows,
            "confidence",
        ),
        "mean_achieved_occlusion": calculate_mean(
            rows,
            "achieved_occlusion",
        ),
    }


def build_summary(
    prediction_rows: List[Dict[str, object]],
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    summary_rows: List[
        Dict[str, object]
    ] = []

    occlusion_rows: List[
        Dict[str, object]
    ] = []

    for method in METHOD_NAMES:
        method_rows = [
            row
            for row in prediction_rows
            if row["method"] == method
        ]

        if not method_rows:
            raise ValueError(
                f"No rows found for method: {method}"
            )

        for cn2 in EXPECTED_CN2_VALUES:
            for distance in EXPECTED_DISTANCES:
                condition_rows = [
                    row
                    for row in method_rows
                    if (
                        np.isclose(
                            float(
                                row["cn2"]
                            ),
                            cn2,
                            rtol=1.0e-5,
                            atol=0.0,
                        )
                        and np.isclose(
                            float(
                                row["distance"]
                            ),
                            distance,
                            rtol=0.0,
                            atol=1.0e-8,
                        )
                    )
                ]

                if not condition_rows:
                    raise ValueError(
                        "Missing condition rows: "
                        f"method={method}, "
                        f"Cn2={cn2:.3e}, "
                        f"distance={distance:.1f}"
                    )

                summary_rows.append(
                    make_summary_row(
                        method=method,
                        cn2=cn2,
                        distance=distance,
                        rows=condition_rows,
                    )
                )

                for occlusion in EXPECTED_OCCLUSIONS:
                    subset = [
                        row
                        for row in condition_rows
                        if np.isclose(
                            float(
                                row[
                                    "target_occlusion"
                                ]
                            ),
                            occlusion,
                            rtol=0.0,
                            atol=1.0e-8,
                        )
                    ]

                    if not subset:
                        raise ValueError(
                            "Missing condition rows: "
                            f"method={method}, "
                            f"Cn2={cn2:.3e}, "
                            f"distance={distance:.1f}, "
                            f"occlusion={occlusion:.1f}"
                        )

                    occlusion_rows.append(
                        make_summary_row(
                            method=method,
                            cn2=cn2,
                            distance=distance,
                            target_occlusion=(
                                occlusion
                            ),
                            rows=subset,
                        )
                    )

    return (
        summary_rows,
        occlusion_rows,
    )


def save_csv(
    path: Path,
    rows: List[Dict[str, object]],
) -> None:
    if not rows:
        raise ValueError(
            f"No rows available for: {path}"
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


def find_summary_row(
    summary_rows: List[Dict[str, object]],
    *,
    method: str,
    cn2: float,
    distance: float,
) -> Dict[str, object]:
    for row in summary_rows:
        if row["method"] != method:
            continue

        if not np.isclose(
            float(
                row["cn2"]
            ),
            cn2,
            rtol=1.0e-5,
            atol=0.0,
        ):
            continue

        if not np.isclose(
            float(
                row["distance"]
            ),
            distance,
            rtol=0.0,
            atol=1.0e-8,
        ):
            continue

        return row

    raise KeyError(
        "Summary row not found: "
        f"method={method}, "
        f"Cn2={cn2:.3e}, "
        f"distance={distance:.1f}"
    )


def build_report(
    summary_rows: List[Dict[str, object]],
) -> str:
    lines = [
        "Ablation results by turbulence strength and distance",
        f"Predictions: {PREDICTION_PATH}",
        f"Dataset: {H5_PATH}",
        "",
    ]

    for method in METHOD_NAMES:
        lines.append(
            f"[{method}]"
        )

        header = (
            "Cn2"
            + "".join(
                f", distance={distance:.0f}"
                for distance in EXPECTED_DISTANCES
            )
        )

        lines.append(
            header
        )

        for cn2 in EXPECTED_CN2_VALUES:
            values = []

            for distance in EXPECTED_DISTANCES:
                row = find_summary_row(
                    summary_rows,
                    method=method,
                    cn2=cn2,
                    distance=distance,
                )

                values.append(
                    float(
                        row[
                            "label_accuracy"
                        ]
                    )
                )

            line = (
                f"{cn2:.3e}"
                + "".join(
                    f", {value:.8f}"
                    for value in values
                )
            )

            lines.append(
                line
            )

        lines.append(
            ""
        )

    lines.extend(
        [
            f"Summary CSV: {SUMMARY_PATH}",
            (
                "Occlusion summary CSV: "
                f"{SUMMARY_OCCLUSION_PATH}"
            ),
        ]
    )

    return "\n".join(
        lines
    )


def main() -> None:
    print("=" * 78)
    print("SUMMARIZE RESULTS BY CN2 AND DISTANCE")
    print("=" * 78)

    conditions = load_conditions()

    print(
        "Conditions shape:",
        conditions.shape,
    )

    prediction_rows = load_predictions(
        conditions
    )

    print(
        "Prediction rows:",
        len(prediction_rows),
    )

    summary_rows, occlusion_rows = (
        build_summary(
            prediction_rows
        )
    )

    print(
        "Cn2-distance summary rows:",
        len(summary_rows),
    )

    print(
        "Cn2-distance-occlusion rows:",
        len(occlusion_rows),
    )

    save_csv(
        SUMMARY_PATH,
        summary_rows,
    )

    save_csv(
        SUMMARY_OCCLUSION_PATH,
        occlusion_rows,
    )

    report_text = build_report(
        summary_rows
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    print("")
    print(
        report_text
    )

    print("")
    print("=" * 78)
    print(
        "CN2-DISTANCE SUMMARY COMPLETE"
    )
    print("=" * 78)

    print(
        "Summary:",
        SUMMARY_PATH,
    )

    print(
        "Occlusion summary:",
        SUMMARY_OCCLUSION_PATH,
    )

    print(
        "Report:",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()