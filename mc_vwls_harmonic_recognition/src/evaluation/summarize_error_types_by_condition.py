"""
Summarize MC-VWLS error types by turbulence strength, propagation distance,
occlusion ratio, and receiver SNR.

Inputs:
    results/csv/mc_vwls_ablation_full_test_predictions.csv
    data/generated/occlusion_clean_v2.h5

Outputs:
    results/csv/mc_vwls_error_types_by_cn2.csv
    results/csv/mc_vwls_error_types_by_distance.csv
    results/csv/mc_vwls_error_types_by_occlusion.csv
    results/csv/mc_vwls_error_types_by_snr.csv
    results/csv/mc_vwls_error_types_by_cn2_distance.csv
    results/validation/mc_vwls_error_types_by_condition_report.txt
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

PREDICTION_PATH = (
    ROOT
    / "results"
    / "csv"
    / "mc_vwls_ablation_full_test_predictions.csv"
)

DATASET_PATH = (
    ROOT
    / "data"
    / "generated"
    / "occlusion_clean_v2.h5"
)

OUTPUT_DIRECTORY = (
    ROOT
    / "results"
    / "csv"
)

REPORT_PATH = (
    ROOT
    / "results"
    / "validation"
    / "mc_vwls_error_types_by_condition_report.txt"
)

CN2_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "mc_vwls_error_types_by_cn2.csv"
)

DISTANCE_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "mc_vwls_error_types_by_distance.csv"
)

OCCLUSION_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "mc_vwls_error_types_by_occlusion.csv"
)

SNR_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "mc_vwls_error_types_by_snr.csv"
)

CN2_DISTANCE_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "mc_vwls_error_types_by_cn2_distance.csv"
)

TARGET_METHOD = "A3_MC_VWLS"

CONDITION_CN2_COLUMN = 0
CONDITION_DISTANCE_COLUMN = 1
CONDITION_TARGET_OCCLUSION_COLUMN = 3

PHASE_BIN_COUNT = 8

REQUIRED_PREDICTION_COLUMNS = (
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
    "label_correct",
    "order_correct",
    "phase_correct",
)


def load_prediction_rows(
    path: Path,
) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction CSV does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file
        )

        fieldnames = list(
            reader.fieldnames
            or []
        )

        missing_columns = (
            set(REQUIRED_PREDICTION_COLUMNS)
            - set(fieldnames)
        )

        if missing_columns:
            raise KeyError(
                "Prediction CSV is missing columns: "
                f"{sorted(missing_columns)}"
            )

        rows = [
            row
            for row in reader
            if row["method"] == TARGET_METHOD
        ]

    if not rows:
        raise ValueError(
            f"No prediction rows found for method: {TARGET_METHOD}"
        )

    return rows


def find_conditions_dataset(
    h5_file: h5py.File,
) -> h5py.Dataset:
    if "conditions" in h5_file:
        dataset = h5_file["conditions"]

        if isinstance(
            dataset,
            h5py.Dataset,
        ):
            return dataset

    discovered: List[
        h5py.Dataset
    ] = []

    def visitor(
        name: str,
        item: h5py.Dataset | h5py.Group,
    ) -> None:
        if (
            isinstance(
                item,
                h5py.Dataset,
            )
            and name.split("/")[-1]
            == "conditions"
        ):
            discovered.append(
                item
            )

    h5_file.visititems(
        visitor
    )

    if len(discovered) != 1:
        raise KeyError(
            "Expected exactly one HDF5 dataset named "
            f"'conditions', found {len(discovered)}."
        )

    return discovered[0]


def load_conditions(
    path: Path,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"HDF5 dataset does not exist: {path}"
        )

    with h5py.File(
        path,
        "r",
    ) as h5_file:
        conditions_dataset = find_conditions_dataset(
            h5_file
        )

        conditions = np.asarray(
            conditions_dataset[:],
            dtype=np.float64,
        )

    if conditions.ndim != 2:
        raise ValueError(
            "Conditions must be a two-dimensional array: "
            f"shape={conditions.shape}"
        )

    if conditions.shape[1] < 4:
        raise ValueError(
            "Conditions array must contain at least four columns: "
            f"shape={conditions.shape}"
        )

    return conditions


def parse_int(
    row: Dict[str, str],
    column: str,
) -> int:
    try:
        return int(
            round(
                float(
                    row[column]
                )
            )
        )

    except Exception as error:
        raise ValueError(
            f"Cannot parse integer column {column}: "
            f"{row[column]!r}"
        ) from error


def parse_float(
    row: Dict[str, str],
    column: str,
) -> float:
    try:
        return float(
            row[column]
        )

    except Exception as error:
        raise ValueError(
            f"Cannot parse float column {column}: "
            f"{row[column]!r}"
        ) from error


def circular_phase_distance(
    true_phase: int,
    predicted_phase: int,
) -> int:
    direct_distance = abs(
        true_phase
        - predicted_phase
    )

    return min(
        direct_distance,
        PHASE_BIN_COUNT
        - direct_distance,
    )


def make_empty_counter() -> Dict[str, int]:
    return {
        "sample_count": 0,
        "correct_count": 0,
        "error_count": 0,
        "same_order_error_count": 0,
        "cross_order_error_count": 0,
        "adjacent_phase_error_count": 0,
        "same_order_adjacent_phase_error_count": 0,
        "phase_only_error_count": 0,
        "order_only_error_count": 0,
        "order_and_phase_error_count": 0,
    }


def update_counter(
    counter: Dict[str, int],
    *,
    true_order: int,
    predicted_order: int,
    true_phase: int,
    predicted_phase: int,
) -> None:
    counter["sample_count"] += 1

    order_correct = (
        true_order
        == predicted_order
    )

    phase_correct = (
        true_phase
        == predicted_phase
    )

    label_correct = (
        order_correct
        and phase_correct
    )

    if label_correct:
        counter["correct_count"] += 1
        return

    counter["error_count"] += 1

    phase_distance = circular_phase_distance(
        true_phase,
        predicted_phase,
    )

    if order_correct:
        counter[
            "same_order_error_count"
        ] += 1
    else:
        counter[
            "cross_order_error_count"
        ] += 1

    if phase_distance == 1:
        counter[
            "adjacent_phase_error_count"
        ] += 1

        if order_correct:
            counter[
                "same_order_adjacent_phase_error_count"
            ] += 1

    if (
        order_correct
        and not phase_correct
    ):
        counter[
            "phase_only_error_count"
        ] += 1

    elif (
        not order_correct
        and phase_correct
    ):
        counter[
            "order_only_error_count"
        ] += 1

    else:
        counter[
            "order_and_phase_error_count"
        ] += 1


def safe_ratio(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return float("nan")

    return (
        numerator
        / denominator
    )


def finalize_counter(
    counter: Dict[str, int],
) -> Dict[str, object]:
    sample_count = counter[
        "sample_count"
    ]

    error_count = counter[
        "error_count"
    ]

    result: Dict[
        str,
        object
    ] = dict(
        counter
    )

    result.update(
        {
            "label_accuracy": safe_ratio(
                counter["correct_count"],
                sample_count,
            ),
            "error_rate": safe_ratio(
                error_count,
                sample_count,
            ),
            "same_order_error_fraction_of_errors": safe_ratio(
                counter[
                    "same_order_error_count"
                ],
                error_count,
            ),
            "cross_order_error_fraction_of_errors": safe_ratio(
                counter[
                    "cross_order_error_count"
                ],
                error_count,
            ),
            "adjacent_phase_error_fraction_of_errors": safe_ratio(
                counter[
                    "adjacent_phase_error_count"
                ],
                error_count,
            ),
            "same_order_adjacent_phase_error_fraction_of_errors": (
                safe_ratio(
                    counter[
                        "same_order_adjacent_phase_error_count"
                    ],
                    error_count,
                )
            ),
            "phase_only_error_fraction_of_errors": safe_ratio(
                counter[
                    "phase_only_error_count"
                ],
                error_count,
            ),
            "order_only_error_fraction_of_errors": safe_ratio(
                counter[
                    "order_only_error_count"
                ],
                error_count,
            ),
            "order_and_phase_error_fraction_of_errors": safe_ratio(
                counter[
                    "order_and_phase_error_count"
                ],
                error_count,
            ),
        }
    )

    return result


def save_csv(
    path: Path,
    rows: Sequence[Dict[str, object]],
) -> None:
    if not rows:
        raise ValueError(
            f"No output rows available for: {path}"
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


def sort_key(
    key: Tuple[float, ...],
) -> Tuple[float, ...]:
    return tuple(
        float(value)
        for value in key
    )


def build_group_rows(
    grouped_counters: Dict[
        Tuple[float, ...],
        Dict[str, int],
    ],
    key_names: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[
        Dict[str, object]
    ] = []

    for key in sorted(
        grouped_counters,
        key=sort_key,
    ):
        key_values = (
            key
            if isinstance(
                key,
                tuple,
            )
            else (
                key,
            )
        )

        row: Dict[
            str,
            object
        ] = {}

        for name, value in zip(
            key_names,
            key_values,
        ):
            row[name] = value

        row.update(
            finalize_counter(
                grouped_counters[
                    key
                ]
            )
        )

        rows.append(
            row
        )

    return rows


def format_report_group(
    title: str,
    rows: Iterable[Dict[str, object]],
    key_columns: Sequence[str],
) -> List[str]:
    lines = [
        title
    ]

    for row in rows:
        key_text = ", ".join(
            f"{column}={row[column]}"
            for column in key_columns
        )

        lines.append(
            (
                f"{key_text}, "
                f"samples={row['sample_count']}, "
                f"accuracy={float(row['label_accuracy']):.8f}, "
                f"errors={row['error_count']}, "
                f"cross_order_fraction="
                f"{float(row['cross_order_error_fraction_of_errors']):.8f}, "
                f"same_order_fraction="
                f"{float(row['same_order_error_fraction_of_errors']):.8f}, "
                f"adjacent_phase_fraction="
                f"{float(row['adjacent_phase_error_fraction_of_errors']):.8f}"
            )
        )

    lines.append("")

    return lines


def main() -> None:
    print("=" * 78)
    print("SUMMARIZE MC-VWLS ERROR TYPES BY CONDITION")
    print("=" * 78)

    prediction_rows = load_prediction_rows(
        PREDICTION_PATH
    )

    conditions = load_conditions(
        DATASET_PATH
    )

    print(
        "Selected prediction rows:",
        len(prediction_rows),
    )

    print(
        "Conditions shape:",
        conditions.shape,
    )

    by_cn2: Dict[
        Tuple[float],
        Dict[str, int],
    ] = defaultdict(
        make_empty_counter
    )

    by_distance: Dict[
        Tuple[float],
        Dict[str, int],
    ] = defaultdict(
        make_empty_counter
    )

    by_occlusion: Dict[
        Tuple[float],
        Dict[str, int],
    ] = defaultdict(
        make_empty_counter
    )

    by_snr: Dict[
        Tuple[float],
        Dict[str, int],
    ] = defaultdict(
        make_empty_counter
    )

    by_cn2_distance: Dict[
        Tuple[float, float],
        Dict[str, int],
    ] = defaultdict(
        make_empty_counter
    )

    global_counter = make_empty_counter()

    for row_index, row in enumerate(
        prediction_rows
    ):
        sample_index = parse_int(
            row,
            "sample_index",
        )

        if not (
            0
            <= sample_index
            < conditions.shape[0]
        ):
            raise IndexError(
                "Sample index outside HDF5 conditions: "
                f"row={row_index}, "
                f"sample_index={sample_index}, "
                f"condition_count={conditions.shape[0]}"
            )

        condition = conditions[
            sample_index
        ]

        cn2 = float(
            condition[
                CONDITION_CN2_COLUMN
            ]
        )

        distance = float(
            condition[
                CONDITION_DISTANCE_COLUMN
            ]
        )

        hdf5_occlusion = float(
            condition[
                CONDITION_TARGET_OCCLUSION_COLUMN
            ]
        )

        csv_occlusion = parse_float(
            row,
            "target_occlusion",
        )

        if not np.isclose(
            hdf5_occlusion,
            csv_occlusion,
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise ValueError(
                "Occlusion mismatch between prediction CSV "
                "and HDF5 conditions: "
                f"sample_index={sample_index}, "
                f"CSV={csv_occlusion}, "
                f"HDF5={hdf5_occlusion}"
            )

        snr_db = parse_float(
            row,
            "target_snr_db",
        )

        true_order = parse_int(
            row,
            "true_order",
        )

        predicted_order = parse_int(
            row,
            "predicted_order",
        )

        true_phase = parse_int(
            row,
            "true_phase_bin",
        )

        predicted_phase = parse_int(
            row,
            "predicted_phase_bin",
        )

        counters = (
            global_counter,
            by_cn2[
                (
                    cn2,
                )
            ],
            by_distance[
                (
                    distance,
                )
            ],
            by_occlusion[
                (
                    csv_occlusion,
                )
            ],
            by_snr[
                (
                    snr_db,
                )
            ],
            by_cn2_distance[
                (
                    cn2,
                    distance,
                )
            ],
        )

        for counter in counters:
            update_counter(
                counter,
                true_order=true_order,
                predicted_order=predicted_order,
                true_phase=true_phase,
                predicted_phase=predicted_phase,
            )

    cn2_rows = build_group_rows(
        by_cn2,
        (
            "cn2",
        ),
    )

    distance_rows = build_group_rows(
        by_distance,
        (
            "distance",
        ),
    )

    occlusion_rows = build_group_rows(
        by_occlusion,
        (
            "target_occlusion",
        ),
    )

    snr_rows = build_group_rows(
        by_snr,
        (
            "target_snr_db",
        ),
    )

    cn2_distance_rows = build_group_rows(
        by_cn2_distance,
        (
            "cn2",
            "distance",
        ),
    )

    save_csv(
        CN2_OUTPUT_PATH,
        cn2_rows,
    )

    save_csv(
        DISTANCE_OUTPUT_PATH,
        distance_rows,
    )

    save_csv(
        OCCLUSION_OUTPUT_PATH,
        occlusion_rows,
    )

    save_csv(
        SNR_OUTPUT_PATH,
        snr_rows,
    )

    save_csv(
        CN2_DISTANCE_OUTPUT_PATH,
        cn2_distance_rows,
    )

    global_result = finalize_counter(
        global_counter
    )

    report_lines = [
        "MC-VWLS error-type analysis by condition",
        f"Prediction CSV: {PREDICTION_PATH}",
        f"Dataset: {DATASET_PATH}",
        f"Selected method: {TARGET_METHOD}",
        "",
        "[GLOBAL]",
        (
            f"samples={global_result['sample_count']}, "
            f"correct={global_result['correct_count']}, "
            f"errors={global_result['error_count']}, "
            f"accuracy={float(global_result['label_accuracy']):.8f}"
        ),
        (
            "same_order_error_fraction="
            f"{float(global_result['same_order_error_fraction_of_errors']):.8f}"
        ),
        (
            "cross_order_error_fraction="
            f"{float(global_result['cross_order_error_fraction_of_errors']):.8f}"
        ),
        (
            "adjacent_phase_error_fraction="
            f"{float(global_result['adjacent_phase_error_fraction_of_errors']):.8f}"
        ),
        (
            "same_order_adjacent_phase_error_fraction="
            f"{float(global_result['same_order_adjacent_phase_error_fraction_of_errors']):.8f}"
        ),
        "",
    ]

    report_lines.extend(
        format_report_group(
            "[BY CN2]",
            cn2_rows,
            (
                "cn2",
            ),
        )
    )

    report_lines.extend(
        format_report_group(
            "[BY DISTANCE]",
            distance_rows,
            (
                "distance",
            ),
        )
    )

    report_lines.extend(
        format_report_group(
            "[BY OCCLUSION]",
            occlusion_rows,
            (
                "target_occlusion",
            ),
        )
    )

    report_lines.extend(
        format_report_group(
            "[BY SNR]",
            snr_rows,
            (
                "target_snr_db",
            ),
        )
    )

    report_lines.extend(
        format_report_group(
            "[BY CN2 AND DISTANCE]",
            cn2_distance_rows,
            (
                "cn2",
                "distance",
            ),
        )
    )

    report_lines.extend(
        [
            "[OUTPUTS]",
            str(CN2_OUTPUT_PATH),
            str(DISTANCE_OUTPUT_PATH),
            str(OCCLUSION_OUTPUT_PATH),
            str(SNR_OUTPUT_PATH),
            str(CN2_DISTANCE_OUTPUT_PATH),
        ]
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    print(
        f"Global accuracy: "
        f"{float(global_result['label_accuracy']):.8f}"
    )

    print(
        f"Global cross-order error fraction: "
        f"{float(global_result['cross_order_error_fraction_of_errors']):.8f}"
    )

    print(
        "Saved:",
        CN2_OUTPUT_PATH,
    )

    print(
        "Saved:",
        DISTANCE_OUTPUT_PATH,
    )

    print(
        "Saved:",
        OCCLUSION_OUTPUT_PATH,
    )

    print(
        "Saved:",
        SNR_OUTPUT_PATH,
    )

    print(
        "Saved:",
        CN2_DISTANCE_OUTPUT_PATH,
    )

    print("")
    print("=" * 78)
    print("MC-VWLS ERROR-TYPE CONDITION SUMMARY COMPLETE")
    print("=" * 78)

    print(
        "Report:",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()