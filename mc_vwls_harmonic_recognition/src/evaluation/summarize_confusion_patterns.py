"""
Summarize MC-VWLS confusion patterns.

Input:
    results/csv/mc_vwls_confusion_matrix_counts.csv

Outputs:
    results/csv/mc_vwls_confusion_pattern_summary.csv
    results/csv/mc_vwls_top_confusions.csv
    results/validation/mc_vwls_confusion_pattern_report.txt
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

COUNT_MATRIX_PATH = (
    ROOT
    / "results"
    / "csv"
    / "mc_vwls_confusion_matrix_counts.csv"
)

SUMMARY_PATH = (
    ROOT
    / "results"
    / "csv"
    / "mc_vwls_confusion_pattern_summary.csv"
)

TOP_CONFUSIONS_PATH = (
    ROOT
    / "results"
    / "csv"
    / "mc_vwls_top_confusions.csv"
)

REPORT_PATH = (
    ROOT
    / "results"
    / "validation"
    / "mc_vwls_confusion_pattern_report.txt"
)

OAM_ORDERS = (
    1,
    2,
    3,
    4,
)

PHASE_BINS = tuple(
    range(8)
)

CLASS_COUNT = (
    len(OAM_ORDERS)
    * len(PHASE_BINS)
)

TOP_CONFUSION_COUNT = 20


def decode_class(
    class_index: int,
) -> Tuple[int, int]:
    if not (
        0
        <= class_index
        < CLASS_COUNT
    ):
        raise ValueError(
            f"Invalid class index: {class_index}"
        )

    order = (
        class_index
        // len(PHASE_BINS)
        + 1
    )

    phase_bin = (
        class_index
        % len(PHASE_BINS)
    )

    return order, phase_bin


def circular_phase_distance(
    first_phase: int,
    second_phase: int,
) -> int:
    direct_distance = abs(
        first_phase
        - second_phase
    )

    return min(
        direct_distance,
        len(PHASE_BINS)
        - direct_distance,
    )


def load_count_matrix(
    path: Path,
) -> Tuple[np.ndarray, List[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Count matrix CSV does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.reader(
            csv_file
        )

        rows = list(
            reader
        )

    if len(rows) != CLASS_COUNT + 1:
        raise ValueError(
            "Unexpected count-matrix row count: "
            f"expected={CLASS_COUNT + 1}, "
            f"actual={len(rows)}"
        )

    header = rows[0]

    if len(header) != CLASS_COUNT + 1:
        raise ValueError(
            "Unexpected count-matrix column count: "
            f"expected={CLASS_COUNT + 1}, "
            f"actual={len(header)}"
        )

    class_labels = header[1:]

    matrix = np.zeros(
        (
            CLASS_COUNT,
            CLASS_COUNT,
        ),
        dtype=np.int64,
    )

    for row_index, row in enumerate(
        rows[1:]
    ):
        if len(row) != CLASS_COUNT + 1:
            raise ValueError(
                "Unexpected row length in count matrix: "
                f"row={row_index}, "
                f"expected={CLASS_COUNT + 1}, "
                f"actual={len(row)}"
            )

        row_label = row[0]

        if row_label != class_labels[
            row_index
        ]:
            raise ValueError(
                "Row label does not match header class label: "
                f"row={row_index}, "
                f"row_label={row_label}, "
                f"header_label={class_labels[row_index]}"
            )

        matrix[
            row_index
        ] = np.asarray(
            [
                int(
                    value
                )
                for value in row[1:]
            ],
            dtype=np.int64,
        )

    return matrix, class_labels


def save_csv(
    path: Path,
    rows: Sequence[Dict[str, object]],
) -> None:
    if not rows:
        raise ValueError(
            f"No rows available for output: {path}"
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


def calculate_global_statistics(
    matrix: np.ndarray,
) -> Dict[str, float | int]:
    total_observations = int(
        matrix.sum()
    )

    correct_observations = int(
        np.trace(
            matrix
        )
    )

    error_observations = (
        total_observations
        - correct_observations
    )

    same_order_error_count = 0
    cross_order_error_count = 0
    adjacent_phase_error_count = 0
    nonadjacent_phase_error_count = 0
    same_order_adjacent_phase_error_count = 0

    for true_index in range(
        CLASS_COUNT
    ):
        true_order, true_phase = decode_class(
            true_index
        )

        for predicted_index in range(
            CLASS_COUNT
        ):
            if true_index == predicted_index:
                continue

            count = int(
                matrix[
                    true_index,
                    predicted_index,
                ]
            )

            if count == 0:
                continue

            predicted_order, predicted_phase = decode_class(
                predicted_index
            )

            phase_distance = circular_phase_distance(
                true_phase,
                predicted_phase,
            )

            if true_order == predicted_order:
                same_order_error_count += count

                if phase_distance == 1:
                    same_order_adjacent_phase_error_count += count

            else:
                cross_order_error_count += count

            if phase_distance == 1:
                adjacent_phase_error_count += count
            else:
                nonadjacent_phase_error_count += count

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

    return {
        "total_observations": total_observations,
        "correct_observations": correct_observations,
        "error_observations": error_observations,
        "overall_accuracy": safe_ratio(
            correct_observations,
            total_observations,
        ),
        "same_order_error_count": same_order_error_count,
        "same_order_error_fraction_of_all_errors": safe_ratio(
            same_order_error_count,
            error_observations,
        ),
        "cross_order_error_count": cross_order_error_count,
        "cross_order_error_fraction_of_all_errors": safe_ratio(
            cross_order_error_count,
            error_observations,
        ),
        "adjacent_phase_error_count": adjacent_phase_error_count,
        "adjacent_phase_error_fraction_of_all_errors": safe_ratio(
            adjacent_phase_error_count,
            error_observations,
        ),
        "nonadjacent_phase_error_count": nonadjacent_phase_error_count,
        "nonadjacent_phase_error_fraction_of_all_errors": safe_ratio(
            nonadjacent_phase_error_count,
            error_observations,
        ),
        "same_order_adjacent_phase_error_count": (
            same_order_adjacent_phase_error_count
        ),
        "same_order_adjacent_phase_error_fraction_of_all_errors": (
            safe_ratio(
                same_order_adjacent_phase_error_count,
                error_observations,
            )
        ),
    }


def build_order_summary(
    matrix: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[
        Dict[str, object]
    ] = []

    for order in OAM_ORDERS:
        start_index = (
            (order - 1)
            * len(PHASE_BINS)
        )

        stop_index = (
            start_index
            + len(PHASE_BINS)
        )

        order_rows = matrix[
            start_index:stop_index,
            :
        ]

        total = int(
            order_rows.sum()
        )

        diagonal_correct = int(
            sum(
                matrix[
                    class_index,
                    class_index,
                ]
                for class_index in range(
                    start_index,
                    stop_index,
                )
            )
        )

        predicted_same_order = int(
            matrix[
                start_index:stop_index,
                start_index:stop_index,
            ].sum()
        )

        order_correct = (
            predicted_same_order
        )

        phase_correct = 0

        for true_index in range(
            start_index,
            stop_index,
        ):
            _, true_phase = decode_class(
                true_index
            )

            for predicted_index in range(
                CLASS_COUNT
            ):
                _, predicted_phase = decode_class(
                    predicted_index
                )

                if (
                    true_phase
                    == predicted_phase
                ):
                    phase_correct += int(
                        matrix[
                            true_index,
                            predicted_index,
                        ]
                    )

        rows.append(
            {
                "oam_order": order,
                "sample_count": total,
                "label_correct_count": diagonal_correct,
                "label_accuracy": (
                    diagonal_correct
                    / total
                ),
                "order_correct_count": order_correct,
                "order_accuracy": (
                    order_correct
                    / total
                ),
                "phase_correct_count": phase_correct,
                "phase_accuracy": (
                    phase_correct
                    / total
                ),
            }
        )

    return rows


def build_top_confusions(
    matrix: np.ndarray,
    class_labels: Sequence[str],
) -> List[Dict[str, object]]:
    confusion_rows: List[
        Dict[str, object]
    ] = []

    for true_index in range(
        CLASS_COUNT
    ):
        true_order, true_phase = decode_class(
            true_index
        )

        true_support = int(
            matrix[
                true_index
            ].sum()
        )

        for predicted_index in range(
            CLASS_COUNT
        ):
            if true_index == predicted_index:
                continue

            count = int(
                matrix[
                    true_index,
                    predicted_index,
                ]
            )

            if count == 0:
                continue

            predicted_order, predicted_phase = decode_class(
                predicted_index
            )

            confusion_rows.append(
                {
                    "true_class": class_labels[
                        true_index
                    ],
                    "predicted_class": class_labels[
                        predicted_index
                    ],
                    "true_order": true_order,
                    "true_phase_bin": true_phase,
                    "predicted_order": predicted_order,
                    "predicted_phase_bin": predicted_phase,
                    "phase_bin_distance": circular_phase_distance(
                        true_phase,
                        predicted_phase,
                    ),
                    "same_order": int(
                        true_order
                        == predicted_order
                    ),
                    "count": count,
                    "fraction_of_true_class": (
                        count
                        / true_support
                    ),
                }
            )

    confusion_rows.sort(
        key=lambda row: (
            -int(
                row["count"]
            ),
            -float(
                row[
                    "fraction_of_true_class"
                ]
            ),
            str(
                row["true_class"]
            ),
            str(
                row["predicted_class"]
            ),
        )
    )

    return confusion_rows[
        :TOP_CONFUSION_COUNT
    ]


def main() -> None:
    print("=" * 78)
    print("SUMMARIZE MC-VWLS CONFUSION PATTERNS")
    print("=" * 78)

    matrix, class_labels = load_count_matrix(
        COUNT_MATRIX_PATH
    )

    global_statistics = calculate_global_statistics(
        matrix
    )

    order_summary = build_order_summary(
        matrix
    )

    top_confusions = build_top_confusions(
        matrix,
        class_labels,
    )

    summary_rows: List[
        Dict[str, object]
    ] = []

    for key, value in (
        global_statistics.items()
    ):
        summary_rows.append(
            {
                "scope": "global",
                "item": key,
                "value": (
                    f"{value:.8f}"
                    if isinstance(
                        value,
                        float,
                    )
                    else value
                ),
            }
        )

    for row in order_summary:
        order = int(
            row["oam_order"]
        )

        for key in (
            "sample_count",
            "label_correct_count",
            "label_accuracy",
            "order_correct_count",
            "order_accuracy",
            "phase_correct_count",
            "phase_accuracy",
        ):
            value = row[
                key
            ]

            summary_rows.append(
                {
                    "scope": f"oam_order_{order}",
                    "item": key,
                    "value": (
                        f"{value:.8f}"
                        if isinstance(
                            value,
                            float,
                        )
                        else value
                    ),
                }
            )

    save_csv(
        SUMMARY_PATH,
        summary_rows,
    )

    save_csv(
        TOP_CONFUSIONS_PATH,
        top_confusions,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_lines = [
        "MC-VWLS confusion-pattern report",
        f"Count matrix: {COUNT_MATRIX_PATH}",
        "",
        "[GLOBAL]",
        (
            "Total observations: "
            f"{global_statistics['total_observations']}"
        ),
        (
            "Correct observations: "
            f"{global_statistics['correct_observations']}"
        ),
        (
            "Error observations: "
            f"{global_statistics['error_observations']}"
        ),
        (
            "Overall accuracy: "
            f"{global_statistics['overall_accuracy']:.8f}"
        ),
        (
            "Same-order errors: "
            f"{global_statistics['same_order_error_count']} "
            f"({global_statistics['same_order_error_fraction_of_all_errors']:.8f})"
        ),
        (
            "Cross-order errors: "
            f"{global_statistics['cross_order_error_count']} "
            f"({global_statistics['cross_order_error_fraction_of_all_errors']:.8f})"
        ),
        (
            "Adjacent-phase errors: "
            f"{global_statistics['adjacent_phase_error_count']} "
            f"({global_statistics['adjacent_phase_error_fraction_of_all_errors']:.8f})"
        ),
        (
            "Same-order adjacent-phase errors: "
            f"{global_statistics['same_order_adjacent_phase_error_count']} "
            f"({global_statistics['same_order_adjacent_phase_error_fraction_of_all_errors']:.8f})"
        ),
        "",
        "[BY OAM ORDER]",
    ]

    for row in order_summary:
        report_lines.append(
            (
                f"order={row['oam_order']}, "
                f"samples={row['sample_count']}, "
                f"label_accuracy={row['label_accuracy']:.8f}, "
                f"order_accuracy={row['order_accuracy']:.8f}, "
                f"phase_accuracy={row['phase_accuracy']:.8f}"
            )
        )

    report_lines.extend(
        [
            "",
            "[TOP 20 CONFUSIONS]",
        ]
    )

    for rank, row in enumerate(
        top_confusions,
        start=1,
    ):
        report_lines.append(
            (
                f"{rank:02d}. "
                f"{row['true_class']} -> "
                f"{row['predicted_class']}, "
                f"count={row['count']}, "
                f"fraction_of_true_class="
                f"{row['fraction_of_true_class']:.8f}, "
                f"same_order={row['same_order']}, "
                f"phase_distance={row['phase_bin_distance']}"
            )
        )

    report_lines.extend(
        [
            "",
            f"Summary CSV: {SUMMARY_PATH}",
            f"Top-confusion CSV: {TOP_CONFUSIONS_PATH}",
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    print(
        f"Total observations: "
        f"{global_statistics['total_observations']}"
    )

    print(
        f"Overall accuracy: "
        f"{global_statistics['overall_accuracy']:.8f}"
    )

    print(
        f"Same-order error fraction: "
        f"{global_statistics['same_order_error_fraction_of_all_errors']:.8f}"
    )

    print(
        f"Cross-order error fraction: "
        f"{global_statistics['cross_order_error_fraction_of_all_errors']:.8f}"
    )

    print(
        f"Adjacent-phase error fraction: "
        f"{global_statistics['adjacent_phase_error_fraction_of_all_errors']:.8f}"
    )

    print(
        "Saved:",
        SUMMARY_PATH,
    )

    print(
        "Saved:",
        TOP_CONFUSIONS_PATH,
    )

    print("")
    print("=" * 78)
    print("MC-VWLS CONFUSION PATTERN SUMMARY COMPLETE")
    print("=" * 78)

    print(
        "Report:",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()