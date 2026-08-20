"""
Generate the 32-class normalized confusion matrix for MC-VWLS.

Input:
    results/csv/mc_vwls_ablation_full_test_predictions.csv

Outputs:
    results/figures/fig_mc_vwls_confusion_matrix.png
    results/figures/fig_mc_vwls_confusion_matrix.pdf
    results/csv/mc_vwls_confusion_matrix_counts.csv
    results/csv/mc_vwls_confusion_matrix_normalized.csv
    results/validation/mc_vwls_confusion_matrix_report.txt
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

PREDICTION_PATH = (
    ROOT
    / "results"
    / "csv"
    / "mc_vwls_ablation_full_test_predictions.csv"
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
    / "mc_vwls_confusion_matrix_report.txt"
)

PNG_PATH = (
    FIGURE_DIRECTORY
    / "fig_mc_vwls_confusion_matrix.png"
)

PDF_PATH = (
    FIGURE_DIRECTORY
    / "fig_mc_vwls_confusion_matrix.pdf"
)

COUNT_MATRIX_PATH = (
    CSV_DIRECTORY
    / "mc_vwls_confusion_matrix_counts.csv"
)

NORMALIZED_MATRIX_PATH = (
    CSV_DIRECTORY
    / "mc_vwls_confusion_matrix_normalized.csv"
)

TARGET_METHOD = "A3_MC_VWLS"

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

TRUE_LABEL_ALIASES = (
    "true_label",
    "target_label",
    "label",
    "ground_truth_label",
)

PREDICTED_LABEL_ALIASES = (
    "predicted_label",
    "prediction",
    "pred_label",
    "estimated_label",
)

TRUE_ORDER_ALIASES = (
    "true_order",
    "target_order",
    "oam_order",
    "ground_truth_order",
)

PREDICTED_ORDER_ALIASES = (
    "predicted_order",
    "pred_order",
    "estimated_order",
)

TRUE_PHASE_ALIASES = (
    "true_phase_bin",
    "target_phase_bin",
    "phase_bin",
    "ground_truth_phase_bin",
)

PREDICTED_PHASE_ALIASES = (
    "predicted_phase_bin",
    "pred_phase_bin",
    "estimated_phase_bin",
)

METHOD_ALIASES = (
    "method",
    "algorithm",
    "variant",
)


def load_csv(
    path: Path,
) -> Tuple[List[Dict[str, str]], List[str]]:
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

        rows = list(
            reader
        )

    if not fieldnames:
        raise ValueError(
            f"CSV has no header: {path}"
        )

    if not rows:
        raise ValueError(
            f"CSV has no data rows: {path}"
        )

    return rows, fieldnames


def find_column(
    fieldnames: Sequence[str],
    aliases: Iterable[str],
    *,
    required: bool,
) -> str | None:
    lookup = {
        name.strip().lower(): name
        for name in fieldnames
    }

    for alias in aliases:
        key = alias.strip().lower()

        if key in lookup:
            return lookup[key]

    if required:
        raise KeyError(
            "Required column was not found. "
            f"Accepted names: {tuple(aliases)}. "
            f"Available columns: {tuple(fieldnames)}"
        )

    return None


def parse_integer(
    value: str,
    *,
    field_name: str,
) -> int:
    try:
        return int(
            round(
                float(
                    value
                )
            )
        )

    except Exception as error:
        raise ValueError(
            f"Cannot parse integer value in "
            f"{field_name}: {value!r}"
        ) from error


def encode_class(
    order: int,
    phase_bin: int,
) -> int:
    if order not in OAM_ORDERS:
        raise ValueError(
            f"Invalid OAM order: {order}"
        )

    if phase_bin not in PHASE_BINS:
        raise ValueError(
            f"Invalid phase bin: {phase_bin}"
        )

    return (
        (order - 1)
        * len(PHASE_BINS)
        + phase_bin
    )


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


def make_class_labels() -> List[str]:
    labels = []

    for class_index in range(
        CLASS_COUNT
    ):
        order, phase_bin = decode_class(
            class_index
        )

        labels.append(
            f"l={order},p={phase_bin}"
        )

    return labels


def extract_labels(
    rows: Sequence[Dict[str, str]],
    fieldnames: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, int]:
    method_column = find_column(
        fieldnames,
        METHOD_ALIASES,
        required=False,
    )

    true_label_column = find_column(
        fieldnames,
        TRUE_LABEL_ALIASES,
        required=False,
    )

    predicted_label_column = find_column(
        fieldnames,
        PREDICTED_LABEL_ALIASES,
        required=False,
    )

    true_order_column = find_column(
        fieldnames,
        TRUE_ORDER_ALIASES,
        required=False,
    )

    predicted_order_column = find_column(
        fieldnames,
        PREDICTED_ORDER_ALIASES,
        required=False,
    )

    true_phase_column = find_column(
        fieldnames,
        TRUE_PHASE_ALIASES,
        required=False,
    )

    predicted_phase_column = find_column(
        fieldnames,
        PREDICTED_PHASE_ALIASES,
        required=False,
    )

    has_direct_labels = (
        true_label_column is not None
        and predicted_label_column is not None
    )

    has_component_labels = (
        true_order_column is not None
        and predicted_order_column is not None
        and true_phase_column is not None
        and predicted_phase_column is not None
    )

    if not (
        has_direct_labels
        or has_component_labels
    ):
        raise KeyError(
            "The prediction CSV must contain either "
            "true/predicted label columns or "
            "true/predicted order and phase-bin columns. "
            f"Available columns: {tuple(fieldnames)}"
        )

    true_labels: List[int] = []
    predicted_labels: List[int] = []
    selected_rows = 0

    for row in rows:
        if method_column is not None:
            method = str(
                row[
                    method_column
                ]
            ).strip()

            if method != TARGET_METHOD:
                continue

        selected_rows += 1

        if has_component_labels:
            true_order = parse_integer(
                row[
                    true_order_column
                ],
                field_name=true_order_column,
            )

            true_phase = parse_integer(
                row[
                    true_phase_column
                ],
                field_name=true_phase_column,
            )

            predicted_order = parse_integer(
                row[
                    predicted_order_column
                ],
                field_name=predicted_order_column,
            )

            predicted_phase = parse_integer(
                row[
                    predicted_phase_column
                ],
                field_name=predicted_phase_column,
            )

            true_label = encode_class(
                true_order,
                true_phase,
            )

            predicted_label = encode_class(
                predicted_order,
                predicted_phase,
            )

        else:
            true_label = parse_integer(
                row[
                    true_label_column
                ],
                field_name=true_label_column,
            )

            predicted_label = parse_integer(
                row[
                    predicted_label_column
                ],
                field_name=predicted_label_column,
            )

            if not (
                0
                <= true_label
                < CLASS_COUNT
            ):
                raise ValueError(
                    f"True label outside 0–31: {true_label}"
                )

            if not (
                0
                <= predicted_label
                < CLASS_COUNT
            ):
                raise ValueError(
                    "Predicted label outside 0–31: "
                    f"{predicted_label}"
                )

        true_labels.append(
            true_label
        )

        predicted_labels.append(
            predicted_label
        )

    if selected_rows == 0:
        raise ValueError(
            f"No rows selected for method: {TARGET_METHOD}"
        )

    return (
        np.asarray(
            true_labels,
            dtype=np.int64,
        ),
        np.asarray(
            predicted_labels,
            dtype=np.int64,
        ),
        selected_rows,
    )


def calculate_confusion_matrix(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> np.ndarray:
    matrix = np.zeros(
        (
            CLASS_COUNT,
            CLASS_COUNT,
        ),
        dtype=np.int64,
    )

    np.add.at(
        matrix,
        (
            true_labels,
            predicted_labels,
        ),
        1,
    )

    return matrix


def normalize_rows(
    matrix: np.ndarray,
) -> np.ndarray:
    row_sums = matrix.sum(
        axis=1,
        keepdims=True,
    )

    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(
            matrix,
            dtype=np.float64,
        ),
        where=row_sums > 0,
    )

    return normalized


def save_matrix_csv(
    path: Path,
    matrix: np.ndarray,
    class_labels: Sequence[str],
    *,
    decimal_digits: int | None,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.writer(
            csv_file
        )

        writer.writerow(
            [
                "true_class",
                *class_labels,
            ]
        )

        for row_index, label in enumerate(
            class_labels
        ):
            values = []

            for value in matrix[
                row_index
            ]:
                if decimal_digits is None:
                    values.append(
                        int(
                            value
                        )
                    )
                else:
                    values.append(
                        f"{float(value):.{decimal_digits}f}"
                    )

            writer.writerow(
                [
                    label,
                    *values,
                ]
            )


def plot_confusion_matrix(
    normalized_matrix: np.ndarray,
    class_labels: Sequence[str],
) -> None:
    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(13.0, 11.0)
    )

    image = axis.imshow(
        normalized_matrix,
        origin="upper",
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
    )

    tick_locations = np.arange(
        CLASS_COUNT
    )

    axis.set_xticks(
        tick_locations
    )

    axis.set_yticks(
        tick_locations
    )

    axis.set_xticklabels(
        class_labels,
        rotation=90,
        fontsize=7,
    )

    axis.set_yticklabels(
        class_labels,
        fontsize=7,
    )

    axis.set_xlabel(
        "Predicted class"
    )

    axis.set_ylabel(
        "True class"
    )

    for boundary in (
        7.5,
        15.5,
        23.5,
    ):
        axis.axhline(
            boundary,
            linewidth=1.0,
        )

        axis.axvline(
            boundary,
            linewidth=1.0,
        )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Row-normalized probability"
    )

    figure.tight_layout()

    figure.savefig(
        PNG_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        PDF_PATH,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def main() -> None:
    print("=" * 78)
    print("GENERATE MC-VWLS CONFUSION MATRIX")
    print("=" * 78)

    rows, fieldnames = load_csv(
        PREDICTION_PATH
    )

    print(
        "Prediction rows:",
        len(rows),
    )

    print(
        "Columns:",
        fieldnames,
    )

    true_labels, predicted_labels, selected_rows = extract_labels(
        rows,
        fieldnames,
    )

    count_matrix = calculate_confusion_matrix(
        true_labels,
        predicted_labels,
    )

    normalized_matrix = normalize_rows(
        count_matrix
    )

    class_labels = make_class_labels()

    save_matrix_csv(
        COUNT_MATRIX_PATH,
        count_matrix,
        class_labels,
        decimal_digits=None,
    )

    save_matrix_csv(
        NORMALIZED_MATRIX_PATH,
        normalized_matrix,
        class_labels,
        decimal_digits=8,
    )

    plot_confusion_matrix(
        normalized_matrix,
        class_labels,
    )

    observed_accuracy = float(
        np.mean(
            true_labels
            == predicted_labels
        )
    )

    class_support = count_matrix.sum(
        axis=1
    )

    missing_classes = [
        class_labels[index]
        for index, support in enumerate(
            class_support
        )
        if support == 0
    ]

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_lines = [
        "MC-VWLS confusion matrix report",
        f"Prediction CSV: {PREDICTION_PATH}",
        f"Target method: {TARGET_METHOD}",
        f"Selected observations: {selected_rows}",
        f"Class count: {CLASS_COUNT}",
        f"Observed label accuracy: {observed_accuracy:.8f}",
        f"Missing true classes: {len(missing_classes)}",
    ]

    if missing_classes:
        report_lines.append(
            "Missing class labels: "
            + ", ".join(
                missing_classes
            )
        )

    report_lines.extend(
        [
            "",
            f"Count matrix CSV: {COUNT_MATRIX_PATH}",
            f"Normalized matrix CSV: {NORMALIZED_MATRIX_PATH}",
            f"PNG figure: {PNG_PATH}",
            f"PDF figure: {PDF_PATH}",
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    print(
        f"Selected observations: {selected_rows}"
    )

    print(
        f"Observed label accuracy: {observed_accuracy:.8f}"
    )

    print(
        "Saved:",
        COUNT_MATRIX_PATH,
    )

    print(
        "Saved:",
        NORMALIZED_MATRIX_PATH,
    )

    print(
        "Saved:",
        PNG_PATH,
    )

    print(
        "Saved:",
        PDF_PATH,
    )

    print("")
    print("=" * 78)
    print("MC-VWLS CONFUSION MATRIX COMPLETE")
    print("=" * 78)

    print(
        "Report:",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()