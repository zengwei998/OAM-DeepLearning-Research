"""
Generate main experimental figures from completed ablation results.

Inputs:
    results/csv/mc_vwls_ablation_summary.csv
    results/csv/ablation_by_cn2_distance.csv
    results/csv/ablation_accuracy_confidence_intervals.csv

Outputs:
    results/figures/fig_method_overall_accuracy.png
    results/figures/fig_method_overall_accuracy.pdf

    results/figures/fig_accuracy_vs_occlusion.png
    results/figures/fig_accuracy_vs_occlusion.pdf

    results/figures/fig_accuracy_vs_snr.png
    results/figures/fig_accuracy_vs_snr.pdf

    results/figures/fig_mc_vwls_cn2_distance_heatmap.png
    results/figures/fig_mc_vwls_cn2_distance_heatmap.pdf
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

ABLATION_SUMMARY_PATH = (
    ROOT
    / "results"
    / "csv"
    / "mc_vwls_ablation_summary.csv"
)

CN2_DISTANCE_PATH = (
    ROOT
    / "results"
    / "csv"
    / "ablation_by_cn2_distance.csv"
)

CI_PATH = (
    ROOT
    / "results"
    / "csv"
    / "ablation_accuracy_confidence_intervals.csv"
)

FIGURE_DIRECTORY = (
    ROOT
    / "results"
    / "figures"
)

METHOD_NAMES = (
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

OCCLUSION_LEVELS = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
)

SNR_LEVELS = (
    20.0,
    15.0,
    10.0,
    5.0,
    0.0,
)

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
            f"CSV contains no rows: {path}"
        )

    return rows


def find_summary_row(
    rows: List[Dict[str, str]],
    *,
    method: str,
    scope: str,
    target_snr_db: float | None = None,
    target_occlusion: float | None = None,
) -> Dict[str, str]:
    for row in rows:
        if row["method"] != method:
            continue

        if row["scope"] != scope:
            continue

        if target_snr_db is not None:
            if not np.isclose(
                float(
                    row["target_snr_db"]
                ),
                target_snr_db,
                rtol=0.0,
                atol=1.0e-8,
            ):
                continue

        if target_occlusion is not None:
            if not np.isclose(
                float(
                    row["target_occlusion"]
                ),
                target_occlusion,
                rtol=0.0,
                atol=1.0e-8,
            ):
                continue

        return row

    raise KeyError(
        "Summary row was not found: "
        f"method={method}, "
        f"scope={scope}, "
        f"snr={target_snr_db}, "
        f"occlusion={target_occlusion}"
    )


def find_ci_row(
    rows: List[Dict[str, str]],
    *,
    method: str,
) -> Dict[str, str]:
    for row in rows:
        if (
            row["method"] == method
            and row["metric"]
            == "label_correct"
        ):
            return row

    raise KeyError(
        f"CI row was not found for method: {method}"
    )


def save_figure(
    figure: plt.Figure,
    filename_stem: str,
) -> None:
    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        FIGURE_DIRECTORY
        / f"{filename_stem}.png"
    )

    pdf_path = (
        FIGURE_DIRECTORY
        / f"{filename_stem}.pdf"
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        "Saved:",
        png_path,
    )

    print(
        "Saved:",
        pdf_path,
    )


def plot_overall_accuracy(
    summary_rows: List[Dict[str, str]],
    ci_rows: List[Dict[str, str]],
) -> None:
    accuracies = []
    lower_errors = []
    upper_errors = []

    for method in METHOD_NAMES:
        summary = find_summary_row(
            summary_rows,
            method=method,
            scope="overall",
        )

        ci = find_ci_row(
            ci_rows,
            method=method,
        )

        accuracy = float(
            summary["label_accuracy"]
        )

        lower = float(
            ci["wilson_95_ci_lower"]
        )

        upper = float(
            ci["wilson_95_ci_upper"]
        )

        accuracies.append(
            accuracy
        )

        lower_errors.append(
            accuracy - lower
        )

        upper_errors.append(
            upper - accuracy
        )

    x_values = np.arange(
        len(METHOD_NAMES)
    )

    figure, axis = plt.subplots(
        figsize=(7.0, 4.8)
    )

    axis.bar(
        x_values,
        accuracies,
        yerr=np.asarray(
            [
                lower_errors,
                upper_errors,
            ]
        ),
        capsize=5,
    )

    axis.set_xticks(
        x_values
    )

    axis.set_xticklabels(
        [
            METHOD_LABELS[method]
            for method in METHOD_NAMES
        ]
    )

    axis.set_ylabel(
        "Label accuracy"
    )

    axis.set_ylim(
        0.83,
        0.86,
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    for x_value, accuracy in zip(
        x_values,
        accuracies,
    ):
        axis.text(
            x_value,
            accuracy + 0.001,
            f"{accuracy:.4f}",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()

    save_figure(
        figure,
        "fig_method_overall_accuracy",
    )


def plot_accuracy_vs_occlusion(
    summary_rows: List[Dict[str, str]],
) -> None:
    figure, axis = plt.subplots(
        figsize=(7.2, 4.8)
    )

    for method in METHOD_NAMES:
        values = []

        for occlusion in OCCLUSION_LEVELS:
            row = find_summary_row(
                summary_rows,
                method=method,
                scope="occlusion",
                target_occlusion=occlusion,
            )

            values.append(
                float(
                    row["label_accuracy"]
                )
            )

        axis.plot(
            OCCLUSION_LEVELS,
            values,
            marker="o",
            label=METHOD_LABELS[method],
        )

    axis.set_xlabel(
        "Target energy occlusion ratio"
    )

    axis.set_ylabel(
        "Label accuracy"
    )

    axis.set_xticks(
        OCCLUSION_LEVELS
    )

    axis.grid(
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    save_figure(
        figure,
        "fig_accuracy_vs_occlusion",
    )


def plot_accuracy_vs_snr(
    summary_rows: List[Dict[str, str]],
) -> None:
    figure, axis = plt.subplots(
        figsize=(7.2, 4.8)
    )

    x_values = tuple(
        reversed(
            SNR_LEVELS
        )
    )

    for method in METHOD_NAMES:
        values = []

        for snr_db in x_values:
            row = find_summary_row(
                summary_rows,
                method=method,
                scope="snr",
                target_snr_db=snr_db,
            )

            values.append(
                float(
                    row["label_accuracy"]
                )
            )

        axis.plot(
            x_values,
            values,
            marker="o",
            label=METHOD_LABELS[method],
        )

    axis.set_xlabel(
        "Nominal SNR (dB)"
    )

    axis.set_ylabel(
        "Label accuracy"
    )

    axis.set_xticks(
        x_values
    )

    axis.grid(
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    save_figure(
        figure,
        "fig_accuracy_vs_snr",
    )


def plot_cn2_distance_heatmap(
    cn2_rows: List[Dict[str, str]],
) -> None:
    matrix = np.empty(
        (
            len(CN2_VALUES),
            len(DISTANCES),
        ),
        dtype=np.float64,
    )

    for row_index, cn2 in enumerate(
        CN2_VALUES
    ):
        for column_index, distance in enumerate(
            DISTANCES
        ):
            matching_rows = [
                row
                for row in cn2_rows
                if (
                    row["method"]
                    == "A3_MC_VWLS"
                    and np.isclose(
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

            if len(matching_rows) != 1:
                raise ValueError(
                    "Expected one MC-VWLS condition row, "
                    f"found {len(matching_rows)} for "
                    f"Cn2={cn2:.3e}, "
                    f"distance={distance:.1f}"
                )

            matrix[
                row_index,
                column_index,
            ] = float(
                matching_rows[0][
                    "label_accuracy"
                ]
            )

    figure, axis = plt.subplots(
        figsize=(7.2, 5.6)
    )

    image = axis.imshow(
        matrix,
        aspect="auto",
        origin="upper",
        vmin=0.0,
        vmax=1.0,
    )

    axis.set_xticks(
        np.arange(
            len(DISTANCES)
        )
    )

    axis.set_xticklabels(
        [
            f"{distance:.0f}"
            for distance in DISTANCES
        ]
    )

    axis.set_yticks(
        np.arange(
            len(CN2_VALUES)
        )
    )

    axis.set_yticklabels(
        [
            f"{cn2:.1e}"
            for cn2 in CN2_VALUES
        ]
    )

    axis.set_xlabel(
        "Propagation distance"
    )

    axis.set_ylabel(
        r"$C_n^2$"
    )

    for row_index in range(
        matrix.shape[0]
    ):
        for column_index in range(
            matrix.shape[1]
        ):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.3f}",
                ha="center",
                va="center",
            )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )

    colorbar.set_label(
        "Label accuracy"
    )

    figure.tight_layout()

    save_figure(
        figure,
        "fig_mc_vwls_cn2_distance_heatmap",
    )


def main() -> None:
    print("=" * 78)
    print("GENERATE MAIN RESULT FIGURES")
    print("=" * 78)

    summary_rows = load_csv(
        ABLATION_SUMMARY_PATH
    )

    cn2_rows = load_csv(
        CN2_DISTANCE_PATH
    )

    ci_rows = load_csv(
        CI_PATH
    )

    plot_overall_accuracy(
        summary_rows,
        ci_rows,
    )

    plot_accuracy_vs_occlusion(
        summary_rows
    )

    plot_accuracy_vs_snr(
        summary_rows
    )

    plot_cn2_distance_heatmap(
        cn2_rows
    )

    print("")
    print("=" * 78)
    print("MAIN RESULT FIGURES COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()