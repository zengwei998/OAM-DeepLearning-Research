"""
Generate publication-oriented quality-control figures and summary tables
for the corrected local-occlusion dataset.

Input:
    data/generated/occlusion_clean_v2.h5

Outputs:
    results/figures/fig_occlusion_qc.png
    results/figures/fig_occlusion_qc.pdf
    results/csv/occlusion_level_summary.csv
    results/validation/occlusion_qc_report.txt
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib

# Use a non-interactive backend so the script can save figures reliably.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

H5_PATH = (
    ROOT
    / "data"
    / "generated"
    / "occlusion_clean_v2.h5"
)

FIGURE_DIR = (
    ROOT
    / "results"
    / "figures"
)

CSV_DIR = (
    ROOT
    / "results"
    / "csv"
)

VALIDATION_DIR = (
    ROOT
    / "results"
    / "validation"
)

PNG_PATH = (
    FIGURE_DIR
    / "fig_occlusion_qc.png"
)

PDF_PATH = (
    FIGURE_DIR
    / "fig_occlusion_qc.pdf"
)

CSV_PATH = (
    CSV_DIR
    / "occlusion_level_summary.csv"
)

REPORT_PATH = (
    VALIDATION_DIR
    / "occlusion_qc_report.txt"
)

TARGET_LEVELS = np.asarray(
    [0.0, 0.1, 0.2, 0.3, 0.4],
    dtype=np.float64,
)

# Representative physical condition used in the displayed example.
# State 15 corresponds to:
# l = 2, phase bin = 7
SELECTED_LABEL = 15
SELECTED_CN2 = 1.0e-14
SELECTED_DISTANCE = 750.0
SELECTED_PROPAGATION_SEED = 0

DISPLAY_PERCENTILE = 99.8
LOG_EPSILON = 1.0e-12


def configure_matplotlib() -> None:
    """
    Configure consistent publication-oriented formatting.
    """

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "axes.linewidth": 0.8,
        }
    )


def find_representative_group(
    labels: np.ndarray,
    conditions: np.ndarray,
    base_indices: np.ndarray,
) -> Tuple[int, np.ndarray]:
    """
    Find the five nested occlusion observations for one fixed physical
    realization.

    Returns:
        selected base index,
        five HDF5 sample indices ordered by target occlusion level.
    """

    candidate_indices = np.where(
        (labels == SELECTED_LABEL)
        & np.isclose(
            conditions[:, 0],
            SELECTED_CN2,
            rtol=1.0e-5,
            atol=0.0,
        )
        & np.isclose(
            conditions[:, 1],
            SELECTED_DISTANCE,
            atol=1.0e-6,
        )
        & np.isclose(
            conditions[:, 2],
            SELECTED_PROPAGATION_SEED,
            atol=1.0e-6,
        )
    )[0]

    if len(candidate_indices) != 5:
        raise ValueError(
            "Expected five samples for the representative condition, "
            f"but found {len(candidate_indices)}."
        )

    selected_base_values = np.unique(
        base_indices[candidate_indices]
    )

    if len(selected_base_values) != 1:
        raise ValueError(
            "Representative samples do not share one base index."
        )

    order = np.argsort(
        conditions[candidate_indices, 3]
    )

    ordered_indices = candidate_indices[order]

    if not np.allclose(
        conditions[ordered_indices, 3],
        TARGET_LEVELS,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Representative group does not contain the five expected "
            "occlusion levels."
        )

    return (
        int(selected_base_values[0]),
        ordered_indices,
    )


def calculate_level_statistics(
    conditions: np.ndarray,
    masks: h5py.Dataset,
) -> List[Dict[str, float]]:
    """
    Calculate dataset-level statistics for each target occlusion level.
    """

    rows: List[Dict[str, float]] = []

    target_values = conditions[:, 3]
    achieved_values = conditions[:, 4]
    radii = conditions[:, 7]

    for target in TARGET_LEVELS:
        selected_indices = np.where(
            np.isclose(
                target_values,
                target,
                atol=1.0e-12,
            )
        )[0]

        achieved = achieved_values[selected_indices]
        radius = radii[selected_indices]

        # Reading every binary mask would be unnecessary.
        # Use an evenly spaced reproducible subset for geometric statistics.
        subset_size = min(
            512,
            len(selected_indices),
        )

        subset_positions = np.linspace(
            0,
            len(selected_indices) - 1,
            subset_size,
            dtype=np.int64,
        )

        subset_indices = selected_indices[
            subset_positions
        ]

        visible_fractions = []

        for index in subset_indices:
            mask = masks[int(index)]
            visible_fractions.append(
                float(np.mean(mask))
            )

        visible_fractions_array = np.asarray(
            visible_fractions,
            dtype=np.float64,
        )

        absolute_error = np.abs(
            achieved - target
        )

        rows.append(
            {
                "target_occlusion_ratio": float(target),
                "sample_count": int(len(selected_indices)),
                "achieved_mean": float(np.mean(achieved)),
                "achieved_std": float(np.std(achieved)),
                "achieved_min": float(np.min(achieved)),
                "achieved_max": float(np.max(achieved)),
                "absolute_error_mean": float(
                    np.mean(absolute_error)
                ),
                "absolute_error_max": float(
                    np.max(absolute_error)
                ),
                "radius_mean_pixels": float(
                    np.mean(radius)
                ),
                "radius_std_pixels": float(
                    np.std(radius)
                ),
                "radius_min_pixels": float(
                    np.min(radius)
                ),
                "radius_max_pixels": float(
                    np.max(radius)
                ),
                "visible_area_fraction_mean": float(
                    np.mean(visible_fractions_array)
                ),
                "visible_area_fraction_std": float(
                    np.std(visible_fractions_array)
                ),
            }
        )

    return rows


def save_summary_csv(
    rows: List[Dict[str, float]],
) -> None:
    """
    Save level-wise statistics as a reusable CSV table.
    """

    CSV_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
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
        "radius_min_pixels",
        "radius_max_pixels",
        "visible_area_fraction_mean",
        "visible_area_fraction_std",
    ]

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def create_qc_figure(
    intensity_ds: h5py.Dataset,
    mask_ds: h5py.Dataset,
    conditions: np.ndarray,
    sample_indices: np.ndarray,
) -> None:
    """
    Generate a two-row publication-quality visualization.

    Row 1:
        Log-scaled received intensity.

    Row 2:
        Binary visible-region mask.
    """

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    intensities = [
        np.asarray(
            intensity_ds[int(index)],
            dtype=np.float64,
        )
        for index in sample_indices
    ]

    masks = [
        np.asarray(
            mask_ds[int(index)],
            dtype=np.uint8,
        )
        for index in sample_indices
    ]

    # One common display range ensures fair visual comparison.
    nonzero_values = np.concatenate(
        [
            image[image > 0]
            for image in intensities
            if np.any(image > 0)
        ]
    )

    linear_upper = float(
        np.percentile(
            nonzero_values,
            DISPLAY_PERCENTILE,
        )
    )

    log_images = [
        np.log10(
            image / linear_upper
            + LOG_EPSILON
        )
        for image in intensities
    ]

    log_lower = -6.0
    log_upper = 0.0

    figure, axes = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=(11.5, 4.9),
        constrained_layout=True,
    )

    intensity_artist = None

    for column, sample_index in enumerate(
        sample_indices
    ):
        target = float(
            conditions[int(sample_index), 3]
        )

        achieved = float(
            conditions[int(sample_index), 4]
        )

        radius = float(
            conditions[int(sample_index), 7]
        )

        intensity_artist = axes[0, column].imshow(
            log_images[column],
            origin="lower",
            cmap="inferno",
            vmin=log_lower,
            vmax=log_upper,
            interpolation="nearest",
        )

        axes[0, column].set_title(
            (
                rf"$\eta_E^{{target}}={target:.1f}$"
                "\n"
                rf"$\eta_E^{{actual}}={achieved:.4f}$"
            )
        )

        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])

        axes[1, column].imshow(
            masks[column],
            origin="lower",
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

        axes[1, column].set_title(
            rf"$r_m={radius:.1f}$ pixels"
        )

        axes[1, column].set_xticks([])
        axes[1, column].set_yticks([])

    axes[0, 0].set_ylabel(
        "Received intensity\n(log scale)"
    )

    axes[1, 0].set_ylabel(
        "Visibility mask\n(white = visible)"
    )

    if intensity_artist is not None:
        colorbar = figure.colorbar(
            intensity_artist,
            ax=axes[0, :],
            location="right",
            shrink=0.82,
            pad=0.015,
        )

        colorbar.set_label(
            r"$\log_{10}(I/I_{99.8\%})$"
        )

    figure.suptitle(
        (
            "Nested random local occlusion under a fixed turbulence "
            "realization\n"
            rf"state={SELECTED_LABEL}, "
            rf"$C_n^2={SELECTED_CN2:.1e}\ \mathrm{{m^{{-2/3}}}}$, "
            rf"$z={SELECTED_DISTANCE:.0f}\ \mathrm{{m}}$, "
            rf"seed={SELECTED_PROPAGATION_SEED}"
        ),
        y=1.03,
    )

    figure.savefig(
        PNG_PATH,
        bbox_inches="tight",
    )

    figure.savefig(
        PDF_PATH,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_report(
    selected_base_index: int,
    sample_indices: np.ndarray,
    conditions: np.ndarray,
    rows: List[Dict[str, float]],
) -> None:
    """
    Save a human-readable QC report.
    """

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = []

    lines.append(
        "Occlusion dataset v2 visualization and summary report"
    )

    lines.append(
        f"Source: {H5_PATH}"
    )

    lines.append(
        f"Selected base index: {selected_base_index}"
    )

    lines.append(
        "Selected HDF5 sample indices: "
        + str(
            [
                int(index)
                for index in sample_indices
            ]
        )
    )

    lines.append("")
    lines.append("Representative conditions:")

    for index in sample_indices:
        condition = conditions[int(index)]

        lines.append(
            (
                f"index={int(index)}, "
                f"target={condition[3]:.6f}, "
                f"achieved={condition[4]:.10f}, "
                f"center=({condition[5]:.4f}, "
                f"{condition[6]:.4f}), "
                f"radius={condition[7]:.4f}"
            )
        )

    lines.append("")
    lines.append("Dataset-level summary:")

    for row in rows:
        lines.append(
            (
                f"target={row['target_occlusion_ratio']:.1f}, "
                f"n={row['sample_count']}, "
                f"achieved_mean={row['achieved_mean']:.10f}, "
                f"error_mean={row['absolute_error_mean']:.10f}, "
                f"error_max={row['absolute_error_max']:.10f}, "
                f"radius_mean={row['radius_mean_pixels']:.4f}, "
                f"visible_area_mean="
                f"{row['visible_area_fraction_mean']:.6f}"
            )
        )

    lines.append("")
    lines.append(f"PNG figure: {PNG_PATH}")
    lines.append(f"PDF figure: {PDF_PATH}")
    lines.append(f"CSV table: {CSV_PATH}")

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """
    Run visualization and summary generation.
    """

    print("=" * 78)
    print("OCCLUSION DATASET V2 — FIGURE AND TABLE GENERATION")
    print("=" * 78)

    print("Source:", H5_PATH)

    if not H5_PATH.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {H5_PATH}"
        )

    configure_matplotlib()

    with h5py.File(
        H5_PATH,
        "r",
    ) as h5:
        intensity_ds = h5["intensity"]
        mask_ds = h5["visible_mask"]

        labels = h5["labels"][:]
        conditions = h5["conditions"][:]
        base_indices = h5["base_indices"][:]

        (
            selected_base_index,
            sample_indices,
        ) = find_representative_group(
            labels=labels,
            conditions=conditions,
            base_indices=base_indices,
        )

        print(
            "Selected base index:",
            selected_base_index,
        )

        print(
            "Selected sample indices:",
            sample_indices.tolist(),
        )

        rows = calculate_level_statistics(
            conditions=conditions,
            masks=mask_ds,
        )

        save_summary_csv(
            rows=rows,
        )

        create_qc_figure(
            intensity_ds=intensity_ds,
            mask_ds=mask_ds,
            conditions=conditions,
            sample_indices=sample_indices,
        )

        save_report(
            selected_base_index=selected_base_index,
            sample_indices=sample_indices,
            conditions=conditions,
            rows=rows,
        )

    print("")
    print("=" * 78)
    print("FIGURE AND TABLE GENERATION COMPLETE")
    print("=" * 78)

    print("PNG:", PNG_PATH)
    print("PDF:", PDF_PATH)
    print("CSV:", CSV_PATH)
    print("Report:", REPORT_PATH)


if __name__ == "__main__":
    main()