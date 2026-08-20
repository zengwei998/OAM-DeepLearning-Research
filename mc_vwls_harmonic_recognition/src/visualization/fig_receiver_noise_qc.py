"""
Generate receiver-noise QC figure and statistics.

Outputs:
    results/figures/fig_receiver_noise_qc.png
    results/figures/fig_receiver_noise_qc.pdf
    results/csv/receiver_noise_qc.csv
    results/validation/receiver_noise_qc_report.txt
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.physics.receiver_noise import (
    SUPPORTED_SNR_DB,
    add_deterministic_awgn,
)


ROOT = Path(__file__).resolve().parents[2]

H5_PATH = (
    ROOT
    / "data"
    / "generated"
    / "occlusion_clean_v2.h5"
)

FIGURE_DIR = ROOT / "results" / "figures"
CSV_DIR = ROOT / "results" / "csv"
VALIDATION_DIR = ROOT / "results" / "validation"

PNG_PATH = FIGURE_DIR / "fig_receiver_noise_qc.png"
PDF_PATH = FIGURE_DIR / "fig_receiver_noise_qc.pdf"
CSV_PATH = CSV_DIR / "receiver_noise_qc.csv"
REPORT_PATH = VALIDATION_DIR / "receiver_noise_qc_report.txt"

# Representative sample used previously in the occlusion QC figure:
# target energy occlusion approximately 0.2.
SELECTED_SAMPLE_INDEX = 21702

# Dataset-wide QC subset.
AUDIT_SAMPLE_COUNT = 512
AUDIT_SEED = 20260804

DISPLAY_PERCENTILE = 99.8
LOG_EPSILON = 1.0e-12


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "axes.linewidth": 0.8,
        }
    )


def calculate_statistics(
    intensity_ds: h5py.Dataset,
) -> List[Dict[str, float]]:
    """
    Calculate measured SNR and clipping statistics on a fixed subset.
    """

    rng = np.random.default_rng(AUDIT_SEED)

    selected_indices = rng.choice(
        len(intensity_ds),
        size=min(AUDIT_SAMPLE_COUNT, len(intensity_ds)),
        replace=False,
    )

    rows: List[Dict[str, float]] = []

    for snr_db in SUPPORTED_SNR_DB:
        preclip_values = []
        postclip_values = []
        clipped_fractions = []
        signal_powers = []
        preclip_noise_powers = []
        postclip_error_powers = []

        for sample_index in selected_indices:
            clean = intensity_ds[int(sample_index)]

            observation = add_deterministic_awgn(
                clean_intensity=clean,
                sample_index=int(sample_index),
                snr_db=float(snr_db),
            )

            preclip_values.append(
                observation.measured_snr_preclip_db
            )

            postclip_values.append(
                observation.measured_snr_postclip_db
            )

            clipped_fractions.append(
                observation.clipped_pixel_fraction
            )

            signal_powers.append(
                observation.signal_power
            )

            preclip_noise_powers.append(
                observation.preclip_noise_power
            )

            postclip_error_powers.append(
                observation.postclip_error_power
            )

        preclip_array = np.asarray(preclip_values)
        postclip_array = np.asarray(postclip_values)
        clipped_array = np.asarray(clipped_fractions)

        rows.append(
            {
                "target_snr_db": float(snr_db),
                "sample_count": int(len(selected_indices)),
                "preclip_snr_mean_db": float(
                    np.mean(preclip_array)
                ),
                "preclip_snr_std_db": float(
                    np.std(preclip_array)
                ),
                "preclip_snr_min_db": float(
                    np.min(preclip_array)
                ),
                "preclip_snr_max_db": float(
                    np.max(preclip_array)
                ),
                "postclip_snr_mean_db": float(
                    np.mean(postclip_array)
                ),
                "postclip_snr_std_db": float(
                    np.std(postclip_array)
                ),
                "postclip_snr_min_db": float(
                    np.min(postclip_array)
                ),
                "postclip_snr_max_db": float(
                    np.max(postclip_array)
                ),
                "postclip_snr_shift_mean_db": float(
                    np.mean(postclip_array - preclip_array)
                ),
                "clipped_fraction_mean": float(
                    np.mean(clipped_array)
                ),
                "clipped_fraction_std": float(
                    np.std(clipped_array)
                ),
                "clipped_fraction_min": float(
                    np.min(clipped_array)
                ),
                "clipped_fraction_max": float(
                    np.max(clipped_array)
                ),
                "signal_power_mean": float(
                    np.mean(signal_powers)
                ),
                "preclip_noise_power_mean": float(
                    np.mean(preclip_noise_powers)
                ),
                "postclip_error_power_mean": float(
                    np.mean(postclip_error_powers)
                ),
            }
        )

    return rows


def save_csv(
    rows: List[Dict[str, float]],
) -> None:
    CSV_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(rows[0].keys())

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


def create_figure(
    clean: np.ndarray,
) -> None:
    """
    Display clean image and five noisy observations.
    """

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    observations = [
        add_deterministic_awgn(
            clean_intensity=clean,
            sample_index=SELECTED_SAMPLE_INDEX,
            snr_db=float(snr_db),
        )
        for snr_db in SUPPORTED_SNR_DB
    ]

    images = [
        np.asarray(clean, dtype=np.float64)
    ] + [
        np.asarray(
            observation.intensity,
            dtype=np.float64,
        )
        for observation in observations
    ]

    all_positive = np.concatenate(
        [
            image[image > 0]
            for image in images
            if np.any(image > 0)
        ]
    )

    display_upper = float(
        np.percentile(
            all_positive,
            DISPLAY_PERCENTILE,
        )
    )

    log_images = [
        np.log10(
            image / display_upper
            + LOG_EPSILON
        )
        for image in images
    ]

    figure, axes = plt.subplots(
        nrows=1,
        ncols=6,
        figsize=(13.2, 2.8),
        constrained_layout=True,
    )

    titles = ["Clean"]

    for observation in observations:
        titles.append(
            (
                f"{observation.target_snr_db:.0f} dB\n"
                f"post={observation.measured_snr_postclip_db:.2f} dB"
            )
        )

    artist = None

    for column, image in enumerate(log_images):
        artist = axes[column].imshow(
            image,
            origin="lower",
            cmap="inferno",
            vmin=-6.0,
            vmax=0.0,
            interpolation="nearest",
        )

        axes[column].set_title(
            titles[column]
        )

        axes[column].set_xticks([])
        axes[column].set_yticks([])

    if artist is not None:
        colorbar = figure.colorbar(
            artist,
            ax=axes,
            location="right",
            shrink=0.83,
            pad=0.012,
        )

        colorbar.set_label(
            r"$\log_{10}(I/I_{99.8\%})$"
        )

    figure.suptitle(
        (
            "Deterministic AWGN receiver observations "
            "with nonnegative clipping\n"
            f"clean sample index = {SELECTED_SAMPLE_INDEX}"
        ),
        y=1.07,
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
    rows: List[Dict[str, float]],
) -> None:
    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        "Receiver-noise QC report",
        f"Source: {H5_PATH}",
        f"Representative sample: {SELECTED_SAMPLE_INDEX}",
        f"Audit sample count: {AUDIT_SAMPLE_COUNT}",
        "",
        (
            "SNR definition: image-domain signal-to-noise ratio "
            "before nonnegative clipping."
        ),
        "",
    ]

    for row in rows:
        lines.append(
            (
                f"target={row['target_snr_db']:.1f} dB, "
                f"preclip_mean={row['preclip_snr_mean_db']:.10f} dB, "
                f"postclip_mean={row['postclip_snr_mean_db']:.6f} dB, "
                f"shift_mean={row['postclip_snr_shift_mean_db']:.6f} dB, "
                f"clipped_fraction_mean="
                f"{row['clipped_fraction_mean']:.6f}"
            )
        )

    lines.extend(
        [
            "",
            f"PNG: {PNG_PATH}",
            f"PDF: {PDF_PATH}",
            f"CSV: {CSV_PATH}",
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 78)
    print("RECEIVER-NOISE QC FIGURE AND TABLE")
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

        clean = np.asarray(
            intensity_ds[SELECTED_SAMPLE_INDEX],
            dtype=np.float64,
        )

        rows = calculate_statistics(
            intensity_ds=intensity_ds,
        )

    save_csv(rows)
    create_figure(clean)
    save_report(rows)

    print("")
    print("=" * 78)
    print("RECEIVER-NOISE QC COMPLETE")
    print("=" * 78)
    print("PNG:", PNG_PATH)
    print("PDF:", PDF_PATH)
    print("CSV:", CSV_PATH)
    print("Report:", REPORT_PATH)


if __name__ == "__main__":
    main()