"""
Locate the native Windows crash occurring in plot_failure_cases.py.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

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


def stage(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    stage("=" * 78)
    stage("DIAGNOSE FAILURE-CASE NATIVE CRASH")
    stage("=" * 78)

    stage("[1] Python and package information")

    stage(f"Python: {sys.version}")
    stage(f"NumPy: {np.__version__}")
    stage(f"h5py: {h5py.__version__}")

    stage("[2] Import receiver-noise module")

    from src.physics.receiver_noise import (
        add_deterministic_awgn,
    )

    stage("PASS receiver-noise import")

    stage("[3] Import polar-sampling module")

    from src.algorithms.polar_sampling import (
        extract_polar_profile,
        normalize_angular_profile,
    )

    stage("PASS polar-sampling import")

    stage("[4] Import harmonic-fitting module")

    from src.algorithms.harmonic_fit import (
        recognize_harmonic_state,
    )

    stage("PASS harmonic-fitting import")

    stage("[5] Read one MC-VWLS prediction row")

    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(
            f"Prediction CSV does not exist: {PREDICTION_PATH}"
        )

    selected_row = None

    with PREDICTION_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            if row["method"] == "A3_MC_VWLS":
                selected_row = row
                break

    if selected_row is None:
        raise RuntimeError(
            "No A3_MC_VWLS prediction row was found."
        )

    sample_index = int(
        round(
            float(
                selected_row["sample_index"]
            )
        )
    )

    snr_db = float(
        selected_row["target_snr_db"]
    )

    expected_order = int(
        round(
            float(
                selected_row["predicted_order"]
            )
        )
    )

    expected_phase = int(
        round(
            float(
                selected_row["predicted_phase_bin"]
            )
        )
    )

    stage(
        f"PASS prediction row: sample_index={sample_index}, "
        f"SNR={snr_db:.1f}, "
        f"expected=({expected_order}, {expected_phase})"
    )

    stage("[6] Open HDF5 file")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {DATASET_PATH}"
        )

    with h5py.File(
        DATASET_PATH,
        "r",
    ) as h5_file:
        stage("PASS HDF5 open")

        stage(
            "HDF5 keys: "
            + ", ".join(
                sorted(
                    h5_file.keys()
                )
            )
        )

        stage("[7] Read conditions shape without loading data")

        conditions_dataset = h5_file[
            "conditions"
        ]

        stage(
            f"PASS conditions metadata: "
            f"shape={conditions_dataset.shape}, "
            f"dtype={conditions_dataset.dtype}"
        )

        stage("[8] Read one conditions row")

        condition = np.asarray(
            conditions_dataset[
                sample_index
            ],
            dtype=np.float64,
        )

        stage(
            f"PASS conditions row: {condition.tolist()}"
        )

        stage("[9] Read one intensity sample")

        intensity = np.asarray(
            h5_file[
                "intensity"
            ][
                sample_index
            ],
            dtype=np.float32,
        )

        stage(
            f"PASS intensity: "
            f"shape={intensity.shape}, "
            f"dtype={intensity.dtype}, "
            f"minimum={float(np.min(intensity)):.8e}, "
            f"maximum={float(np.max(intensity)):.8e}"
        )

        stage("[10] Read one visible-mask sample")

        visible_mask = np.asarray(
            h5_file[
                "visible_mask"
            ][
                sample_index
            ],
            dtype=np.float32,
        )

        stage(
            f"PASS visible mask: "
            f"shape={visible_mask.shape}, "
            f"dtype={visible_mask.dtype}, "
            f"minimum={float(np.min(visible_mask)):.8f}, "
            f"maximum={float(np.max(visible_mask)):.8f}"
        )

    stage("[11] Generate deterministic noisy observation")

    observation = add_deterministic_awgn(
        clean_intensity=intensity,
        sample_index=sample_index,
        snr_db=snr_db,
    )

    stage(
        f"PASS receiver noise: "
        f"shape={observation.intensity.shape}, "
        f"minimum={float(np.min(observation.intensity)):.8e}, "
        f"maximum={float(np.max(observation.intensity)):.8e}"
    )

    stage("[12] Execute polar sampling")

    polar = extract_polar_profile(
        intensity=observation.intensity,
        visible_mask=visible_mask,
        angular_samples=180,
        radial_samples=64,
        visibility_threshold=0.0,
    )

    stage(
        f"PASS polar sampling: "
        f"polar_intensity_shape={polar.polar_intensity.shape}, "
        f"profile_shape={polar.angular_profile.shape}"
    )

    stage("[13] Normalize angular profile")

    normalized_profile = normalize_angular_profile(
        angular_profile=polar.angular_profile,
        valid_angles=polar.valid_angles,
        remove_mean=True,
        unit_norm=False,
    )

    stage(
        f"PASS profile normalization: "
        f"minimum={float(np.min(normalized_profile)):.8e}, "
        f"maximum={float(np.max(normalized_profile)):.8e}"
    )

    stage("[14] Execute harmonic recognition")

    recognition = recognize_harmonic_state(
        theta=polar.theta,
        angular_profile=normalized_profile,
        angular_visibility=polar.angular_visibility,
        valid_angles=polar.valid_angles,
        candidate_orders=(1, 2, 3, 4),
        phase_bins=8,
        regularization=0.0,
        weight_power=0.5,
    )

    stage(
        f"PASS harmonic recognition: "
        f"predicted=({recognition.predicted_order}, "
        f"{recognition.predicted_phase_bin}), "
        f"confidence={recognition.confidence:.8f}"
    )

    if recognition.predicted_order != expected_order:
        raise ValueError(
            "Reconstructed order does not match CSV: "
            f"expected={expected_order}, "
            f"actual={recognition.predicted_order}"
        )

    if recognition.predicted_phase_bin != expected_phase:
        raise ValueError(
            "Reconstructed phase does not match CSV: "
            f"expected={expected_phase}, "
            f"actual={recognition.predicted_phase_bin}"
        )

    stage("")
    stage("=" * 78)
    stage("ALL DIAGNOSTIC STAGES PASSED")
    stage("=" * 78)


if __name__ == "__main__":
    main()