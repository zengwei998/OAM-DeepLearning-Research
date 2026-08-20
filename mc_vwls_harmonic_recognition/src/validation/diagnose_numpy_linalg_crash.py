"""
Verify a BLAS-free construction of the 3x3 weighted normal equations.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import h5py
import numpy as np

from src.algorithms.harmonic_fit import build_design_matrix
from src.algorithms.polar_sampling import (
    extract_polar_profile,
    normalize_angular_profile,
)
from src.physics.receiver_noise import add_deterministic_awgn


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

EPSILON = 1.0e-12


def stage(message: str) -> None:
    print(message, flush=True)


def load_first_mc_vwls_row() -> dict[str, str]:
    with PREDICTION_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            if row["method"] == "A3_MC_VWLS":
                return row

    raise RuntimeError(
        "No A3_MC_VWLS prediction row was found."
    )


def build_normal_equations_without_matmul(
    weighted_design: np.ndarray,
    weighted_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.asarray(
        weighted_design,
        dtype=np.float64,
    )

    target = np.asarray(
        weighted_target,
        dtype=np.float64,
    )

    if design.ndim != 2:
        raise ValueError(
            "weighted_design must be two-dimensional."
        )

    if design.shape[1] != 3:
        raise ValueError(
            "weighted_design must contain exactly three columns."
        )

    if target.shape != (design.shape[0],):
        raise ValueError(
            "weighted_target shape does not match weighted_design."
        )

    column_0 = design[:, 0]
    column_1 = design[:, 1]
    column_2 = design[:, 2]

    normal_matrix = np.empty(
        (3, 3),
        dtype=np.float64,
    )

    normal_matrix[0, 0] = float(
        np.sum(column_0 * column_0, dtype=np.float64)
    )
    normal_matrix[0, 1] = float(
        np.sum(column_0 * column_1, dtype=np.float64)
    )
    normal_matrix[0, 2] = float(
        np.sum(column_0 * column_2, dtype=np.float64)
    )
    normal_matrix[1, 0] = normal_matrix[0, 1]
    normal_matrix[1, 1] = float(
        np.sum(column_1 * column_1, dtype=np.float64)
    )
    normal_matrix[1, 2] = float(
        np.sum(column_1 * column_2, dtype=np.float64)
    )
    normal_matrix[2, 0] = normal_matrix[0, 2]
    normal_matrix[2, 1] = normal_matrix[1, 2]
    normal_matrix[2, 2] = float(
        np.sum(column_2 * column_2, dtype=np.float64)
    )

    right_hand_side = np.asarray(
        [
            np.sum(
                column_0 * target,
                dtype=np.float64,
            ),
            np.sum(
                column_1 * target,
                dtype=np.float64,
            ),
            np.sum(
                column_2 * target,
                dtype=np.float64,
            ),
        ],
        dtype=np.float64,
    )

    return (
        normal_matrix,
        right_hand_side,
    )


def predict_without_matmul(
    design: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(
        design,
        dtype=np.float64,
    )

    coefficient_array = np.asarray(
        coefficients,
        dtype=np.float64,
    )

    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError(
            "design must have shape (N, 3)."
        )

    if coefficient_array.shape != (3,):
        raise ValueError(
            "coefficients must have shape (3,)."
        )

    return (
        matrix[:, 0] * coefficient_array[0]
        + matrix[:, 1] * coefficient_array[1]
        + matrix[:, 2] * coefficient_array[2]
    )


def main() -> None:
    stage("=" * 78)
    stage("VERIFY BLAS-FREE HARMONIC NORMAL EQUATIONS")
    stage("=" * 78)

    stage(f"Python: {sys.version}")
    stage(f"NumPy: {np.__version__}")
    stage(f"h5py: {h5py.__version__}")

    stage("[1] Load one prediction row")

    row = load_first_mc_vwls_row()

    sample_index = int(
        round(
            float(
                row["sample_index"]
            )
        )
    )

    snr_db = float(
        row["target_snr_db"]
    )

    expected_order = int(
        round(
            float(
                row["predicted_order"]
            )
        )
    )

    expected_phase = int(
        round(
            float(
                row["predicted_phase_bin"]
            )
        )
    )

    stage(
        f"PASS [1]: sample_index={sample_index}, "
        f"SNR={snr_db:.1f}, "
        f"expected=({expected_order}, {expected_phase})"
    )

    stage("[2] Read one HDF5 observation")

    with h5py.File(
        DATASET_PATH,
        "r",
    ) as h5_file:
        clean_intensity = np.asarray(
            h5_file["intensity"][sample_index],
            dtype=np.float32,
        ).copy()

        visible_mask = np.asarray(
            h5_file["visible_mask"][sample_index],
            dtype=np.float32,
        ).copy()

    stage("PASS [2]")

    stage("[3] Generate deterministic receiver noise")

    observation = add_deterministic_awgn(
        clean_intensity=clean_intensity,
        sample_index=sample_index,
        snr_db=snr_db,
    )

    stage("PASS [3]")

    stage("[4] Extract and normalize polar profile")

    polar = extract_polar_profile(
        intensity=observation.intensity,
        visible_mask=visible_mask,
        angular_samples=180,
        radial_samples=64,
        visibility_threshold=0.0,
    )

    profile = normalize_angular_profile(
        angular_profile=polar.angular_profile,
        valid_angles=polar.valid_angles,
        remove_mean=True,
        unit_norm=False,
    )

    stage("PASS [4]")

    theta = np.asarray(
        polar.theta,
        dtype=np.float64,
    )

    visibility = np.asarray(
        polar.angular_visibility,
        dtype=np.float64,
    )

    valid = np.asarray(
        polar.valid_angles,
        dtype=bool,
    )

    profile = np.asarray(
        profile,
        dtype=np.float64,
    )

    theta_valid = theta[valid]
    profile_valid = profile[valid]
    visibility_valid = visibility[valid]

    weights = np.maximum(
        visibility_valid,
        EPSILON,
    ) ** 0.5

    sqrt_weights = np.sqrt(
        weights
    )

    predicted_candidates = []

    for oam_order in (1, 2, 3, 4):
        stage(
            f"[5.{oam_order}] Fit candidate order {oam_order}"
        )

        design = build_design_matrix(
            theta=theta_valid,
            harmonic_order=2 * oam_order,
        )

        weighted_design = (
            design
            * sqrt_weights[:, None]
        )

        weighted_target = (
            profile_valid
            * sqrt_weights
        )

        stage(
            f"[5.{oam_order}.1] Build normal equations without @"
        )

        (
            normal_matrix,
            right_hand_side,
        ) = build_normal_equations_without_matmul(
            weighted_design=weighted_design,
            weighted_target=weighted_target,
        )

        stage(
            f"PASS [5.{oam_order}.1]: "
            f"normal_matrix={normal_matrix.tolist()}"
        )

        stage(
            f"[5.{oam_order}.2] Solve 3x3 system"
        )

        def solve_3x3_without_lapack(
                matrix: np.ndarray,
                vector: np.ndarray,
        ) -> np.ndarray:
            augmented = [
                [
                    float(matrix[row_index, column_index])
                    for column_index in range(3)
                ]
                + [
                    float(vector[row_index])
                ]
                for row_index in range(3)
            ]

            for pivot_index in range(3):
                pivot_row = max(
                    range(pivot_index, 3),
                    key=lambda row_index: abs(
                        augmented[row_index][pivot_index]
                    ),
                )

                if abs(
                        augmented[pivot_row][pivot_index]
                ) <= 1.0e-15:
                    raise np.linalg.LinAlgError(
                        "Singular 3x3 normal matrix."
                    )

                if pivot_row != pivot_index:
                    augmented[pivot_index], augmented[pivot_row] = (
                        augmented[pivot_row],
                        augmented[pivot_index],
                    )

                pivot_value = augmented[pivot_index][pivot_index]

                for column_index in range(
                        pivot_index,
                        4,
                ):
                    augmented[pivot_index][column_index] /= pivot_value

                for row_index in range(3):
                    if row_index == pivot_index:
                        continue

                    factor = augmented[row_index][pivot_index]

                    for column_index in range(
                            pivot_index,
                            4,
                    ):
                        augmented[row_index][column_index] -= (
                                factor
                                * augmented[pivot_index][column_index]
                        )

            return np.asarray(
                [
                    augmented[row_index][3]
                    for row_index in range(3)
                ],
                dtype=np.float64,
            )

        coefficients = solve_3x3_without_lapack(
            normal_matrix,
            right_hand_side,
        )

        stage(
            f"PASS [5.{oam_order}.2]: "
            f"coefficients={coefficients.tolist()}"
        )

        stage(
            f"[5.{oam_order}.3] Predict without @"
        )

        prediction = predict_without_matmul(
            design=design,
            coefficients=coefficients,
        )

        residual = (
            profile_valid
            - prediction
        )

        weight_sum = float(
            np.sum(
                weights,
                dtype=np.float64,
            )
        )

        residual_mse = float(
            np.sum(
                weights * residual**2,
                dtype=np.float64,
            )
            / max(
                weight_sum,
                EPSILON,
            )
        )

        cosine_coefficient = float(
            coefficients[1]
        )

        sine_coefficient = float(
            coefficients[2]
        )

        amplitude = float(
            np.hypot(
                cosine_coefficient,
                sine_coefficient,
            )
        )

        phase_rad = float(
            np.mod(
                np.arctan2(
                    sine_coefficient,
                    cosine_coefficient,
                ),
                2.0 * np.pi,
            )
        )

        score = float(
            amplitude
            / np.sqrt(
                residual_mse
                + EPSILON
            )
        )

        predicted_candidates.append(
            {
                "order": oam_order,
                "phase_rad": phase_rad,
                "score": score,
            }
        )

        stage(
            f"PASS [5.{oam_order}.3]: "
            f"score={score:.8f}, "
            f"phase={phase_rad:.8f}"
        )

    ranked = sorted(
        predicted_candidates,
        key=lambda candidate: float(
            candidate["score"]
        ),
        reverse=True,
    )

    best = ranked[0]

    phase_width = (
        2.0 * np.pi
        / 8
    )

    predicted_phase = int(
        np.floor(
            float(
                best["phase_rad"]
            )
            / phase_width
            + 0.5
        )
    ) % 8

    predicted_order = int(
        best["order"]
    )

    stage("")
    stage(
        f"Reconstructed prediction: "
        f"({predicted_order}, {predicted_phase})"
    )

    if predicted_order != expected_order:
        raise ValueError(
            "Order mismatch: "
            f"expected={expected_order}, "
            f"actual={predicted_order}"
        )

    if predicted_phase != expected_phase:
        raise ValueError(
            "Phase mismatch: "
            f"expected={expected_phase}, "
            f"actual={predicted_phase}"
        )

    stage("")
    stage("=" * 78)
    stage("BLAS-FREE HARMONIC FIT PASSED")
    stage("=" * 78)


if __name__ == "__main__":
    main()