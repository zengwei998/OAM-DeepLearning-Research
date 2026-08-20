"""
Evaluate mask-constrained and visibility-weighted harmonic-recognition
ablation variants on the complete frozen test set.

Ablation variants:
    A0_DAF:
        Direct angular Fourier recognition.
        No mask-support normalization.
        No angular-visibility weighting.

    A1_MASK_ULS:
        Mask-support-normalized angular profile.
        Unweighted least-squares harmonic fitting.

    A2_RAW_VWLS:
        Raw zero-filled angular profile.
        Angular-visibility-weighted least-squares harmonic fitting.
        No mask-support normalization.

    A3_MC_VWLS:
        Mask-support-normalized angular profile.
        Angular-visibility-weighted least-squares harmonic fitting.

Frozen MC-VWLS parameters selected using the validation set:
    weight_power = 0.5
    visibility_threshold = 0.0
    regularization = 0.0

Outputs:
    results/csv/mc_vwls_ablation_full_test_predictions.csv
    results/csv/mc_vwls_ablation_summary.csv
    results/validation/mc_vwls_ablation_full_test_report.txt
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

from src.algorithms.daf import (
    recognize_daf_state,
)
from src.algorithms.harmonic_fit import (
    recognize_harmonic_state,
)
from src.algorithms.polar_sampling import (
    extract_polar_profile,
    normalize_angular_profile,
)
from src.algorithms.uls import (
    recognize_uls_state,
)
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

SPLIT_PATH = (
    ROOT
    / "data"
    / "manifest"
    / "sample_split_v1.npz"
)

PREDICTION_CSV_PATH = (
    ROOT
    / "results"
    / "csv"
    / "mc_vwls_ablation_full_test_predictions.csv"
)

SUMMARY_CSV_PATH = (
    ROOT
    / "results"
    / "csv"
    / "mc_vwls_ablation_summary.csv"
)

REPORT_PATH = (
    ROOT
    / "results"
    / "validation"
    / "mc_vwls_ablation_full_test_report.txt"
)

EXPECTED_TEST_SAMPLES = 6720

ANGULAR_SAMPLES = 180
RADIAL_SAMPLES = 64
PHASE_BINS = 8

CANDIDATE_ORDERS = (
    1,
    2,
    3,
    4,
)

# Frozen using validation-set tuning only.
VISIBILITY_THRESHOLD = 0.0
REGULARIZATION = 0.0
WEIGHT_POWER = 0.5

EPSILON = 1.0e-12

METHOD_NAMES = (
    "A0_DAF",
    "A1_MASK_ULS",
    "A2_RAW_VWLS",
    "A3_MC_VWLS",
)


def load_test_indices() -> np.ndarray:
    """
    Load complete frozen test-set indices.

    Split coding:
        0 = training
        1 = validation
        2 = test
    """

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Split file does not exist: {SPLIT_PATH}"
        )

    with np.load(
        SPLIT_PATH,
        allow_pickle=False,
    ) as split:
        if "sample_split_codes" not in split.files:
            raise KeyError(
                "sample_split_codes was not found. "
                f"Available keys: {sorted(split.files)}"
            )

        split_codes = np.asarray(
            split["sample_split_codes"],
            dtype=np.int64,
        )

    if split_codes.ndim != 1:
        raise ValueError(
            "sample_split_codes must be one-dimensional."
        )

    test_indices = np.flatnonzero(
        split_codes == 2
    ).astype(
        np.int64
    )

    if len(test_indices) != EXPECTED_TEST_SAMPLES:
        unique_codes, counts = np.unique(
            split_codes,
            return_counts=True,
        )

        code_counts = dict(
            zip(
                unique_codes.tolist(),
                counts.tolist(),
            )
        )

        raise ValueError(
            "Unexpected test-set size. "
            f"Expected {EXPECTED_TEST_SAMPLES}, "
            f"found {len(test_indices)}. "
            f"Split-code counts: {code_counts}"
        )

    return np.sort(
        test_indices
    )


def circular_phase_bin_error(
    true_bin: int,
    predicted_bin: int,
) -> int:
    """
    Calculate circular phase-bin distance.
    """

    direct_error = abs(
        int(true_bin)
        - int(predicted_bin)
    )

    return int(
        min(
            direct_error,
            PHASE_BINS - direct_error,
        )
    )


def build_raw_angular_profile(
    polar_intensity: np.ndarray,
    radius: np.ndarray,
) -> np.ndarray:
    """
    Construct a zero-filled angular profile without mask normalization.

    Polar samples inside blocked regions remain zero. The radial
    integration uses the physical polar-area factor r.
    """

    intensity = np.asarray(
        polar_intensity,
        dtype=np.float64,
    )

    radius_array = np.asarray(
        radius,
        dtype=np.float64,
    )

    if intensity.ndim != 2:
        raise ValueError(
            "polar_intensity must be two-dimensional."
        )

    if radius_array.ndim != 1:
        raise ValueError(
            "radius must be one-dimensional."
        )

    if intensity.shape[0] != len(radius_array):
        raise ValueError(
            "Radius length does not match the polar radial dimension."
        )

    if not np.all(
        np.isfinite(intensity)
    ):
        raise ValueError(
            "polar_intensity contains NaN or Inf."
        )

    if not np.all(
        np.isfinite(radius_array)
    ):
        raise ValueError(
            "radius contains NaN or Inf."
        )

    radial_weights = radius_array[
        :,
        None,
    ]

    full_radial_support = float(
        np.sum(
            radial_weights,
            dtype=np.float64,
        )
    )

    if full_radial_support <= EPSILON:
        raise ValueError(
            "Full radial support is zero."
        )

    raw_profile = np.sum(
        intensity
        * radial_weights,
        axis=0,
        dtype=np.float64,
    ) / full_radial_support

    if not np.all(
        np.isfinite(raw_profile)
    ):
        raise ValueError(
            "Raw angular profile contains NaN or Inf."
        )

    return np.asarray(
        raw_profile,
        dtype=np.float64,
    )


def make_prediction_row(
    *,
    method: str,
    sample_index: int,
    snr_db: float,
    target_occlusion: float,
    true_label: int,
    predicted_label: int,
    predicted_order: int,
    predicted_phase_bin: int,
    predicted_phase_rad: float,
    confidence: float,
    harmonic_margin: float,
    best_score: float,
    second_best_score: float,
    mean_visibility: float,
    minimum_visibility: float,
    valid_fraction: float,
) -> Dict[str, object]:
    """
    Construct one prediction record.
    """

    true_order = (
        true_label
        // PHASE_BINS
        + 1
    )

    true_phase_bin = (
        true_label
        % PHASE_BINS
    )

    return {
        "method": method,
        "sample_index": int(
            sample_index
        ),
        "target_snr_db": float(
            snr_db
        ),
        "target_occlusion": float(
            target_occlusion
        ),
        "true_label": int(
            true_label
        ),
        "predicted_label": int(
            predicted_label
        ),
        "true_order": int(
            true_order
        ),
        "predicted_order": int(
            predicted_order
        ),
        "true_phase_bin": int(
            true_phase_bin
        ),
        "predicted_phase_bin": int(
            predicted_phase_bin
        ),
        "predicted_phase_rad": float(
            predicted_phase_rad
        ),
        "phase_bin_error": circular_phase_bin_error(
            true_bin=true_phase_bin,
            predicted_bin=predicted_phase_bin,
        ),
        "label_correct": int(
            predicted_label
            == true_label
        ),
        "order_correct": int(
            predicted_order
            == true_order
        ),
        "phase_correct": int(
            predicted_phase_bin
            == true_phase_bin
        ),
        "confidence": float(
            confidence
        ),
        "harmonic_margin": float(
            harmonic_margin
        ),
        "best_score": float(
            best_score
        ),
        "second_best_score": float(
            second_best_score
        ),
        "mean_visibility": float(
            mean_visibility
        ),
        "minimum_visibility": float(
            minimum_visibility
        ),
        "valid_fraction": float(
            valid_fraction
        ),
    }


def evaluate_observation(
    *,
    sample_index: int,
    snr_db: float,
    clean: np.ndarray,
    visible_mask: np.ndarray,
    true_label: int,
    target_occlusion: float,
) -> List[Dict[str, object]]:
    """
    Evaluate all four ablation variants on one noisy observation.
    """

    observation = add_deterministic_awgn(
        clean_intensity=clean,
        sample_index=sample_index,
        snr_db=snr_db,
    )

    polar = extract_polar_profile(
        intensity=observation.intensity,
        visible_mask=visible_mask,
        angular_samples=ANGULAR_SAMPLES,
        radial_samples=RADIAL_SAMPLES,
        visibility_threshold=VISIBILITY_THRESHOLD,
    )

    theta = np.asarray(
        polar.theta,
        dtype=np.float64,
    )

    angular_visibility = np.asarray(
        polar.angular_visibility,
        dtype=np.float64,
    )

    valid_angles = np.asarray(
        polar.valid_angles,
        dtype=bool,
    )

    raw_profile = build_raw_angular_profile(
        polar_intensity=polar.polar_intensity,
        radius=polar.radius,
    )

    normalized_profile = normalize_angular_profile(
        angular_profile=polar.angular_profile,
        valid_angles=valid_angles,
        remove_mean=True,
        unit_norm=False,
    )

    # A0:
    # Raw zero-filled profile plus direct angular Fourier recognition.
    a0_result = recognize_daf_state(
        theta=theta,
        angular_profile=raw_profile,
        candidate_orders=CANDIDATE_ORDERS,
        phase_bins=PHASE_BINS,
    )

    # A1:
    # Mask-support-normalized profile plus unweighted least squares.
    a1_result = recognize_uls_state(
        theta=theta,
        angular_profile=normalized_profile,
        candidate_orders=CANDIDATE_ORDERS,
        phase_bins=PHASE_BINS,
    )

    # A2:
    # Raw zero-filled profile plus visibility-weighted least squares.
    a2_result = recognize_harmonic_state(
        theta=theta,
        angular_profile=raw_profile,
        angular_visibility=angular_visibility,
        valid_angles=valid_angles,
        candidate_orders=CANDIDATE_ORDERS,
        phase_bins=PHASE_BINS,
        regularization=REGULARIZATION,
        weight_power=WEIGHT_POWER,
    )

    # A3:
    # Mask-support normalization plus visibility-weighted least squares.
    a3_result = recognize_harmonic_state(
        theta=theta,
        angular_profile=normalized_profile,
        angular_visibility=angular_visibility,
        valid_angles=valid_angles,
        candidate_orders=CANDIDATE_ORDERS,
        phase_bins=PHASE_BINS,
        regularization=REGULARIZATION,
        weight_power=WEIGHT_POWER,
    )

    mean_visibility = float(
        np.mean(
            angular_visibility,
            dtype=np.float64,
        )
    )

    minimum_visibility = float(
        np.min(
            angular_visibility
        )
    )

    valid_fraction = float(
        np.mean(
            valid_angles,
            dtype=np.float64,
        )
    )

    numeric_values = np.asarray(
        [
            a0_result.predicted_phase_rad,
            a0_result.confidence,
            a0_result.best_score,
            a0_result.second_best_score,
            a1_result.predicted_phase_rad,
            a1_result.confidence,
            a1_result.best_score,
            a1_result.second_best_score,
            a2_result.predicted_phase_rad,
            a2_result.confidence,
            a2_result.best_score,
            a2_result.second_best_score,
            a3_result.predicted_phase_rad,
            a3_result.confidence,
            a3_result.best_score,
            a3_result.second_best_score,
            mean_visibility,
            minimum_visibility,
            valid_fraction,
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(numeric_values)
    ):
        raise FloatingPointError(
            "Ablation evaluation produced NaN or Inf. "
            f"sample_index={sample_index}, "
            f"snr_db={snr_db}"
        )

    return [
        make_prediction_row(
            method="A0_DAF",
            sample_index=sample_index,
            snr_db=snr_db,
            target_occlusion=target_occlusion,
            true_label=true_label,
            predicted_label=(
                a0_result.predicted_label
            ),
            predicted_order=(
                a0_result.predicted_order
            ),
            predicted_phase_bin=(
                a0_result.predicted_phase_bin
            ),
            predicted_phase_rad=(
                a0_result.predicted_phase_rad
            ),
            confidence=a0_result.confidence,
            harmonic_margin=(
                a0_result.harmonic_margin
            ),
            best_score=a0_result.best_score,
            second_best_score=(
                a0_result.second_best_score
            ),
            mean_visibility=mean_visibility,
            minimum_visibility=minimum_visibility,
            valid_fraction=valid_fraction,
        ),
        make_prediction_row(
            method="A1_MASK_ULS",
            sample_index=sample_index,
            snr_db=snr_db,
            target_occlusion=target_occlusion,
            true_label=true_label,
            predicted_label=(
                a1_result.predicted_label
            ),
            predicted_order=(
                a1_result.predicted_order
            ),
            predicted_phase_bin=(
                a1_result.predicted_phase_bin
            ),
            predicted_phase_rad=(
                a1_result.predicted_phase_rad
            ),
            confidence=a1_result.confidence,
            harmonic_margin=(
                a1_result.harmonic_margin
            ),
            best_score=a1_result.best_score,
            second_best_score=(
                a1_result.second_best_score
            ),
            mean_visibility=mean_visibility,
            minimum_visibility=minimum_visibility,
            valid_fraction=valid_fraction,
        ),
        make_prediction_row(
            method="A2_RAW_VWLS",
            sample_index=sample_index,
            snr_db=snr_db,
            target_occlusion=target_occlusion,
            true_label=true_label,
            predicted_label=(
                a2_result.predicted_label
            ),
            predicted_order=(
                a2_result.predicted_order
            ),
            predicted_phase_bin=(
                a2_result.predicted_phase_bin
            ),
            predicted_phase_rad=(
                a2_result.predicted_phase_rad
            ),
            confidence=a2_result.confidence,
            harmonic_margin=(
                a2_result.harmonic_margin
            ),
            best_score=a2_result.best_score,
            second_best_score=(
                a2_result.second_best_score
            ),
            mean_visibility=mean_visibility,
            minimum_visibility=minimum_visibility,
            valid_fraction=valid_fraction,
        ),
        make_prediction_row(
            method="A3_MC_VWLS",
            sample_index=sample_index,
            snr_db=snr_db,
            target_occlusion=target_occlusion,
            true_label=true_label,
            predicted_label=(
                a3_result.predicted_label
            ),
            predicted_order=(
                a3_result.predicted_order
            ),
            predicted_phase_bin=(
                a3_result.predicted_phase_bin
            ),
            predicted_phase_rad=(
                a3_result.predicted_phase_rad
            ),
            confidence=a3_result.confidence,
            harmonic_margin=(
                a3_result.harmonic_margin
            ),
            best_score=a3_result.best_score,
            second_best_score=(
                a3_result.second_best_score
            ),
            mean_visibility=mean_visibility,
            minimum_visibility=minimum_visibility,
            valid_fraction=valid_fraction,
        ),
    ]


def calculate_mean(
    rows: List[Dict[str, object]],
    key: str,
) -> float:
    """
    Calculate the mean of one numeric result field.
    """

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


def append_summary_row(
    summary_rows: List[Dict[str, object]],
    *,
    method: str,
    scope: str,
    rows: List[Dict[str, object]],
    target_snr_db: object = "",
    target_occlusion: object = "",
    true_order: object = "",
) -> None:
    """
    Append one aggregate summary row.
    """

    summary_rows.append(
        {
            "method": method,
            "scope": scope,
            "target_snr_db": target_snr_db,
            "target_occlusion": target_occlusion,
            "true_order": true_order,
            "sample_count": len(rows),
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
            "mean_visibility": calculate_mean(
                rows,
                "mean_visibility",
            ),
            "mean_valid_fraction": calculate_mean(
                rows,
                "valid_fraction",
            ),
        }
    )


def build_summary_rows(
    prediction_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """
    Build overall and condition-wise result summaries.
    """

    summary_rows: List[
        Dict[str, object]
    ] = []

    occlusion_levels = sorted(
        {
            float(
                row["target_occlusion"]
            )
            for row in prediction_rows
        }
    )

    for method in METHOD_NAMES:
        method_rows = [
            row
            for row in prediction_rows
            if row["method"] == method
        ]

        append_summary_row(
            summary_rows,
            method=method,
            scope="overall",
            rows=method_rows,
        )

        for snr_db in SUPPORTED_SNR_DB:
            snr_rows = [
                row
                for row in method_rows
                if float(
                    row["target_snr_db"]
                ) == float(
                    snr_db
                )
            ]

            append_summary_row(
                summary_rows,
                method=method,
                scope="snr",
                rows=snr_rows,
                target_snr_db=float(
                    snr_db
                ),
            )

        for occlusion in occlusion_levels:
            occlusion_rows = [
                row
                for row in method_rows
                if np.isclose(
                    float(
                        row["target_occlusion"]
                    ),
                    occlusion,
                    rtol=0.0,
                    atol=1.0e-8,
                )
            ]

            append_summary_row(
                summary_rows,
                method=method,
                scope="occlusion",
                rows=occlusion_rows,
                target_occlusion=occlusion,
            )

        for snr_db in SUPPORTED_SNR_DB:
            for occlusion in occlusion_levels:
                condition_rows = [
                    row
                    for row in method_rows
                    if (
                        float(
                            row["target_snr_db"]
                        )
                        == float(
                            snr_db
                        )
                        and np.isclose(
                            float(
                                row[
                                    "target_occlusion"
                                ]
                            ),
                            occlusion,
                            rtol=0.0,
                            atol=1.0e-8,
                        )
                    )
                ]

                append_summary_row(
                    summary_rows,
                    method=method,
                    scope="snr_occlusion",
                    rows=condition_rows,
                    target_snr_db=float(
                        snr_db
                    ),
                    target_occlusion=occlusion,
                )

        for order in CANDIDATE_ORDERS:
            order_rows = [
                row
                for row in method_rows
                if int(
                    row["true_order"]
                ) == int(
                    order
                )
            ]

            append_summary_row(
                summary_rows,
                method=method,
                scope="order",
                rows=order_rows,
                true_order=int(
                    order
                ),
            )

    return summary_rows


def save_csv(
    path: Path,
    rows: List[Dict[str, object]],
) -> None:
    """
    Save a list of dictionaries as CSV.
    """

    if not rows:
        raise ValueError(
            f"No rows were generated for {path}."
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
        newline="",
        encoding="utf-8-sig",
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
    scope: str,
    target_snr_db: object = None,
    target_occlusion: object = None,
    true_order: object = None,
) -> Dict[str, object]:
    """
    Locate one summary row.
    """

    for row in summary_rows:
        if row["method"] != method:
            continue

        if row["scope"] != scope:
            continue

        if target_snr_db is not None:
            if float(
                row["target_snr_db"]
            ) != float(
                target_snr_db
            ):
                continue

        if target_occlusion is not None:
            if not np.isclose(
                float(
                    row["target_occlusion"]
                ),
                float(
                    target_occlusion
                ),
                rtol=0.0,
                atol=1.0e-8,
            ):
                continue

        if true_order is not None:
            if int(
                row["true_order"]
            ) != int(
                true_order
            ):
                continue

        return row

    raise KeyError(
        "Summary row was not found: "
        f"method={method}, "
        f"scope={scope}, "
        f"target_snr_db={target_snr_db}, "
        f"target_occlusion={target_occlusion}, "
        f"true_order={true_order}"
    )


def build_report(
    summary_rows: List[Dict[str, object]],
    *,
    total_noisy_observations: int,
    total_predictions: int,
    elapsed_seconds: float,
) -> str:
    """
    Build the human-readable ablation report.
    """

    lines = [
        "MC-VWLS full-test ablation report",
        f"Dataset: {H5_PATH}",
        f"Split: {SPLIT_PATH}",
        f"Test clean samples: {EXPECTED_TEST_SAMPLES}",
        f"SNR levels: {SUPPORTED_SNR_DB}",
        (
            "Noisy observations: "
            f"{total_noisy_observations}"
        ),
        (
            "Total method predictions: "
            f"{total_predictions}"
        ),
        "",
        "[FROZEN PARAMETERS]",
        (
            "weight_power="
            f"{WEIGHT_POWER:.8f}"
        ),
        (
            "visibility_threshold="
            f"{VISIBILITY_THRESHOLD:.8f}"
        ),
        (
            "regularization="
            f"{REGULARIZATION:.10e}"
        ),
        "",
        "[ABLATION DEFINITIONS]",
        (
            "A0_DAF: raw zero-filled profile, "
            "no mask normalization, no visibility weighting"
        ),
        (
            "A1_MASK_ULS: mask-normalized profile, "
            "unweighted least squares"
        ),
        (
            "A2_RAW_VWLS: raw zero-filled profile, "
            "visibility-weighted least squares"
        ),
        (
            "A3_MC_VWLS: mask-normalized profile, "
            "visibility-weighted least squares"
        ),
        "",
        "[OVERALL]",
    ]

    for method in METHOD_NAMES:
        row = find_summary_row(
            summary_rows,
            method=method,
            scope="overall",
        )

        lines.append(
            (
                f"method={method}, "
                f"samples={row['sample_count']}, "
                f"label_accuracy="
                f"{float(row['label_accuracy']):.8f}, "
                f"order_accuracy="
                f"{float(row['order_accuracy']):.8f}, "
                f"phase_accuracy="
                f"{float(row['phase_accuracy']):.8f}, "
                f"mean_phase_bin_error="
                f"{float(row['mean_phase_bin_error']):.8f}, "
                f"mean_confidence="
                f"{float(row['mean_confidence']):.8f}"
            )
        )

    lines.extend(
        [
            "",
            "[LABEL ACCURACY BY SNR]",
        ]
    )

    for snr_db in SUPPORTED_SNR_DB:
        for method in METHOD_NAMES:
            row = find_summary_row(
                summary_rows,
                method=method,
                scope="snr",
                target_snr_db=snr_db,
            )

            lines.append(
                (
                    f"SNR={float(snr_db):.1f} dB, "
                    f"method={method}, "
                    f"label_accuracy="
                    f"{float(row['label_accuracy']):.8f}"
                )
            )

    occlusion_levels = (
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
    )

    lines.extend(
        [
            "",
            "[LABEL ACCURACY BY OCCLUSION]",
        ]
    )

    for occlusion in occlusion_levels:
        for method in METHOD_NAMES:
            row = find_summary_row(
                summary_rows,
                method=method,
                scope="occlusion",
                target_occlusion=occlusion,
            )

            lines.append(
                (
                    f"occlusion={occlusion:.1f}, "
                    f"method={method}, "
                    f"label_accuracy="
                    f"{float(row['label_accuracy']):.8f}"
                )
            )

    lines.extend(
        [
            "",
            "[LABEL ACCURACY BY OAM ORDER]",
        ]
    )

    for order in CANDIDATE_ORDERS:
        for method in METHOD_NAMES:
            row = find_summary_row(
                summary_rows,
                method=method,
                scope="order",
                true_order=order,
            )

            lines.append(
                (
                    f"order={order}, "
                    f"method={method}, "
                    f"label_accuracy="
                    f"{float(row['label_accuracy']):.8f}, "
                    f"order_accuracy="
                    f"{float(row['order_accuracy']):.8f}, "
                    f"phase_accuracy="
                    f"{float(row['phase_accuracy']):.8f}"
                )
            )

    lines.extend(
        [
            "",
            (
                "elapsed_seconds="
                f"{elapsed_seconds:.6f}"
            ),
            (
                "mean_seconds_per_noisy_observation="
                f"{elapsed_seconds / total_noisy_observations:.10f}"
            ),
            (
                "mean_seconds_per_method_prediction="
                f"{elapsed_seconds / total_predictions:.10f}"
            ),
            f"Prediction CSV: {PREDICTION_CSV_PATH}",
            f"Summary CSV: {SUMMARY_CSV_PATH}",
        ]
    )

    return "\n".join(
        lines
    )


def main() -> None:
    print("=" * 78)
    print("MC-VWLS FULL TEST ABLATION")
    print("=" * 78)

    if not H5_PATH.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {H5_PATH}"
        )

    test_indices = load_test_indices()

    total_noisy_observations = (
        len(test_indices)
        * len(SUPPORTED_SNR_DB)
    )

    total_predictions = (
        total_noisy_observations
        * len(METHOD_NAMES)
    )

    print(
        "Test clean samples:",
        len(test_indices),
    )

    print(
        "SNR levels:",
        SUPPORTED_SNR_DB,
    )

    print(
        "Noisy observations:",
        total_noisy_observations,
    )

    print(
        "Ablation methods:",
        METHOD_NAMES,
    )

    print(
        "Total method predictions:",
        total_predictions,
    )

    print(
        "Frozen weight_power:",
        WEIGHT_POWER,
    )

    print(
        "Frozen visibility_threshold:",
        VISIBILITY_THRESHOLD,
    )

    print(
        "Frozen regularization:",
        REGULARIZATION,
    )

    prediction_rows: List[
        Dict[str, object]
    ] = []

    start_time = time.perf_counter()
    completed_observations = 0

    with h5py.File(
        H5_PATH,
        "r",
    ) as h5:
        required_datasets = {
            "intensity",
            "visible_mask",
            "labels",
            "conditions",
        }

        missing_datasets = (
            required_datasets
            - set(
                h5.keys()
            )
        )

        if missing_datasets:
            raise KeyError(
                "Required HDF5 datasets are missing: "
                f"{sorted(missing_datasets)}"
            )

        intensity_ds = h5[
            "intensity"
        ]

        mask_ds = h5[
            "visible_mask"
        ]

        labels_ds = h5[
            "labels"
        ]

        conditions_ds = h5[
            "conditions"
        ]

        dataset_length = len(
            intensity_ds
        )

        if (
            len(mask_ds)
            != dataset_length
            or len(labels_ds)
            != dataset_length
            or len(conditions_ds)
            != dataset_length
        ):
            raise ValueError(
                "HDF5 dataset lengths are inconsistent."
            )

        for sample_index_value in test_indices:
            sample_index = int(
                sample_index_value
            )

            if not (
                0
                <= sample_index
                < dataset_length
            ):
                raise IndexError(
                    "Sample index is outside the dataset: "
                    f"{sample_index}"
                )

            clean = np.asarray(
                intensity_ds[
                    sample_index
                ],
                dtype=np.float32,
            )

            visible_mask = np.asarray(
                mask_ds[
                    sample_index
                ],
                dtype=np.float32,
            )

            true_label = int(
                labels_ds[
                    sample_index
                ]
            )

            target_occlusion = float(
                conditions_ds[
                    sample_index,
                    3,
                ]
            )

            for snr_db in SUPPORTED_SNR_DB:
                rows = evaluate_observation(
                    sample_index=sample_index,
                    snr_db=float(
                        snr_db
                    ),
                    clean=clean,
                    visible_mask=visible_mask,
                    true_label=true_label,
                    target_occlusion=target_occlusion,
                )

                prediction_rows.extend(
                    rows
                )

                completed_observations += 1

            if (
                completed_observations % 500 == 0
                or completed_observations
                == total_noisy_observations
            ):
                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                print(
                    f"Progress: "
                    f"{completed_observations}/"
                    f"{total_noisy_observations}, "
                    f"elapsed={elapsed:.2f} s"
                )

    if (
        completed_observations
        != total_noisy_observations
    ):
        raise RuntimeError(
            "Unexpected completed-observation count. "
            f"Expected {total_noisy_observations}, "
            f"found {completed_observations}."
        )

    if len(
        prediction_rows
    ) != total_predictions:
        raise RuntimeError(
            "Unexpected prediction-row count. "
            f"Expected {total_predictions}, "
            f"found {len(prediction_rows)}."
        )

    summary_rows = build_summary_rows(
        prediction_rows
    )

    save_csv(
        PREDICTION_CSV_PATH,
        prediction_rows,
    )

    save_csv(
        SUMMARY_CSV_PATH,
        summary_rows,
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    report_text = build_report(
        summary_rows,
        total_noisy_observations=(
            total_noisy_observations
        ),
        total_predictions=total_predictions,
        elapsed_seconds=elapsed_seconds,
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
        "MC-VWLS FULL TEST ABLATION COMPLETE"
    )
    print("=" * 78)

    print(
        "Predictions:",
        PREDICTION_CSV_PATH,
    )

    print(
        "Summary:",
        SUMMARY_CSV_PATH,
    )

    print(
        "Report:",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()