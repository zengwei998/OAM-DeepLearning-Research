"""
Evaluate DAF, ULS, and MC-VWLS on the complete frozen test set.

Outputs:
    results/csv/traditional_methods_full_test_predictions.csv
    results/csv/traditional_methods_summary.csv
    results/validation/traditional_methods_full_test_report.txt
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

from src.algorithms.daf import recognize_daf_state
from src.algorithms.harmonic_fit import recognize_harmonic_state
from src.algorithms.polar_sampling import (
    extract_polar_profile,
    normalize_angular_profile,
)
from src.algorithms.uls import recognize_uls_state
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
    / "traditional_methods_full_test_predictions.csv"
)

SUMMARY_CSV_PATH = (
    ROOT
    / "results"
    / "csv"
    / "traditional_methods_summary.csv"
)

REPORT_PATH = (
    ROOT
    / "results"
    / "validation"
    / "traditional_methods_full_test_report.txt"
)

EXPECTED_TEST_SAMPLES = 6720

ANGULAR_SAMPLES = 180
RADIAL_SAMPLES = 64
VISIBILITY_THRESHOLD = 0.05
REGULARIZATION = 1.0e-6
WEIGHT_POWER = 2.0
PHASE_BINS = 8
EPSILON = 1.0e-12

METHOD_NAMES = (
    "DAF",
    "ULS",
    "MC-VWLS",
)


def load_test_indices() -> np.ndarray:
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
    ).astype(np.int64)

    if len(test_indices) != EXPECTED_TEST_SAMPLES:
        raise ValueError(
            "Unexpected test-set size. "
            f"Expected {EXPECTED_TEST_SAMPLES}, "
            f"found {len(test_indices)}."
        )

    return np.sort(test_indices)


def build_daf_profile(
    polar_intensity: np.ndarray,
    radius: np.ndarray,
) -> np.ndarray:
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
            "Radius length does not match polar intensity."
        )

    radial_weights = radius_array[:, None]

    full_support = float(
        np.sum(
            radial_weights,
            dtype=np.float64,
        )
    )

    if full_support <= EPSILON:
        raise ValueError(
            "Full radial support is zero."
        )

    return np.sum(
        intensity * radial_weights,
        axis=0,
        dtype=np.float64,
    ) / full_support


def circular_phase_bin_error(
    true_bin: int,
    predicted_bin: int,
) -> int:
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
    confidence: float,
    harmonic_margin: float,
    best_score: float,
    second_best_score: float,
    mean_visibility: float,
    valid_fraction: float,
) -> Dict[str, float]:
    true_order = (
        true_label // PHASE_BINS + 1
    )

    true_phase_bin = (
        true_label % PHASE_BINS
    )

    return {
        "method": method,
        "sample_index": int(sample_index),
        "target_snr_db": float(snr_db),
        "target_occlusion": float(target_occlusion),
        "true_label": int(true_label),
        "predicted_label": int(predicted_label),
        "true_order": int(true_order),
        "predicted_order": int(predicted_order),
        "true_phase_bin": int(true_phase_bin),
        "predicted_phase_bin": int(
            predicted_phase_bin
        ),
        "phase_bin_error": circular_phase_bin_error(
            true_bin=true_phase_bin,
            predicted_bin=predicted_phase_bin,
        ),
        "label_correct": int(
            predicted_label == true_label
        ),
        "order_correct": int(
            predicted_order == true_order
        ),
        "phase_correct": int(
            predicted_phase_bin == true_phase_bin
        ),
        "confidence": float(confidence),
        "harmonic_margin": float(
            harmonic_margin
        ),
        "best_score": float(best_score),
        "second_best_score": float(
            second_best_score
        ),
        "mean_visibility": float(
            mean_visibility
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
) -> List[Dict[str, float]]:
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

    daf_profile = build_daf_profile(
        polar_intensity=polar.polar_intensity,
        radius=polar.radius,
    )

    normalized_profile = normalize_angular_profile(
        angular_profile=polar.angular_profile,
        valid_angles=polar.valid_angles,
        remove_mean=True,
        unit_norm=False,
    )

    daf_result = recognize_daf_state(
        theta=polar.theta,
        angular_profile=daf_profile,
        candidate_orders=(1, 2, 3, 4),
        phase_bins=PHASE_BINS,
    )

    uls_result = recognize_uls_state(
        theta=polar.theta,
        angular_profile=normalized_profile,
        candidate_orders=(1, 2, 3, 4),
        phase_bins=PHASE_BINS,
    )

    mc_result = recognize_harmonic_state(
        theta=polar.theta,
        angular_profile=normalized_profile,
        angular_visibility=polar.angular_visibility,
        valid_angles=polar.valid_angles,
        candidate_orders=(1, 2, 3, 4),
        phase_bins=PHASE_BINS,
        regularization=REGULARIZATION,
        weight_power=WEIGHT_POWER,
    )

    mean_visibility = float(
        np.mean(
            polar.angular_visibility,
            dtype=np.float64,
        )
    )

    valid_fraction = float(
        np.mean(
            polar.valid_angles,
            dtype=np.float64,
        )
    )

    return [
        make_prediction_row(
            method="DAF",
            sample_index=sample_index,
            snr_db=snr_db,
            target_occlusion=target_occlusion,
            true_label=true_label,
            predicted_label=daf_result.predicted_label,
            predicted_order=daf_result.predicted_order,
            predicted_phase_bin=daf_result.predicted_phase_bin,
            confidence=daf_result.confidence,
            harmonic_margin=daf_result.harmonic_margin,
            best_score=daf_result.best_score,
            second_best_score=daf_result.second_best_score,
            mean_visibility=mean_visibility,
            valid_fraction=valid_fraction,
        ),
        make_prediction_row(
            method="ULS",
            sample_index=sample_index,
            snr_db=snr_db,
            target_occlusion=target_occlusion,
            true_label=true_label,
            predicted_label=uls_result.predicted_label,
            predicted_order=uls_result.predicted_order,
            predicted_phase_bin=uls_result.predicted_phase_bin,
            confidence=uls_result.confidence,
            harmonic_margin=uls_result.harmonic_margin,
            best_score=uls_result.best_score,
            second_best_score=uls_result.second_best_score,
            mean_visibility=mean_visibility,
            valid_fraction=valid_fraction,
        ),
        make_prediction_row(
            method="MC-VWLS",
            sample_index=sample_index,
            snr_db=snr_db,
            target_occlusion=target_occlusion,
            true_label=true_label,
            predicted_label=mc_result.predicted_label,
            predicted_order=mc_result.predicted_order,
            predicted_phase_bin=mc_result.predicted_phase_bin,
            confidence=mc_result.confidence,
            harmonic_margin=mc_result.harmonic_margin,
            best_score=mc_result.best_score,
            second_best_score=mc_result.second_best_score,
            mean_visibility=mean_visibility,
            valid_fraction=valid_fraction,
        ),
    ]


def calculate_mean(
    rows: List[Dict[str, float]],
    key: str,
) -> float:
    if not rows:
        return float("nan")

    return float(
        np.mean(
            [
                row[key]
                for row in rows
            ],
            dtype=np.float64,
        )
    )


def build_summary_rows(
    prediction_rows: List[Dict[str, float]],
) -> List[Dict[str, float]]:
    summary_rows: List[
        Dict[str, float]
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

        summary_rows.append(
            {
                "method": method,
                "scope": "overall",
                "target_snr_db": "",
                "target_occlusion": "",
                "sample_count": len(method_rows),
                "label_accuracy": calculate_mean(
                    method_rows,
                    "label_correct",
                ),
                "order_accuracy": calculate_mean(
                    method_rows,
                    "order_correct",
                ),
                "phase_accuracy": calculate_mean(
                    method_rows,
                    "phase_correct",
                ),
                "mean_phase_bin_error": calculate_mean(
                    method_rows,
                    "phase_bin_error",
                ),
                "mean_confidence": calculate_mean(
                    method_rows,
                    "confidence",
                ),
            }
        )

        for snr_db in SUPPORTED_SNR_DB:
            snr_rows = [
                row
                for row in method_rows
                if float(
                    row["target_snr_db"]
                ) == float(snr_db)
            ]

            summary_rows.append(
                {
                    "method": method,
                    "scope": "snr",
                    "target_snr_db": float(
                        snr_db
                    ),
                    "target_occlusion": "",
                    "sample_count": len(snr_rows),
                    "label_accuracy": calculate_mean(
                        snr_rows,
                        "label_correct",
                    ),
                    "order_accuracy": calculate_mean(
                        snr_rows,
                        "order_correct",
                    ),
                    "phase_accuracy": calculate_mean(
                        snr_rows,
                        "phase_correct",
                    ),
                    "mean_phase_bin_error": calculate_mean(
                        snr_rows,
                        "phase_bin_error",
                    ),
                    "mean_confidence": calculate_mean(
                        snr_rows,
                        "confidence",
                    ),
                }
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

            summary_rows.append(
                {
                    "method": method,
                    "scope": "occlusion",
                    "target_snr_db": "",
                    "target_occlusion": occlusion,
                    "sample_count": len(
                        occlusion_rows
                    ),
                    "label_accuracy": calculate_mean(
                        occlusion_rows,
                        "label_correct",
                    ),
                    "order_accuracy": calculate_mean(
                        occlusion_rows,
                        "order_correct",
                    ),
                    "phase_accuracy": calculate_mean(
                        occlusion_rows,
                        "phase_correct",
                    ),
                    "mean_phase_bin_error": calculate_mean(
                        occlusion_rows,
                        "phase_bin_error",
                    ),
                    "mean_confidence": calculate_mean(
                        occlusion_rows,
                        "confidence",
                    ),
                }
            )

        for snr_db in SUPPORTED_SNR_DB:
            for occlusion in occlusion_levels:
                condition_rows = [
                    row
                    for row in method_rows
                    if (
                        float(
                            row["target_snr_db"]
                        ) == float(snr_db)
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

                summary_rows.append(
                    {
                        "method": method,
                        "scope": "snr_occlusion",
                        "target_snr_db": float(
                            snr_db
                        ),
                        "target_occlusion": occlusion,
                        "sample_count": len(
                            condition_rows
                        ),
                        "label_accuracy": calculate_mean(
                            condition_rows,
                            "label_correct",
                        ),
                        "order_accuracy": calculate_mean(
                            condition_rows,
                            "order_correct",
                        ),
                        "phase_accuracy": calculate_mean(
                            condition_rows,
                            "phase_correct",
                        ),
                        "mean_phase_bin_error": calculate_mean(
                            condition_rows,
                            "phase_bin_error",
                        ),
                        "mean_confidence": calculate_mean(
                            condition_rows,
                            "confidence",
                        ),
                    }
                )

    return summary_rows


def save_csv(
    path: Path,
    rows: List[Dict[str, float]],
) -> None:
    if not rows:
        raise ValueError(
            f"No rows available for {path}."
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
        writer.writerows(rows)


def build_report(
    summary_rows: List[Dict[str, float]],
    *,
    total_observations: int,
    elapsed_seconds: float,
) -> str:
    lines = [
        "Traditional-method full test report",
        f"Dataset: {H5_PATH}",
        f"Split: {SPLIT_PATH}",
        f"Clean test samples: {EXPECTED_TEST_SAMPLES}",
        f"Noisy observations per method: {total_observations}",
        f"Methods: {METHOD_NAMES}",
        "",
        "[OVERALL]",
    ]

    for method in METHOD_NAMES:
        row = next(
            item
            for item in summary_rows
            if (
                item["method"] == method
                and item["scope"]
                == "overall"
            )
        )

        lines.append(
            (
                f"method={method}, "
                f"samples={row['sample_count']}, "
                f"label_accuracy="
                f"{row['label_accuracy']:.8f}, "
                f"order_accuracy="
                f"{row['order_accuracy']:.8f}, "
                f"phase_accuracy="
                f"{row['phase_accuracy']:.8f}, "
                f"mean_phase_bin_error="
                f"{row['mean_phase_bin_error']:.8f}, "
                f"mean_confidence="
                f"{row['mean_confidence']:.8f}"
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
            row = next(
                item
                for item in summary_rows
                if (
                    item["method"] == method
                    and item["scope"] == "snr"
                    and float(
                        item["target_snr_db"]
                    ) == float(snr_db)
                )
            )

            lines.append(
                (
                    f"SNR={float(snr_db):.1f} dB, "
                    f"method={method}, "
                    f"label_accuracy="
                    f"{row['label_accuracy']:.8f}"
                )
            )

    lines.extend(
        [
            "",
            f"elapsed_seconds={elapsed_seconds:.6f}",
            (
                "mean_seconds_per_noisy_observation="
                f"{elapsed_seconds / total_observations:.10f}"
            ),
            f"Prediction CSV: {PREDICTION_CSV_PATH}",
            f"Summary CSV: {SUMMARY_CSV_PATH}",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    print("=" * 78)
    print("TRADITIONAL METHODS FULL TEST EVALUATION")
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

    total_method_predictions = (
        total_noisy_observations
        * len(METHOD_NAMES)
    )

    print(
        "Clean test samples:",
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
        "Method predictions:",
        total_method_predictions,
    )

    prediction_rows: List[
        Dict[str, float]
    ] = []

    start_time = time.perf_counter()
    completed_observations = 0

    with h5py.File(
        H5_PATH,
        "r",
    ) as h5:
        intensity_ds = h5["intensity"]
        mask_ds = h5["visible_mask"]
        labels_ds = h5["labels"]
        conditions_ds = h5["conditions"]

        for sample_index_value in test_indices:
            sample_index = int(
                sample_index_value
            )

            clean = np.asarray(
                intensity_ds[sample_index],
                dtype=np.float32,
            )

            visible_mask = np.asarray(
                mask_ds[sample_index],
                dtype=np.float32,
            )

            true_label = int(
                labels_ds[sample_index]
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
                    snr_db=float(snr_db),
                    clean=clean,
                    visible_mask=visible_mask,
                    true_label=true_label,
                    target_occlusion=target_occlusion,
                )

                prediction_rows.extend(rows)
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

    if len(prediction_rows) != total_method_predictions:
        raise RuntimeError(
            "Unexpected prediction-row count. "
            f"Expected {total_method_predictions}, "
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
        total_observations=total_noisy_observations,
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
    print(report_text)
    print("")
    print("=" * 78)
    print(
        "TRADITIONAL METHODS FULL TEST COMPLETE"
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