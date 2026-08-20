"""
Evaluate tuned MC-VWLS on the complete frozen test set.

The hyperparameters were selected using the validation set only.

Outputs:
    results/csv/mc_vwls_tuned_full_test_predictions.csv
    results/validation/mc_vwls_tuned_full_test_report.txt
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

from src.algorithms.harmonic_fit import (
    recognize_harmonic_state,
)
from src.algorithms.polar_sampling import (
    extract_polar_profile,
    normalize_angular_profile,
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

CSV_PATH = (
    ROOT
    / "results"
    / "csv"
    / "mc_vwls_tuned_full_test_predictions.csv"
)

REPORT_PATH = (
    ROOT
    / "results"
    / "validation"
    / "mc_vwls_tuned_full_test_report.txt"
)

EXPECTED_TEST_SAMPLES = 6720

ANGULAR_SAMPLES = 180
RADIAL_SAMPLES = 64
PHASE_BINS = 8

# Frozen from validation-set tuning.
VISIBILITY_THRESHOLD = 0.0
REGULARIZATION = 0.0
WEIGHT_POWER = 0.5


def load_test_indices() -> np.ndarray:
    """
    Load test-set sample indices from the frozen split file.

    Frozen split coding:
        0 = train
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
        available_keys = set(
            split.files
        )

        if "sample_split_codes" not in available_keys:
            raise KeyError(
                "sample_split_codes was not found in split file. "
                f"Available keys: {sorted(available_keys)}"
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

    if len(test_indices) == 0:
        unique_codes, counts = np.unique(
            split_codes,
            return_counts=True,
        )

        observed_codes = dict(
            zip(
                unique_codes.tolist(),
                counts.tolist(),
            )
        )

        raise ValueError(
            "No test samples were found for split code 2. "
            f"Observed split codes: {observed_codes}"
        )

    if len(test_indices) != EXPECTED_TEST_SAMPLES:
        raise ValueError(
            "Unexpected test-set size. "
            f"Expected {EXPECTED_TEST_SAMPLES}, "
            f"found {len(test_indices)}."
        )

    return np.sort(
        test_indices
    )


def circular_phase_bin_error(
    true_bin: int,
    predicted_bin: int,
    phase_bins: int = PHASE_BINS,
) -> int:
    """
    Return circular distance between two phase bins.
    """

    direct_error = abs(
        int(true_bin)
        - int(predicted_bin)
    )

    return int(
        min(
            direct_error,
            phase_bins - direct_error,
        )
    )


def evaluate_sample(
    *,
    sample_index: int,
    snr_db: float,
    clean: np.ndarray,
    visible_mask: np.ndarray,
    true_label: int,
    target_occlusion: float,
) -> Dict[str, float]:
    """
    Evaluate one clean sample at one nominal SNR.
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

    normalized_profile = normalize_angular_profile(
        angular_profile=polar.angular_profile,
        valid_angles=polar.valid_angles,
        remove_mean=True,
        unit_norm=False,
    )

    result = recognize_harmonic_state(
        theta=polar.theta,
        angular_profile=normalized_profile,
        angular_visibility=polar.angular_visibility,
        valid_angles=polar.valid_angles,
        candidate_orders=(1, 2, 3, 4),
        phase_bins=PHASE_BINS,
        regularization=REGULARIZATION,
        weight_power=WEIGHT_POWER,
    )

    true_order = (
        true_label
        // PHASE_BINS
        + 1
    )

    true_phase_bin = (
        true_label
        % PHASE_BINS
    )

    phase_bin_error = circular_phase_bin_error(
        true_bin=true_phase_bin,
        predicted_bin=result.predicted_phase_bin,
    )

    return {
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
            result.predicted_label
        ),
        "true_order": int(
            true_order
        ),
        "predicted_order": int(
            result.predicted_order
        ),
        "true_phase_bin": int(
            true_phase_bin
        ),
        "predicted_phase_bin": int(
            result.predicted_phase_bin
        ),
        "phase_bin_error": int(
            phase_bin_error
        ),
        "label_correct": int(
            result.predicted_label
            == true_label
        ),
        "order_correct": int(
            result.predicted_order
            == true_order
        ),
        "phase_correct": int(
            result.predicted_phase_bin
            == true_phase_bin
        ),
        "confidence": float(
            result.confidence
        ),
        "harmonic_margin": float(
            result.harmonic_margin
        ),
        "best_score": float(
            result.best_score
        ),
        "second_best_score": float(
            result.second_best_score
        ),
        "valid_fraction": float(
            result.valid_fraction
        ),
        "mean_visibility": float(
            result.mean_visibility
        ),
        "measured_snr_preclip_db": float(
            observation.measured_snr_preclip_db
        ),
        "measured_snr_postclip_db": float(
            observation.measured_snr_postclip_db
        ),
        "clipped_pixel_fraction": float(
            observation.clipped_pixel_fraction
        ),
    }


def calculate_mean(
    rows: List[Dict[str, float]],
    key: str,
) -> float:
    """
    Calculate the mean value of one result column.
    """

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


def summarize_rows(
    rows: List[Dict[str, float]],
) -> List[str]:
    """
    Build overall and condition-wise summaries.
    """

    if not rows:
        raise ValueError(
            "No prediction rows were generated."
        )

    lines: List[str] = []

    overall_label_accuracy = calculate_mean(
        rows,
        "label_correct",
    )

    overall_order_accuracy = calculate_mean(
        rows,
        "order_correct",
    )

    overall_phase_accuracy = calculate_mean(
        rows,
        "phase_correct",
    )

    overall_phase_bin_error = calculate_mean(
        rows,
        "phase_bin_error",
    )

    overall_confidence = calculate_mean(
        rows,
        "confidence",
    )

    lines.extend(
        [
            f"overall_samples={len(rows)}",
            (
                "overall_label_accuracy="
                f"{overall_label_accuracy:.8f}"
            ),
            (
                "overall_order_accuracy="
                f"{overall_order_accuracy:.8f}"
            ),
            (
                "overall_phase_accuracy="
                f"{overall_phase_accuracy:.8f}"
            ),
            (
                "overall_mean_phase_bin_error="
                f"{overall_phase_bin_error:.8f}"
            ),
            (
                "overall_mean_confidence="
                f"{overall_confidence:.8f}"
            ),
            "",
            "[BY SNR]",
        ]
    )

    for snr_db in SUPPORTED_SNR_DB:
        snr_rows = [
            row
            for row in rows
            if float(
                row["target_snr_db"]
            ) == float(
                snr_db
            )
        ]

        lines.append(
            (
                f"SNR={float(snr_db):.1f} dB, "
                f"samples={len(snr_rows)}, "
                f"label_accuracy="
                f"{calculate_mean(snr_rows, 'label_correct'):.8f}, "
                f"order_accuracy="
                f"{calculate_mean(snr_rows, 'order_correct'):.8f}, "
                f"phase_accuracy="
                f"{calculate_mean(snr_rows, 'phase_correct'):.8f}"
            )
        )

    occlusion_levels = sorted(
        {
            float(
                row["target_occlusion"]
            )
            for row in rows
        }
    )

    lines.extend(
        [
            "",
            "[BY OCCLUSION]",
        ]
    )

    for occlusion in occlusion_levels:
        occlusion_rows = [
            row
            for row in rows
            if np.isclose(
                float(
                    row["target_occlusion"]
                ),
                occlusion,
                rtol=0.0,
                atol=1.0e-8,
            )
        ]

        lines.append(
            (
                f"occlusion={occlusion:.1f}, "
                f"samples={len(occlusion_rows)}, "
                f"label_accuracy="
                f"{calculate_mean(occlusion_rows, 'label_correct'):.8f}, "
                f"order_accuracy="
                f"{calculate_mean(occlusion_rows, 'order_correct'):.8f}, "
                f"phase_accuracy="
                f"{calculate_mean(occlusion_rows, 'phase_correct'):.8f}"
            )
        )

    lines.extend(
        [
            "",
            "[BY SNR AND OCCLUSION]",
        ]
    )

    for snr_db in SUPPORTED_SNR_DB:
        for occlusion in occlusion_levels:
            condition_rows = [
                row
                for row in rows
                if (
                    float(
                        row["target_snr_db"]
                    ) == float(
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

            lines.append(
                (
                    f"SNR={float(snr_db):.1f} dB, "
                    f"occlusion={occlusion:.1f}, "
                    f"samples={len(condition_rows)}, "
                    f"label_accuracy="
                    f"{calculate_mean(condition_rows, 'label_correct'):.8f}, "
                    f"order_accuracy="
                    f"{calculate_mean(condition_rows, 'order_correct'):.8f}, "
                    f"phase_accuracy="
                    f"{calculate_mean(condition_rows, 'phase_correct'):.8f}"
                )
            )

    lines.extend(
        [
            "",
            "[BY OAM ORDER]",
        ]
    )

    for true_order in (
        1,
        2,
        3,
        4,
    ):
        order_rows = [
            row
            for row in rows
            if int(
                row["true_order"]
            ) == true_order
        ]

        lines.append(
            (
                f"order={true_order}, "
                f"samples={len(order_rows)}, "
                f"label_accuracy="
                f"{calculate_mean(order_rows, 'label_correct'):.8f}, "
                f"order_accuracy="
                f"{calculate_mean(order_rows, 'order_correct'):.8f}, "
                f"phase_accuracy="
                f"{calculate_mean(order_rows, 'phase_correct'):.8f}"
            )
        )

    return lines


def save_predictions_csv(
    rows: List[Dict[str, float]],
) -> None:
    """
    Save all test-set prediction rows.
    """

    if not rows:
        raise ValueError(
            "Cannot save an empty prediction table."
        )

    CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[0].keys()
    )

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
        writer.writerows(
            rows
        )


def main() -> None:
    print("=" * 78)
    print("TUNED MC-VWLS FULL TEST EVALUATION")
    print("=" * 78)

    if not H5_PATH.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {H5_PATH}"
        )

    test_indices = load_test_indices()

    total_observations = (
        len(test_indices)
        * len(SUPPORTED_SNR_DB)
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
        "Total noisy observations:",
        total_observations,
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

    rows: List[
        Dict[str, float]
    ] = []

    start_time = time.perf_counter()
    completed = 0

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
                row = evaluate_sample(
                    sample_index=sample_index,
                    snr_db=float(
                        snr_db
                    ),
                    clean=clean,
                    visible_mask=visible_mask,
                    true_label=true_label,
                    target_occlusion=target_occlusion,
                )

                rows.append(
                    row
                )

                completed += 1

            if (
                completed % 500 == 0
                or completed
                == total_observations
            ):
                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                print(
                    f"Progress: {completed}/"
                    f"{total_observations}, "
                    f"elapsed={elapsed:.2f} s"
                )

    if len(rows) != total_observations:
        raise RuntimeError(
            "Unexpected prediction-row count. "
            f"Expected {total_observations}, "
            f"found {len(rows)}."
        )

    save_predictions_csv(
        rows
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    mean_seconds_per_observation = (
        elapsed_seconds
        / len(rows)
    )

    report_lines = [
        "Tuned MC-VWLS full test evaluation report",
        f"Dataset: {H5_PATH}",
        f"Split: {SPLIT_PATH}",
        f"Test clean samples: {len(test_indices)}",
        f"SNR levels: {SUPPORTED_SNR_DB}",
        f"Total noisy observations: {len(rows)}",
        "",
        "[FROZEN VALIDATION-SELECTED PARAMETERS]",
        f"weight_power={WEIGHT_POWER:.8f}",
        (
            "visibility_threshold="
            f"{VISIBILITY_THRESHOLD:.8f}"
        ),
        (
            "regularization="
            f"{REGULARIZATION:.10e}"
        ),
        "",
    ]

    report_lines.extend(
        summarize_rows(
            rows
        )
    )

    report_lines.extend(
        [
            "",
            (
                "elapsed_seconds="
                f"{elapsed_seconds:.6f}"
            ),
            (
                "mean_seconds_per_observation="
                f"{mean_seconds_per_observation:.10f}"
            ),
            f"CSV: {CSV_PATH}",
        ]
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_text = "\n".join(
        report_lines
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
        "TUNED MC-VWLS FULL TEST COMPLETE"
    )
    print("=" * 78)
    print(
        "CSV:",
        CSV_PATH,
    )
    print(
        "Report:",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()