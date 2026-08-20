"""
Validate DAF, ULS, and MC-VWLS on representative receiver observations.
"""

from __future__ import annotations

from pathlib import Path

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
    add_deterministic_awgn,
)


ROOT = Path(__file__).resolve().parents[2]

H5_PATH = (
    ROOT
    / "data"
    / "generated"
    / "occlusion_clean_v2.h5"
)

REPORT_PATH = (
    ROOT
    / "results"
    / "validation"
    / "baseline_check.txt"
)

SAMPLE_INDICES = (
    0,
    7,
    21700,
    21702,
    21704,
    44795,
    44799,
)

TEST_SNR_DB = 10.0
PHASE_BINS = 8

ANGULAR_SAMPLES = 180
RADIAL_SAMPLES = 64
VISIBILITY_THRESHOLD = 0.05
REGULARIZATION = 1.0e-6
WEIGHT_POWER = 2.0

EPSILON = 1.0e-12


def build_daf_profile(
    polar_intensity: np.ndarray,
    radius: np.ndarray,
) -> np.ndarray:
    """
    Build the direct angular Fourier profile.

    DAF uses the observed polar intensity directly. Occluded locations
    remain zero and no mask-support normalization is applied.
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
            "Radius length does not match polar radial dimension."
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

    return np.asarray(
        raw_profile,
        dtype=np.float64,
    )


def main() -> None:
    print("=" * 78)
    print("DAF / ULS / MC-VWLS BASELINE CHECK")
    print("=" * 78)

    if not H5_PATH.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {H5_PATH}"
        )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_lines = []
    all_finite = True

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
            - set(h5.keys())
        )

        if missing_datasets:
            raise KeyError(
                "Required datasets are missing: "
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

        for sample_index in SAMPLE_INDICES:
            if not 0 <= sample_index < dataset_length:
                raise IndexError(
                    f"Sample index is outside dataset: {sample_index}"
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

            true_order = (
                true_label
                // PHASE_BINS
                + 1
            )

            true_phase_bin = (
                true_label
                % PHASE_BINS
            )

            target_occlusion = float(
                conditions_ds[
                    sample_index,
                    3,
                ]
            )

            observation = add_deterministic_awgn(
                clean_intensity=clean,
                sample_index=sample_index,
                snr_db=TEST_SNR_DB,
            )

            polar = extract_polar_profile(
                intensity=observation.intensity,
                visible_mask=visible_mask,
                angular_samples=ANGULAR_SAMPLES,
                radial_samples=RADIAL_SAMPLES,
                visibility_threshold=VISIBILITY_THRESHOLD,
            )

            # DAF:
            # Direct radial integration of the observed intensity.
            # No mask-support normalization and no visibility weighting.
            raw_profile = build_daf_profile(
                polar_intensity=polar.polar_intensity,
                radius=polar.radius,
            )

            # ULS and MC-VWLS:
            # Use mask-support-normalized angular profile.
            normalized_profile = normalize_angular_profile(
                angular_profile=polar.angular_profile,
                valid_angles=polar.valid_angles,
                remove_mean=True,
                unit_norm=False,
            )

            daf_result = recognize_daf_state(
                theta=polar.theta,
                angular_profile=raw_profile,
                candidate_orders=(1, 2, 3, 4),
                phase_bins=PHASE_BINS,
            )

            uls_result = recognize_uls_state(
                theta=polar.theta,
                angular_profile=normalized_profile,
                candidate_orders=(1, 2, 3, 4),
                phase_bins=PHASE_BINS,
            )

            mc_vwls_result = recognize_harmonic_state(
                theta=polar.theta,
                angular_profile=normalized_profile,
                angular_visibility=polar.angular_visibility,
                valid_angles=polar.valid_angles,
                candidate_orders=(1, 2, 3, 4),
                phase_bins=PHASE_BINS,
                regularization=REGULARIZATION,
                weight_power=WEIGHT_POWER,
            )

            values_to_check = np.asarray(
                [
                    daf_result.predicted_phase_rad,
                    daf_result.confidence,
                    daf_result.best_score,
                    daf_result.second_best_score,
                    uls_result.predicted_phase_rad,
                    uls_result.confidence,
                    uls_result.best_score,
                    uls_result.second_best_score,
                    mc_vwls_result.predicted_phase_rad,
                    mc_vwls_result.confidence,
                    mc_vwls_result.best_score,
                    mc_vwls_result.second_best_score,
                ],
                dtype=np.float64,
            )

            finite = bool(
                np.all(
                    np.isfinite(
                        values_to_check
                    )
                )
            )

            if not finite:
                all_finite = False

            report_lines.append(
                (
                    f"sample={sample_index}, "
                    f"true_label={true_label}, "
                    f"true_order={true_order}, "
                    f"true_phase_bin={true_phase_bin}, "
                    f"occlusion={target_occlusion:.1f}, "
                    f"SNR={TEST_SNR_DB:.1f}, "
                    f"DAF={daf_result.predicted_label}, "
                    f"ULS={uls_result.predicted_label}, "
                    f"MC_VWLS={mc_vwls_result.predicted_label}, "
                    f"DAF_order={daf_result.predicted_order}, "
                    f"ULS_order={uls_result.predicted_order}, "
                    f"MC_order={mc_vwls_result.predicted_order}, "
                    f"DAF_conf={daf_result.confidence:.6f}, "
                    f"ULS_conf={uls_result.confidence:.6f}, "
                    f"MC_conf={mc_vwls_result.confidence:.6f}, "
                    f"visibility_mean="
                    f"{np.mean(polar.angular_visibility):.6f}, "
                    f"finite={finite}"
                )
            )

    status = (
        "PASS"
        if all_finite
        else "FAIL"
    )

    report_lines.extend(
        [
            "",
            "=" * 78,
            f"BASELINE CHECK STATUS: {status}",
            "=" * 78,
        ]
    )

    report_text = "\n".join(
        report_lines
    )

    REPORT_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    print(report_text)
    print("")
    print("Saved report:")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()