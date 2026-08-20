"""
Validate visibility-weighted harmonic fitting on representative samples.
"""

from pathlib import Path

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
    / "harmonic_fit_check.txt"
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


def main() -> None:
    print("=" * 78)
    print("VISIBILITY-WEIGHTED HARMONIC FIT CHECK")
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
    finite_ok = True

    with h5py.File(
        H5_PATH,
        "r",
    ) as h5:
        intensity_ds = h5["intensity"]
        mask_ds = h5["visible_mask"]
        labels = h5["labels"][:]
        conditions = h5["conditions"][:]

        for sample_index in SAMPLE_INDICES:
            clean = intensity_ds[sample_index]
            visible_mask = mask_ds[sample_index]

            true_label = int(
                labels[sample_index]
            )

            true_order = (
                true_label // 8 + 1
            )

            true_phase_bin = (
                true_label % 8
            )

            target_occlusion = float(
                conditions[sample_index, 3]
            )

            observation = add_deterministic_awgn(
                clean_intensity=clean,
                sample_index=sample_index,
                snr_db=TEST_SNR_DB,
            )

            polar = extract_polar_profile(
                intensity=observation.intensity,
                visible_mask=visible_mask,
                angular_samples=180,
                radial_samples=64,
                visibility_threshold=0.05,
            )

            normalized_profile = (
                normalize_angular_profile(
                    angular_profile=polar.angular_profile,
                    valid_angles=polar.valid_angles,
                    remove_mean=True,
                    unit_norm=False,
                )
            )

            result = recognize_harmonic_state(
                theta=polar.theta,
                angular_profile=normalized_profile,
                angular_visibility=polar.angular_visibility,
                valid_angles=polar.valid_angles,
                candidate_orders=(1, 2, 3, 4),
                phase_bins=8,
                regularization=1.0e-6,
                weight_power=2.0,
            )

            candidate_scores = {
                order: candidate.score
                for order, candidate
                in result.candidates.items()
            }

            all_finite = bool(
                np.all(
                    np.isfinite(
                        list(
                            candidate_scores.values()
                        )
                    )
                )
                and np.isfinite(
                    result.predicted_phase_rad
                )
                and np.isfinite(
                    result.confidence
                )
            )

            if not all_finite:
                finite_ok = False

            report_lines.append(
                (
                    f"sample={sample_index}, "
                    f"true_label={true_label}, "
                    f"true_order={true_order}, "
                    f"true_phase_bin={true_phase_bin}, "
                    f"occlusion={target_occlusion:.1f}, "
                    f"SNR={TEST_SNR_DB:.1f}, "
                    f"pred_label={result.predicted_label}, "
                    f"pred_order={result.predicted_order}, "
                    f"pred_phase_bin={result.predicted_phase_bin}, "
                    f"phase_rad={result.predicted_phase_rad:.6f}, "
                    f"confidence={result.confidence:.6f}, "
                    f"margin={result.harmonic_margin:.6f}, "
                    f"scores={candidate_scores}, "
                    f"finite={all_finite}"
                )
            )

    status = (
        "PASS"
        if finite_ok
        else "FAIL"
    )

    report_lines.append("")
    report_lines.append("=" * 78)
    report_lines.append(
        f"HARMONIC FIT STATUS: {status}"
    )
    report_lines.append("=" * 78)

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