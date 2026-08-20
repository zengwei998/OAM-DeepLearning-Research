"""
Validate polar sampling and radial visibility normalization.
"""

from pathlib import Path

import h5py
import numpy as np

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
    / "polar_sampling_check.txt"
)

SAMPLE_INDICES = (
    21700,  # no occlusion
    21702,  # target occlusion approximately 0.2
    21704,  # target occlusion approximately 0.4
)

TEST_SNR_DB = 10.0


def main() -> None:
    print("=" * 78)
    print("POLAR SAMPLING AND VISIBILITY NORMALIZATION CHECK")
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
    all_ok = True

    with h5py.File(
        H5_PATH,
        "r",
    ) as h5:
        intensity_ds = h5["intensity"]
        mask_ds = h5["visible_mask"]
        conditions = h5["conditions"][:]

        for sample_index in SAMPLE_INDICES:
            clean = intensity_ds[sample_index]
            visible_mask = mask_ds[sample_index]

            observation = add_deterministic_awgn(
                clean_intensity=clean,
                sample_index=sample_index,
                snr_db=TEST_SNR_DB,
            )

            profile = extract_polar_profile(
                intensity=observation.intensity,
                visible_mask=visible_mask,
                angular_samples=180,
                radial_samples=64,
                visibility_threshold=0.05,
            )

            normalized = normalize_angular_profile(
                angular_profile=profile.angular_profile,
                valid_angles=profile.valid_angles,
                remove_mean=True,
                unit_norm=False,
            )

            finite = bool(
                np.all(
                    np.isfinite(
                        profile.angular_profile
                    )
                )
                and np.all(
                    np.isfinite(
                        profile.angular_visibility
                    )
                )
                and np.all(
                    np.isfinite(normalized)
                )
            )

            visibility_in_range = bool(
                np.all(
                    profile.angular_visibility >= 0.0
                )
                and np.all(
                    profile.angular_visibility <= 1.0 + 1.0e-12
                )
            )

            valid_fraction = float(
                np.mean(
                    profile.valid_angles
                )
            )

            target_occlusion = float(
                conditions[sample_index, 3]
            )

            report_lines.append(
                (
                    f"sample={sample_index}, "
                    f"target_occlusion={target_occlusion:.1f}, "
                    f"SNR={TEST_SNR_DB:.1f} dB, "
                    f"polar_shape={profile.polar_intensity.shape}, "
                    f"visibility_min={np.min(profile.angular_visibility):.6f}, "
                    f"visibility_mean={np.mean(profile.angular_visibility):.6f}, "
                    f"visibility_max={np.max(profile.angular_visibility):.6f}, "
                    f"valid_fraction={valid_fraction:.6f}, "
                    f"profile_min={np.min(profile.angular_profile):.10e}, "
                    f"profile_max={np.max(profile.angular_profile):.10e}, "
                    f"finite={finite}"
                )
            )

            if not finite:
                all_ok = False
                report_lines.append(
                    "  [FAIL] Non-finite values detected."
                )

            if not visibility_in_range:
                all_ok = False
                report_lines.append(
                    "  [FAIL] Visibility outside [0, 1]."
                )

            if profile.polar_intensity.shape != (
                64,
                180,
            ):
                all_ok = False
                report_lines.append(
                    "  [FAIL] Unexpected polar array shape."
                )

            if not 0.0 < valid_fraction <= 1.0:
                all_ok = False
                report_lines.append(
                    "  [FAIL] Invalid valid-angle fraction."
                )

    status = "PASS" if all_ok else "FAIL"

    report_lines.append("")
    report_lines.append("=" * 78)
    report_lines.append(
        f"POLAR SAMPLING STATUS: {status}"
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