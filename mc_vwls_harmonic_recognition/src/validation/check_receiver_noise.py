"""
Validate deterministic AWGN receiver observation generation.

Checks:
1. Five target SNR values.
2. Exact reproducibility.
3. Different SNR values use different noise seeds.
4. Pre-clipping measured SNR matches target.
5. No NaN, Inf, or negative output values after clipping.
6. Clean source data are not modified.
"""

from pathlib import Path

import h5py
import numpy as np

from src.physics.receiver_noise import (
    GLOBAL_NOISE_SEED,
    SUPPORTED_SNR_DB,
    add_deterministic_awgn,
    observation_id,
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
    / "receiver_noise_check.txt"
)

SELECTED_SAMPLE_INDICES = (
    0,
    21702,
    44799,
)

PRECLIP_SNR_TOLERANCE_DB = 1.0e-8


def main() -> None:
    print("=" * 78)
    print("DETERMINISTIC RECEIVER-NOISE CHECK")
    print("=" * 78)

    print("Source:", H5_PATH)
    print("Global noise seed:", GLOBAL_NOISE_SEED)
    print("SNR levels:", list(SUPPORTED_SNR_DB))

    if not H5_PATH.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {H5_PATH}"
        )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_lines = []

    report_lines.append(
        "Deterministic receiver-noise validation report"
    )

    report_lines.append(
        f"Source: {H5_PATH}"
    )

    report_lines.append(
        f"Global noise seed: {GLOBAL_NOISE_SEED}"
    )

    report_lines.append(
        f"SNR levels: {list(SUPPORTED_SNR_DB)}"
    )

    report_lines.append("")

    all_ok = True

    with h5py.File(
        H5_PATH,
        "r",
    ) as h5:
        intensity_ds = h5["intensity"]

        for sample_index in (
            SELECTED_SAMPLE_INDICES
        ):
            clean = np.asarray(
                intensity_ds[sample_index],
                dtype=np.float64,
            )

            clean_copy = clean.copy()

            report_lines.append(
                f"Sample index: {sample_index}"
            )

            seeds = []

            for snr_db in SUPPORTED_SNR_DB:
                observation_a = (
                    add_deterministic_awgn(
                        clean_intensity=clean,
                        sample_index=sample_index,
                        snr_db=snr_db,
                    )
                )

                observation_b = (
                    add_deterministic_awgn(
                        clean_intensity=clean,
                        sample_index=sample_index,
                        snr_db=snr_db,
                    )
                )

                reproducible = bool(
                    np.array_equal(
                        observation_a.intensity,
                        observation_b.intensity,
                    )
                )

                finite = bool(
                    np.all(
                        np.isfinite(
                            observation_a.intensity
                        )
                    )
                )

                nonnegative = bool(
                    np.all(
                        observation_a.intensity
                        >= 0.0
                    )
                )

                preclip_error_db = abs(
                    observation_a
                    .measured_snr_preclip_db
                    - snr_db
                )

                snr_ok = (
                    preclip_error_db
                    <= PRECLIP_SNR_TOLERANCE_DB
                )

                seeds.append(
                    observation_a.noise_seed
                )

                report_lines.append(
                    (
                        f"  {observation_id(sample_index, snr_db)}: "
                        f"seed={observation_a.noise_seed}, "
                        f"preclip={observation_a.measured_snr_preclip_db:.10f} dB, "
                        f"postclip={observation_a.measured_snr_postclip_db:.10f} dB, "
                        f"clipped_fraction="
                        f"{observation_a.clipped_pixel_fraction:.8f}, "
                        f"reproducible={reproducible}, "
                        f"finite={finite}, "
                        f"nonnegative={nonnegative}"
                    )
                )

                if not (
                    reproducible
                    and finite
                    and nonnegative
                    and snr_ok
                ):
                    all_ok = False

                    report_lines.append(
                        "    [FAIL] Observation validation failed."
                    )

            if len(set(seeds)) != len(
                SUPPORTED_SNR_DB
            ):
                all_ok = False

                report_lines.append(
                    "  [FAIL] Duplicate noise seeds occurred "
                    "across SNR levels."
                )

            source_unchanged = bool(
                np.array_equal(
                    clean,
                    clean_copy,
                )
            )

            report_lines.append(
                f"  source_unchanged={source_unchanged}"
            )

            if not source_unchanged:
                all_ok = False

                report_lines.append(
                    "  [FAIL] Clean source image was modified."
                )

            report_lines.append("")

    status = (
        "PASS"
        if all_ok
        else "FAIL"
    )

    report_lines.append("=" * 78)
    report_lines.append(
        f"RECEIVER-NOISE STATUS: {status}"
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