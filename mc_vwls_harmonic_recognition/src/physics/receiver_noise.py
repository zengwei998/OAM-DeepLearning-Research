"""
Deterministic receiver-noise model for OAM intensity observations.

Primary benchmark:
    Additive white Gaussian noise (AWGN) with controlled pre-clipping SNR.

The noise realization is determined uniquely by:
    - clean sample index
    - target SNR
    - global noise seed

This guarantees that all recognition methods receive exactly the same
noisy observation.

Negative detector values are clipped to zero after noise addition.
Both pre-clipping and post-clipping measured SNR values are returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


GLOBAL_NOISE_SEED = 20260804

SUPPORTED_SNR_DB = (
    20.0,
    15.0,
    10.0,
    5.0,
    0.0,
)

POWER_EPSILON = 1.0e-20


@dataclass(frozen=True)
class ReceiverObservation:
    """
    Result of deterministic receiver observation generation.
    """

    intensity: np.ndarray
    target_snr_db: float
    measured_snr_preclip_db: float
    measured_snr_postclip_db: float
    noise_seed: int
    clipped_pixel_fraction: float
    signal_power: float
    preclip_noise_power: float
    postclip_error_power: float


def snr_to_integer_code(
    snr_db: float,
) -> int:
    """
    Convert SNR in dB to a stable integer seed component.

    Multiplication by 1000 also supports future fractional SNR values.
    """

    return int(
        round(
            float(snr_db) * 1000.0
        )
    )


def deterministic_noise_seed(
    sample_index: int,
    snr_db: float,
    global_seed: int = GLOBAL_NOISE_SEED,
) -> int:
    """
    Construct one deterministic uint32 noise seed.
    """

    if sample_index < 0:
        raise ValueError(
            f"sample_index must be nonnegative, got {sample_index}."
        )

    seed_sequence = np.random.SeedSequence(
        [
            int(global_seed),
            int(sample_index),
            snr_to_integer_code(snr_db),
        ]
    )

    return int(
        seed_sequence.generate_state(
            1,
            dtype=np.uint32,
        )[0]
    )


def calculate_snr_db(
    signal: np.ndarray,
    error: np.ndarray,
) -> float:
    """
    Calculate image-domain power SNR:

        SNR = 10 log10(mean(signal^2) / mean(error^2))
    """

    signal_array = np.asarray(
        signal,
        dtype=np.float64,
    )

    error_array = np.asarray(
        error,
        dtype=np.float64,
    )

    signal_power = float(
        np.mean(
            signal_array ** 2,
            dtype=np.float64,
        )
    )

    error_power = float(
        np.mean(
            error_array ** 2,
            dtype=np.float64,
        )
    )

    if signal_power <= POWER_EPSILON:
        raise ValueError(
            "Signal power is zero or numerically invalid."
        )

    if error_power <= POWER_EPSILON:
        return float("inf")

    return float(
        10.0
        * np.log10(
            signal_power / error_power
        )
    )


def validate_clean_intensity(
    clean_intensity: np.ndarray,
) -> np.ndarray:
    """
    Validate and convert one clean intensity image.
    """

    image = np.asarray(
        clean_intensity,
        dtype=np.float64,
    )

    if image.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional intensity image, "
            f"found shape {image.shape}."
        )

    if not np.all(
        np.isfinite(image)
    ):
        raise ValueError(
            "Clean intensity contains NaN or Inf."
        )

    minimum = float(
        np.min(image)
    )

    if minimum < -1.0e-12:
        raise ValueError(
            f"Clean intensity contains negative values: min={minimum}."
        )

    # Remove possible tiny negative roundoff.
    image = np.maximum(
        image,
        0.0,
    )

    signal_power = float(
        np.mean(
            image ** 2,
            dtype=np.float64,
        )
    )

    if signal_power <= POWER_EPSILON:
        raise ValueError(
            "Clean intensity has zero image-domain signal power."
        )

    return image


def generate_unit_rms_noise(
    shape: tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate zero-mean Gaussian noise normalized to exactly unit RMS.

    Exact RMS normalization keeps the realized pre-clipping SNR equal
    to the requested value, apart from floating-point roundoff.
    """

    noise = rng.standard_normal(
        size=shape,
        dtype=np.float64,
    )

    noise = noise - float(
        np.mean(
            noise,
            dtype=np.float64,
        )
    )

    rms = float(
        np.sqrt(
            np.mean(
                noise ** 2,
                dtype=np.float64,
            )
        )
    )

    if rms <= POWER_EPSILON:
        raise RuntimeError(
            "Generated Gaussian noise has invalid RMS."
        )

    return noise / rms


def add_deterministic_awgn(
    clean_intensity: np.ndarray,
    sample_index: int,
    snr_db: float,
    *,
    global_seed: int = GLOBAL_NOISE_SEED,
    clip_negative: bool = True,
    output_dtype: np.dtype = np.float32,
) -> ReceiverObservation:
    """
    Add deterministic AWGN to one clean intensity image.

    Noise scaling:

        P_n = P_s / 10^(SNR/10)

    where:

        P_s = mean(I_clean^2)

    The Gaussian realization is first normalized to exactly unit RMS,
    so measured pre-clipping SNR closely matches the target SNR.

    Args:
        clean_intensity:
            Two-dimensional nonnegative clean intensity.

        sample_index:
            Index in occlusion_clean_v2.h5.

        snr_db:
            Requested SNR in dB.

        global_seed:
            Frozen global receiver-noise seed.

        clip_negative:
            When True, negative detector values are set to zero.

        output_dtype:
            Output image dtype.

    Returns:
        ReceiverObservation containing the noisy image and diagnostics.
    """

    clean = validate_clean_intensity(
        clean_intensity
    )

    target_snr_db = float(
        snr_db
    )

    noise_seed = deterministic_noise_seed(
        sample_index=sample_index,
        snr_db=target_snr_db,
        global_seed=global_seed,
    )

    rng = np.random.default_rng(
        noise_seed
    )

    signal_power = float(
        np.mean(
            clean ** 2,
            dtype=np.float64,
        )
    )

    snr_linear = float(
        10.0 ** (
            target_snr_db / 10.0
        )
    )

    if (
        not np.isfinite(snr_linear)
        or snr_linear <= 0.0
    ):
        raise ValueError(
            f"Invalid SNR: {target_snr_db} dB."
        )

    target_noise_power = (
        signal_power / snr_linear
    )

    unit_noise = generate_unit_rms_noise(
        shape=clean.shape,
        rng=rng,
    )

    noise = (
        unit_noise
        * np.sqrt(target_noise_power)
    )

    noisy_preclip = (
        clean + noise
    )

    measured_snr_preclip_db = (
        calculate_snr_db(
            signal=clean,
            error=noise,
        )
    )

    negative_pixels = (
        noisy_preclip < 0.0
    )

    clipped_pixel_fraction = float(
        np.mean(
            negative_pixels
        )
    )

    if clip_negative:
        noisy_output = np.maximum(
            noisy_preclip,
            0.0,
        )
    else:
        noisy_output = noisy_preclip

    postclip_error = (
        noisy_output - clean
    )

    measured_snr_postclip_db = (
        calculate_snr_db(
            signal=clean,
            error=postclip_error,
        )
    )

    preclip_noise_power = float(
        np.mean(
            noise ** 2,
            dtype=np.float64,
        )
    )

    postclip_error_power = float(
        np.mean(
            postclip_error ** 2,
            dtype=np.float64,
        )
    )

    noisy_output = np.asarray(
        noisy_output,
        dtype=output_dtype,
    )

    return ReceiverObservation(
        intensity=noisy_output,
        target_snr_db=target_snr_db,
        measured_snr_preclip_db=measured_snr_preclip_db,
        measured_snr_postclip_db=measured_snr_postclip_db,
        noise_seed=noise_seed,
        clipped_pixel_fraction=clipped_pixel_fraction,
        signal_power=signal_power,
        preclip_noise_power=preclip_noise_power,
        postclip_error_power=postclip_error_power,
    )


def add_awgn_by_level(
    clean_intensity: np.ndarray,
    sample_index: int,
    snr_level_index: int,
    *,
    global_seed: int = GLOBAL_NOISE_SEED,
    clip_negative: bool = True,
) -> ReceiverObservation:
    """
    Convenience interface using the frozen five-level SNR list.
    """

    if not 0 <= snr_level_index < len(
        SUPPORTED_SNR_DB
    ):
        raise IndexError(
            f"snr_level_index must be between 0 and "
            f"{len(SUPPORTED_SNR_DB) - 1}, "
            f"got {snr_level_index}."
        )

    return add_deterministic_awgn(
        clean_intensity=clean_intensity,
        sample_index=sample_index,
        snr_db=SUPPORTED_SNR_DB[
            snr_level_index
        ],
        global_seed=global_seed,
        clip_negative=clip_negative,
    )


def observation_id(
    sample_index: int,
    snr_db: float,
) -> str:
    """
    Return a stable human-readable logical observation ID.
    """

    return (
        f"sample_{int(sample_index):05d}"
        f"_snr_{float(snr_db):+06.1f}dB"
    )