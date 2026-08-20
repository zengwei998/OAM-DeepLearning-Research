"""
Direct angular Fourier baseline for discrete OAM-state recognition.

The current field definition is:
    U_l + exp(i * phi) * U_-l

Therefore, the intensity contains the angular harmonic:
    h = 2*l
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np

from src.algorithms.harmonic_fit import (
    DEFAULT_EPSILON,
    DEFAULT_OAM_ORDERS,
    DEFAULT_PHASE_BINS,
    quantize_phase,
    wrap_phase,
)


@dataclass(frozen=True)
class DAFCandidate:
    oam_order: int
    harmonic_order: int
    cosine_coefficient: float
    sine_coefficient: float
    amplitude: float
    phase_rad: float
    score: float


@dataclass(frozen=True)
class DAFRecognition:
    predicted_order: int
    predicted_phase_bin: int
    predicted_label: int
    predicted_phase_rad: float
    confidence: float
    harmonic_margin: float
    best_score: float
    second_best_score: float
    candidates: Dict[int, DAFCandidate]


def validate_daf_inputs(
    theta: np.ndarray,
    angular_profile: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    theta_array = np.asarray(
        theta,
        dtype=np.float64,
    )

    profile = np.asarray(
        angular_profile,
        dtype=np.float64,
    )

    if theta_array.ndim != 1:
        raise ValueError(
            "theta must be one-dimensional."
        )

    if profile.ndim != 1:
        raise ValueError(
            "angular_profile must be one-dimensional."
        )

    if theta_array.shape != profile.shape:
        raise ValueError(
            "theta and angular_profile must have identical shapes."
        )

    if not np.all(np.isfinite(theta_array)):
        raise ValueError(
            "theta contains NaN or Inf."
        )

    if not np.all(np.isfinite(profile)):
        raise ValueError(
            "angular_profile contains NaN or Inf."
        )

    if len(theta_array) < 8:
        raise ValueError(
            "Too few angular samples."
        )

    return theta_array, profile


def calculate_fourier_candidate(
    theta: np.ndarray,
    angular_profile: np.ndarray,
    oam_order: int,
) -> DAFCandidate:
    if oam_order <= 0:
        raise ValueError(
            f"oam_order must be positive, got {oam_order}."
        )

    harmonic_order = 2 * int(oam_order)

    cosine_basis = np.cos(
        harmonic_order * theta
    )

    sine_basis = np.sin(
        harmonic_order * theta
    )

    normalization = max(
        float(len(theta)),
        1.0,
    )

    cosine_coefficient = float(
        2.0
        * np.sum(
            angular_profile
            * cosine_basis,
            dtype=np.float64,
        )
        / normalization
    )

    sine_coefficient = float(
        2.0
        * np.sum(
            angular_profile
            * sine_basis,
            dtype=np.float64,
        )
        / normalization
    )

    amplitude = float(
        np.hypot(
            cosine_coefficient,
            sine_coefficient,
        )
    )

    phase_rad = wrap_phase(
        np.arctan2(
            sine_coefficient,
            cosine_coefficient,
        )
    )

    return DAFCandidate(
        oam_order=int(oam_order),
        harmonic_order=int(harmonic_order),
        cosine_coefficient=cosine_coefficient,
        sine_coefficient=sine_coefficient,
        amplitude=amplitude,
        phase_rad=phase_rad,
        score=amplitude,
    )


def recognize_daf_state(
    theta: np.ndarray,
    angular_profile: np.ndarray,
    *,
    candidate_orders: Iterable[int] = DEFAULT_OAM_ORDERS,
    phase_bins: int = DEFAULT_PHASE_BINS,
) -> DAFRecognition:
    theta_array, profile = validate_daf_inputs(
        theta=theta,
        angular_profile=angular_profile,
    )

    profile = profile - float(
        np.mean(
            profile,
            dtype=np.float64,
        )
    )

    candidate_order_list = tuple(
        int(order)
        for order in candidate_orders
    )

    if len(candidate_order_list) < 2:
        raise ValueError(
            "At least two candidate OAM orders are required."
        )

    candidates: Dict[int, DAFCandidate] = {}

    for order in candidate_order_list:
        candidates[order] = calculate_fourier_candidate(
            theta=theta_array,
            angular_profile=profile,
            oam_order=order,
        )

    ranked = sorted(
        candidates.values(),
        key=lambda item: item.score,
        reverse=True,
    )

    best = ranked[0]
    second = ranked[1]

    predicted_phase_bin = quantize_phase(
        phase_rad=best.phase_rad,
        number_of_bins=phase_bins,
    )

    predicted_label = (
        (best.oam_order - 1)
        * phase_bins
        + predicted_phase_bin
    )

    harmonic_margin = float(
        (
            best.score
            - second.score
        )
        / max(
            best.score,
            DEFAULT_EPSILON,
        )
    )

    confidence = float(
        np.clip(
            harmonic_margin,
            0.0,
            1.0,
        )
    )

    return DAFRecognition(
        predicted_order=best.oam_order,
        predicted_phase_bin=predicted_phase_bin,
        predicted_label=predicted_label,
        predicted_phase_rad=best.phase_rad,
        confidence=confidence,
        harmonic_margin=harmonic_margin,
        best_score=best.score,
        second_best_score=second.score,
        candidates=candidates,
    )