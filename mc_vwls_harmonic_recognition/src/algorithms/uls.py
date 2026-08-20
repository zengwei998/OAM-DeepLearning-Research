"""
Unweighted least-squares harmonic baseline for discrete OAM-state
recognition.

The current field definition is:
    U_l + exp(i * phi) * U_-l

Therefore, the intensity harmonic is:
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
class ULSCandidate:
    oam_order: int
    harmonic_order: int
    intercept: float
    cosine_coefficient: float
    sine_coefficient: float
    amplitude: float
    phase_rad: float
    residual_mse: float
    normalized_residual: float
    score: float
    condition_number: float


@dataclass(frozen=True)
class ULSRecognition:
    predicted_order: int
    predicted_phase_bin: int
    predicted_label: int
    predicted_phase_rad: float
    confidence: float
    harmonic_margin: float
    best_score: float
    second_best_score: float
    candidates: Dict[int, ULSCandidate]


def validate_uls_inputs(
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


def fit_single_uls_candidate(
    theta: np.ndarray,
    angular_profile: np.ndarray,
    oam_order: int,
) -> ULSCandidate:
    if oam_order <= 0:
        raise ValueError(
            f"oam_order must be positive, got {oam_order}."
        )

    harmonic_order = 2 * int(oam_order)

    design = np.column_stack(
        [
            np.ones_like(theta),
            np.cos(
                harmonic_order * theta
            ),
            np.sin(
                harmonic_order * theta
            ),
        ]
    )

    coefficients = np.linalg.lstsq(
        design,
        angular_profile,
        rcond=None,
    )[0]

    prediction = design @ coefficients
    residual = angular_profile - prediction

    residual_mse = float(
        np.mean(
            residual ** 2,
            dtype=np.float64,
        )
    )

    signal_variance = float(
        np.var(
            angular_profile,
            dtype=np.float64,
        )
    )

    normalized_residual = float(
        residual_mse
        / max(
            signal_variance,
            DEFAULT_EPSILON,
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

    phase_rad = wrap_phase(
        np.arctan2(
            sine_coefficient,
            cosine_coefficient,
        )
    )

    score = float(
        amplitude
        / np.sqrt(
            residual_mse
            + DEFAULT_EPSILON,
        )
    )

    condition_number = float(
        np.linalg.cond(
            design.T @ design
        )
    )

    return ULSCandidate(
        oam_order=int(oam_order),
        harmonic_order=int(harmonic_order),
        intercept=float(coefficients[0]),
        cosine_coefficient=cosine_coefficient,
        sine_coefficient=sine_coefficient,
        amplitude=amplitude,
        phase_rad=phase_rad,
        residual_mse=residual_mse,
        normalized_residual=normalized_residual,
        score=score,
        condition_number=condition_number,
    )


def recognize_uls_state(
    theta: np.ndarray,
    angular_profile: np.ndarray,
    *,
    candidate_orders: Iterable[int] = DEFAULT_OAM_ORDERS,
    phase_bins: int = DEFAULT_PHASE_BINS,
) -> ULSRecognition:
    theta_array, profile = validate_uls_inputs(
        theta=theta,
        angular_profile=angular_profile,
    )

    candidate_order_list = tuple(
        int(order)
        for order in candidate_orders
    )

    if len(candidate_order_list) < 2:
        raise ValueError(
            "At least two candidate OAM orders are required."
        )

    candidates: Dict[int, ULSCandidate] = {}

    for order in candidate_order_list:
        candidates[order] = fit_single_uls_candidate(
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

    residual_confidence = float(
        1.0
        / (
            1.0
            + best.normalized_residual
        )
    )

    confidence = float(
        np.clip(
            harmonic_margin
            * residual_confidence,
            0.0,
            1.0,
        )
    )

    return ULSRecognition(
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