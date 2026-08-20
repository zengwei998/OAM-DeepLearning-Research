"""
Visibility-weighted harmonic least-squares fitting for discretized
OAM superposition-state recognition.

Current state definition:
    U_l + exp(i * phi) * U_-l

Therefore, the intensity angular harmonic is 2*l rather than l.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


DEFAULT_OAM_ORDERS = (1, 2, 3, 4)
DEFAULT_PHASE_BINS = 8
DEFAULT_REGULARIZATION = 1.0e-6
DEFAULT_EPSILON = 1.0e-12


@dataclass(frozen=True)
class HarmonicCandidate:
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
class HarmonicRecognition:
    predicted_order: int
    predicted_phase_bin: int
    predicted_label: int
    predicted_phase_rad: float
    confidence: float
    harmonic_margin: float
    best_score: float
    second_best_score: float
    valid_fraction: float
    mean_visibility: float
    candidates: Dict[int, HarmonicCandidate]


def wrap_phase(
    phase_rad: float,
) -> float:
    return float(
        np.mod(
            float(phase_rad),
            2.0 * np.pi,
        )
    )


def phase_bin_center(
    phase_bin: int,
    number_of_bins: int = DEFAULT_PHASE_BINS,
) -> float:
    if not 0 <= phase_bin < number_of_bins:
        raise ValueError(
            f"phase_bin must be in [0, {number_of_bins - 1}]."
        )

    return float(
        2.0
        * np.pi
        * phase_bin
        / number_of_bins
    )


def quantize_phase(
    phase_rad: float,
    number_of_bins: int = DEFAULT_PHASE_BINS,
) -> int:
    phase = wrap_phase(
        phase_rad
    )

    bin_width = (
        2.0 * np.pi
        / number_of_bins
    )

    phase_bin = int(
        np.floor(
            phase / bin_width + 0.5
        )
    ) % number_of_bins

    return phase_bin


def build_design_matrix(
    theta: np.ndarray,
    harmonic_order: int,
) -> np.ndarray:
    theta_array = np.asarray(
        theta,
        dtype=np.float64,
    )

    return np.column_stack(
        [
            np.ones_like(theta_array),
            np.cos(
                harmonic_order
                * theta_array
            ),
            np.sin(
                harmonic_order
                * theta_array
            ),
        ]
    )


def validate_profile_inputs(
    theta: np.ndarray,
    angular_profile: np.ndarray,
    angular_visibility: np.ndarray,
    valid_angles: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    theta_array = np.asarray(
        theta,
        dtype=np.float64,
    )

    profile = np.asarray(
        angular_profile,
        dtype=np.float64,
    )

    visibility = np.asarray(
        angular_visibility,
        dtype=np.float64,
    )

    valid = np.asarray(
        valid_angles,
        dtype=bool,
    )

    if not (
        theta_array.shape
        == profile.shape
        == visibility.shape
        == valid.shape
    ):
        raise ValueError(
            "theta, angular_profile, angular_visibility, "
            "and valid_angles must have identical shapes."
        )

    if theta_array.ndim != 1:
        raise ValueError(
            "All harmonic-fitting arrays must be one-dimensional."
        )

    if not np.all(
        np.isfinite(theta_array)
    ):
        raise ValueError(
            "theta contains NaN or Inf."
        )

    if not np.all(
        np.isfinite(profile)
    ):
        raise ValueError(
            "angular_profile contains NaN or Inf."
        )

    if not np.all(
        np.isfinite(visibility)
    ):
        raise ValueError(
            "angular_visibility contains NaN or Inf."
        )

    if np.any(
        visibility < -DEFAULT_EPSILON
    ) or np.any(
        visibility > 1.0 + DEFAULT_EPSILON
    ):
        raise ValueError(
            "angular_visibility must lie in [0, 1]."
        )

    if np.count_nonzero(valid) < 8:
        raise ValueError(
            "Too few valid angular samples for harmonic fitting."
        )

    return (
        theta_array,
        profile,
        visibility,
        valid,
    )


def fit_single_harmonic(
    theta: np.ndarray,
    angular_profile: np.ndarray,
    angular_visibility: np.ndarray,
    valid_angles: np.ndarray,
    oam_order: int,
    *,
    regularization: float = DEFAULT_REGULARIZATION,
    weight_power: float = 2.0,
) -> HarmonicCandidate:
    (
        theta_array,
        profile,
        visibility,
        valid,
    ) = validate_profile_inputs(
        theta=theta,
        angular_profile=angular_profile,
        angular_visibility=angular_visibility,
        valid_angles=valid_angles,
    )

    if oam_order <= 0:
        raise ValueError(
            f"oam_order must be positive, got {oam_order}."
        )

    harmonic_order = (
        2 * int(oam_order)
    )

    theta_valid = theta_array[
        valid
    ]

    profile_valid = profile[
        valid
    ]

    visibility_valid = visibility[
        valid
    ]

    weights = np.maximum(
        visibility_valid,
        DEFAULT_EPSILON,
    ) ** float(weight_power)

    design = build_design_matrix(
        theta=theta_valid,
        harmonic_order=harmonic_order,
    )

    sqrt_weights = np.sqrt(
        weights
    )

    weighted_design = (
        design
        * sqrt_weights[:, None]
    )

    weighted_target = (
        profile_valid
        * sqrt_weights
    )

    normal_matrix = (
        weighted_design.T
        @ weighted_design
    )

    regularizer = np.eye(
        normal_matrix.shape[0],
        dtype=np.float64,
    )

    regularizer[0, 0] = 0.0

    normal_matrix_regularized = (
        normal_matrix
        + float(regularization)
        * regularizer
    )

    right_hand_side = (
        weighted_design.T
        @ weighted_target
    )

    try:
        coefficients = np.linalg.solve(
            normal_matrix_regularized,
            right_hand_side,
        )
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(
            normal_matrix_regularized,
            right_hand_side,
            rcond=None,
        )[0]

    prediction = (
        design
        @ coefficients
    )

    residual = (
        profile_valid
        - prediction
    )

    weight_sum = float(
        np.sum(
            weights,
            dtype=np.float64,
        )
    )

    weighted_residual_mse = float(
        np.sum(
            weights
            * residual ** 2,
            dtype=np.float64,
        )
        / max(
            weight_sum,
            DEFAULT_EPSILON,
        )
    )

    weighted_mean = float(
        np.average(
            profile_valid,
            weights=weights,
        )
    )

    weighted_signal_variance = float(
        np.sum(
            weights
            * (
                profile_valid
                - weighted_mean
            ) ** 2,
            dtype=np.float64,
        )
        / max(
            weight_sum,
            DEFAULT_EPSILON,
        )
    )

    normalized_residual = float(
        weighted_residual_mse
        / max(
            weighted_signal_variance,
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
            weighted_residual_mse
            + DEFAULT_EPSILON
        )
    )

    condition_number = float(
        np.linalg.cond(
            normal_matrix_regularized
        )
    )

    return HarmonicCandidate(
        oam_order=int(oam_order),
        harmonic_order=int(harmonic_order),
        intercept=float(
            coefficients[0]
        ),
        cosine_coefficient=cosine_coefficient,
        sine_coefficient=sine_coefficient,
        amplitude=amplitude,
        phase_rad=phase_rad,
        residual_mse=weighted_residual_mse,
        normalized_residual=normalized_residual,
        score=score,
        condition_number=condition_number,
    )


def recognize_harmonic_state(
    theta: np.ndarray,
    angular_profile: np.ndarray,
    angular_visibility: np.ndarray,
    valid_angles: np.ndarray,
    *,
    candidate_orders: Iterable[int] = DEFAULT_OAM_ORDERS,
    phase_bins: int = DEFAULT_PHASE_BINS,
    regularization: float = DEFAULT_REGULARIZATION,
    weight_power: float = 2.0,
) -> HarmonicRecognition:
    candidate_order_list = tuple(
        int(order)
        for order in candidate_orders
    )

    if len(candidate_order_list) < 2:
        raise ValueError(
            "At least two candidate OAM orders are required."
        )

    if len(
        set(
            candidate_order_list
        )
    ) != len(
        candidate_order_list
    ):
        raise ValueError(
            "candidate_orders contains duplicate values."
        )

    candidates: Dict[
        int,
        HarmonicCandidate,
    ] = {}

    for oam_order in candidate_order_list:
        candidates[
            oam_order
        ] = fit_single_harmonic(
            theta=theta,
            angular_profile=angular_profile,
            angular_visibility=angular_visibility,
            valid_angles=valid_angles,
            oam_order=oam_order,
            regularization=regularization,
            weight_power=weight_power,
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
        (
            best.oam_order
            - 1
        )
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

    valid_array = np.asarray(
        valid_angles,
        dtype=bool,
    )

    visibility_array = np.asarray(
        angular_visibility,
        dtype=np.float64,
    )

    visibility_confidence = float(
        np.mean(
            visibility_array[
                valid_array
            ]
        )
    )

    confidence = float(
        np.clip(
            harmonic_margin
            * residual_confidence
            * visibility_confidence,
            0.0,
            1.0,
        )
    )

    valid_fraction = float(
        np.mean(
            valid_array
        )
    )

    mean_visibility = float(
        np.mean(
            visibility_array
        )
    )

    return HarmonicRecognition(
        predicted_order=best.oam_order,
        predicted_phase_bin=predicted_phase_bin,
        predicted_label=predicted_label,
        predicted_phase_rad=best.phase_rad,
        confidence=confidence,
        harmonic_margin=harmonic_margin,
        best_score=best.score,
        second_best_score=second.score,
        valid_fraction=valid_fraction,
        mean_visibility=mean_visibility,
        candidates=candidates,
    )