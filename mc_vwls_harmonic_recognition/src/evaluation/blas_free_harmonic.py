"""
BLAS-free harmonic fitting used for failure-case reconstruction.

This module preserves the public recognition result format from
src.algorithms.harmonic_fit, but avoids NumPy matrix multiplication
and LAPACK calls that trigger a native Windows crash in the current
environment.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from src.algorithms.harmonic_fit import (
    DEFAULT_EPSILON,
    DEFAULT_OAM_ORDERS,
    DEFAULT_PHASE_BINS,
    DEFAULT_REGULARIZATION,
    HarmonicCandidate,
    HarmonicRecognition,
    build_design_matrix,
    quantize_phase,
    validate_profile_inputs,
    wrap_phase,
)


def build_normal_equations(
    weighted_design: np.ndarray,
    weighted_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.asarray(
        weighted_design,
        dtype=np.float64,
    )

    target = np.asarray(
        weighted_target,
        dtype=np.float64,
    )

    if design.ndim != 2 or design.shape[1] != 3:
        raise ValueError(
            "weighted_design must have shape (N, 3)."
        )

    if target.shape != (design.shape[0],):
        raise ValueError(
            "weighted_target shape does not match weighted_design."
        )

    column_0 = design[:, 0]
    column_1 = design[:, 1]
    column_2 = design[:, 2]

    normal_matrix = np.empty(
        (3, 3),
        dtype=np.float64,
    )

    normal_matrix[0, 0] = float(
        np.sum(
            column_0 * column_0,
            dtype=np.float64,
        )
    )

    normal_matrix[0, 1] = float(
        np.sum(
            column_0 * column_1,
            dtype=np.float64,
        )
    )

    normal_matrix[0, 2] = float(
        np.sum(
            column_0 * column_2,
            dtype=np.float64,
        )
    )

    normal_matrix[1, 0] = normal_matrix[0, 1]

    normal_matrix[1, 1] = float(
        np.sum(
            column_1 * column_1,
            dtype=np.float64,
        )
    )

    normal_matrix[1, 2] = float(
        np.sum(
            column_1 * column_2,
            dtype=np.float64,
        )
    )

    normal_matrix[2, 0] = normal_matrix[0, 2]
    normal_matrix[2, 1] = normal_matrix[1, 2]

    normal_matrix[2, 2] = float(
        np.sum(
            column_2 * column_2,
            dtype=np.float64,
        )
    )

    right_hand_side = np.asarray(
        [
            np.sum(
                column_0 * target,
                dtype=np.float64,
            ),
            np.sum(
                column_1 * target,
                dtype=np.float64,
            ),
            np.sum(
                column_2 * target,
                dtype=np.float64,
            ),
        ],
        dtype=np.float64,
    )

    return (
        normal_matrix,
        right_hand_side,
    )


def solve_3x3(
    matrix: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    matrix_array = np.asarray(
        matrix,
        dtype=np.float64,
    )

    vector_array = np.asarray(
        vector,
        dtype=np.float64,
    )

    if matrix_array.shape != (3, 3):
        raise ValueError(
            "matrix must have shape (3, 3)."
        )

    if vector_array.shape != (3,):
        raise ValueError(
            "vector must have shape (3,)."
        )

    augmented = [
        [
            float(
                matrix_array[
                    row_index,
                    column_index,
                ]
            )
            for column_index in range(3)
        ]
        + [
            float(
                vector_array[
                    row_index
                ]
            )
        ]
        for row_index in range(3)
    ]

    scale = max(
        max(
            abs(value)
            for value in row[:3]
        )
        for row in augmented
    )

    singular_threshold = max(
        DEFAULT_EPSILON * scale,
        1.0e-15,
    )

    for pivot_index in range(3):
        pivot_row = max(
            range(
                pivot_index,
                3,
            ),
            key=lambda row_index: abs(
                augmented[
                    row_index
                ][
                    pivot_index
                ]
            ),
        )

        pivot_value = augmented[
            pivot_row
        ][
            pivot_index
        ]

        if abs(pivot_value) <= singular_threshold:
            raise ArithmeticError(
                "The 3x3 normal matrix is singular."
            )

        if pivot_row != pivot_index:
            (
                augmented[pivot_index],
                augmented[pivot_row],
            ) = (
                augmented[pivot_row],
                augmented[pivot_index],
            )

        pivot_value = augmented[
            pivot_index
        ][
            pivot_index
        ]

        for column_index in range(
            pivot_index,
            4,
        ):
            augmented[
                pivot_index
            ][
                column_index
            ] /= pivot_value

        for row_index in range(3):
            if row_index == pivot_index:
                continue

            elimination_factor = augmented[
                row_index
            ][
                pivot_index
            ]

            for column_index in range(
                pivot_index,
                4,
            ):
                augmented[
                    row_index
                ][
                    column_index
                ] -= (
                    elimination_factor
                    * augmented[
                        pivot_index
                    ][
                        column_index
                    ]
                )

    return np.asarray(
        [
            augmented[row_index][3]
            for row_index in range(3)
        ],
        dtype=np.float64,
    )


def predict_without_matmul(
    design: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    design_array = np.asarray(
        design,
        dtype=np.float64,
    )

    coefficient_array = np.asarray(
        coefficients,
        dtype=np.float64,
    )

    if design_array.ndim != 2 or design_array.shape[1] != 3:
        raise ValueError(
            "design must have shape (N, 3)."
        )

    if coefficient_array.shape != (3,):
        raise ValueError(
            "coefficients must have shape (3,)."
        )

    return (
        design_array[:, 0]
        * coefficient_array[0]
        + design_array[:, 1]
        * coefficient_array[1]
        + design_array[:, 2]
        * coefficient_array[2]
    )


def matrix_infinity_norm(
    matrix: np.ndarray,
) -> float:
    matrix_array = np.asarray(
        matrix,
        dtype=np.float64,
    )

    return float(
        max(
            np.sum(
                np.abs(
                    matrix_array[
                        row_index
                    ]
                ),
                dtype=np.float64,
            )
            for row_index in range(
                matrix_array.shape[0]
            )
        )
    )


def estimate_condition_number(
    matrix: np.ndarray,
) -> float:
    matrix_array = np.asarray(
        matrix,
        dtype=np.float64,
    )

    identity_columns = (
        np.asarray(
            [1.0, 0.0, 0.0],
            dtype=np.float64,
        ),
        np.asarray(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        ),
        np.asarray(
            [0.0, 0.0, 1.0],
            dtype=np.float64,
        ),
    )

    inverse_columns = [
        solve_3x3(
            matrix_array,
            column,
        )
        for column in identity_columns
    ]

    inverse_matrix = np.column_stack(
        inverse_columns
    )

    return float(
        matrix_infinity_norm(
            matrix_array
        )
        * matrix_infinity_norm(
            inverse_matrix
        )
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

    harmonic_order = 2 * int(
        oam_order
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
    ) ** float(
        weight_power
    )

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

    (
        normal_matrix,
        right_hand_side,
    ) = build_normal_equations(
        weighted_design=weighted_design,
        weighted_target=weighted_target,
    )

    normal_matrix_regularized = np.asarray(
        normal_matrix,
        dtype=np.float64,
    ).copy()

    normal_matrix_regularized[
        1,
        1,
    ] += float(
        regularization
    )

    normal_matrix_regularized[
        2,
        2,
    ] += float(
        regularization
    )

    coefficients = solve_3x3(
        matrix=normal_matrix_regularized,
        vector=right_hand_side,
    )

    prediction = predict_without_matmul(
        design=design,
        coefficients=coefficients,
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
            * residual**2,
            dtype=np.float64,
        )
        / max(
            weight_sum,
            DEFAULT_EPSILON,
        )
    )

    weighted_mean = float(
        np.sum(
            weights
            * profile_valid,
            dtype=np.float64,
        )
        / max(
            weight_sum,
            DEFAULT_EPSILON,
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

    condition_number = estimate_condition_number(
        normal_matrix_regularized
    )

    return HarmonicCandidate(
        oam_order=int(
            oam_order
        ),
        harmonic_order=int(
            harmonic_order
        ),
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