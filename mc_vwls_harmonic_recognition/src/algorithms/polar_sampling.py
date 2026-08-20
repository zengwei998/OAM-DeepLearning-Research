"""
Polar sampling and mask-aware radial visibility normalization.

This module converts a two-dimensional receiver intensity image and its
calibrated visibility mask into a one-dimensional angular profile for
harmonic recognition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates


DEFAULT_ANGULAR_SAMPLES = 180
DEFAULT_RADIAL_SAMPLES = 64
DEFAULT_VISIBILITY_THRESHOLD = 0.05
DEFAULT_EPSILON = 1.0e-12


@dataclass(frozen=True)
class PolarProfile:
    """Mask-aware polar representation of one receiver observation."""

    theta: np.ndarray
    radius: np.ndarray
    polar_intensity: np.ndarray
    polar_mask: np.ndarray
    angular_profile: np.ndarray
    angular_visibility: np.ndarray
    valid_angles: np.ndarray
    center_x: float
    center_y: float
    maximum_radius: float


def validate_inputs(
    intensity: np.ndarray,
    visible_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and convert intensity and visibility mask."""

    image = np.asarray(
        intensity,
        dtype=np.float64,
    )

    mask = np.asarray(
        visible_mask,
        dtype=np.float64,
    )

    if image.ndim != 2:
        raise ValueError(
            f"Intensity must be two-dimensional, found {image.shape}."
        )

    if mask.shape != image.shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape "
            f"{image.shape}."
        )

    if not np.all(np.isfinite(image)):
        raise ValueError("Intensity contains NaN or Inf.")

    if not np.all(np.isfinite(mask)):
        raise ValueError("Visibility mask contains NaN or Inf.")

    if float(np.min(image)) < -1.0e-12:
        raise ValueError(
            f"Intensity contains negative values: {np.min(image)}."
        )

    image = np.maximum(image, 0.0)

    # Accept binary masks and interpolated masks in [0, 1].
    if float(np.min(mask)) < -1.0e-12:
        raise ValueError("Visibility mask contains values below zero.")

    if float(np.max(mask)) > 1.0 + 1.0e-12:
        raise ValueError("Visibility mask contains values above one.")

    mask = np.clip(mask, 0.0, 1.0)

    return image, mask


def default_center(
    shape: Tuple[int, int],
) -> Tuple[float, float]:
    """
    Return the fixed optical-axis center.

    The coordinate convention is:
        x = column coordinate
        y = row coordinate
    """

    height, width = shape

    center_x = (width - 1.0) / 2.0
    center_y = (height - 1.0) / 2.0

    return center_x, center_y


def maximum_inscribed_radius(
    shape: Tuple[int, int],
    center_x: float,
    center_y: float,
) -> float:
    """Return the largest radius remaining completely inside the image."""

    height, width = shape

    return float(
        min(
            center_x,
            center_y,
            width - 1.0 - center_x,
            height - 1.0 - center_y,
        )
    )


def build_polar_coordinates(
    shape: Tuple[int, int],
    *,
    angular_samples: int = DEFAULT_ANGULAR_SAMPLES,
    radial_samples: int = DEFAULT_RADIAL_SAMPLES,
    center_x: Optional[float] = None,
    center_y: Optional[float] = None,
    maximum_radius: Optional[float] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
]:
    """
    Build one polar sampling grid.

    Returns:
        theta,
        radius,
        sample_y,
        sample_x,
        center_x,
        center_y,
        maximum_radius
    """

    if angular_samples < 8:
        raise ValueError("angular_samples must be at least 8.")

    if radial_samples < 2:
        raise ValueError("radial_samples must be at least 2.")

    if center_x is None or center_y is None:
        default_x, default_y = default_center(shape)

        if center_x is None:
            center_x = default_x

        if center_y is None:
            center_y = default_y

    center_x = float(center_x)
    center_y = float(center_y)

    if maximum_radius is None:
        maximum_radius = maximum_inscribed_radius(
            shape=shape,
            center_x=center_x,
            center_y=center_y,
        )

    maximum_radius = float(maximum_radius)

    if maximum_radius <= 0:
        raise ValueError(
            f"maximum_radius must be positive, got {maximum_radius}."
        )

    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        angular_samples,
        endpoint=False,
        dtype=np.float64,
    )

    radius = np.linspace(
        0.0,
        maximum_radius,
        radial_samples,
        endpoint=True,
        dtype=np.float64,
    )

    radius_grid, theta_grid = np.meshgrid(
        radius,
        theta,
        indexing="ij",
    )

    sample_x = (
        center_x
        + radius_grid * np.cos(theta_grid)
    )

    sample_y = (
        center_y
        + radius_grid * np.sin(theta_grid)
    )

    return (
        theta,
        radius,
        sample_y,
        sample_x,
        center_x,
        center_y,
        maximum_radius,
    )


def bilinear_polar_sample(
    image: np.ndarray,
    sample_y: np.ndarray,
    sample_x: np.ndarray,
    *,
    outside_value: float = 0.0,
) -> np.ndarray:
    """Sample one image on a prepared polar grid using bilinear interpolation."""

    coordinates = np.vstack(
        [
            sample_y.ravel(),
            sample_x.ravel(),
        ]
    )

    sampled = map_coordinates(
        np.asarray(image, dtype=np.float64),
        coordinates,
        order=1,
        mode="constant",
        cval=float(outside_value),
        prefilter=False,
    )

    return sampled.reshape(
        sample_y.shape
    )


def radial_visibility_normalization(
    polar_intensity: np.ndarray,
    polar_mask: np.ndarray,
    radius: np.ndarray,
    *,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    epsilon: float = DEFAULT_EPSILON,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate intensity radially and normalize by visible radial support.

    Polar arrays have shape:
        (radial_samples, angular_samples)
    """

    intensity = np.asarray(
        polar_intensity,
        dtype=np.float64,
    )

    mask = np.asarray(
        polar_mask,
        dtype=np.float64,
    )

    radius_array = np.asarray(
        radius,
        dtype=np.float64,
    )

    if intensity.shape != mask.shape:
        raise ValueError(
            "Polar intensity and polar mask shapes do not match."
        )

    if intensity.shape[0] != len(radius_array):
        raise ValueError(
            "Radius length does not match polar radial dimension."
        )

    radial_weights = radius_array[:, None]

    full_support = float(
        np.sum(radial_weights)
    )

    if full_support <= epsilon:
        raise ValueError("Radial support is zero.")

    visible_support = np.sum(
        mask * radial_weights,
        axis=0,
        dtype=np.float64,
    )

    angular_visibility = (
        visible_support / full_support
    )

    weighted_intensity = np.sum(
        intensity * radial_weights,
        axis=0,
        dtype=np.float64,
    )

    angular_profile = np.zeros(
        intensity.shape[1],
        dtype=np.float64,
    )

    valid_angles = (
        angular_visibility
        >= float(visibility_threshold)
    )

    angular_profile[valid_angles] = (
        weighted_intensity[valid_angles]
        / np.maximum(
            visible_support[valid_angles],
            epsilon,
        )
    )

    return (
        angular_profile,
        angular_visibility,
        valid_angles,
    )


def extract_polar_profile(
    intensity: np.ndarray,
    visible_mask: np.ndarray,
    *,
    angular_samples: int = DEFAULT_ANGULAR_SAMPLES,
    radial_samples: int = DEFAULT_RADIAL_SAMPLES,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    center_x: Optional[float] = None,
    center_y: Optional[float] = None,
    maximum_radius: Optional[float] = None,
) -> PolarProfile:
    """
    Extract the complete mask-aware polar representation.
    """

    image, mask = validate_inputs(
        intensity=intensity,
        visible_mask=visible_mask,
    )

    (
        theta,
        radius,
        sample_y,
        sample_x,
        resolved_center_x,
        resolved_center_y,
        resolved_maximum_radius,
    ) = build_polar_coordinates(
        shape=image.shape,
        angular_samples=angular_samples,
        radial_samples=radial_samples,
        center_x=center_x,
        center_y=center_y,
        maximum_radius=maximum_radius,
    )

    polar_intensity = bilinear_polar_sample(
        image=image,
        sample_y=sample_y,
        sample_x=sample_x,
        outside_value=0.0,
    )

    polar_mask = bilinear_polar_sample(
        image=mask,
        sample_y=sample_y,
        sample_x=sample_x,
        outside_value=0.0,
    )

    polar_mask = np.clip(
        polar_mask,
        0.0,
        1.0,
    )

    (
        angular_profile,
        angular_visibility,
        valid_angles,
    ) = radial_visibility_normalization(
        polar_intensity=polar_intensity,
        polar_mask=polar_mask,
        radius=radius,
        visibility_threshold=visibility_threshold,
    )

    return PolarProfile(
        theta=theta,
        radius=radius,
        polar_intensity=polar_intensity,
        polar_mask=polar_mask,
        angular_profile=angular_profile,
        angular_visibility=angular_visibility,
        valid_angles=valid_angles,
        center_x=resolved_center_x,
        center_y=resolved_center_y,
        maximum_radius=resolved_maximum_radius,
    )


def normalize_angular_profile(
    angular_profile: np.ndarray,
    valid_angles: np.ndarray,
    *,
    remove_mean: bool = True,
    unit_norm: bool = False,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """
    Prepare an angular profile for harmonic fitting.

    Invalid angles remain zero.
    """

    profile = np.asarray(
        angular_profile,
        dtype=np.float64,
    ).copy()

    valid = np.asarray(
        valid_angles,
        dtype=bool,
    )

    if profile.shape != valid.shape:
        raise ValueError(
            "Angular profile and valid-angle mask shapes do not match."
        )

    profile[~valid] = 0.0

    if remove_mean and np.any(valid):
        profile[valid] -= float(
            np.mean(
                profile[valid],
                dtype=np.float64,
            )
        )

    if unit_norm and np.any(valid):
        norm = float(
            np.linalg.norm(
                profile[valid]
            )
        )

        if norm > epsilon:
            profile[valid] /= norm

    return profile