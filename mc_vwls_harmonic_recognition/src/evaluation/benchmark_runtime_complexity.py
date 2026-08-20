"""
Benchmark runtime and summarize computational complexity for the four
frozen ablation methods:

    A0_DAF
    A1_MASK_ULS
    A2_RAW_VWLS
    A3_MC_VWLS

The current Windows environment has a native BLAS/LAPACK conflict for
specific small matrix operations. Therefore, ULS and VWLS recognition
use the validated BLAS-free harmonic implementation. The mathematical
model and frozen parameters remain unchanged.

Timing definitions:
    recognition_time:
        Recognition after the polar representation and angular profiles
        have already been constructed.

    end_to_end_time:
        Deterministic receiver noise, polar sampling, angular-profile
        construction, and recognition.

Outputs:
    results/csv/mc_vwls_runtime_benchmark.csv
    results/csv/mc_vwls_complexity_summary.csv
    results/figures/fig_mc_vwls_runtime.png
    results/figures/fig_mc_vwls_runtime.pdf
    results/validation/mc_vwls_runtime_complexity_report.txt
"""

from __future__ import annotations

import csv
import gc
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.algorithms.daf import recognize_daf_state
from src.algorithms.polar_sampling import (
    extract_polar_profile,
    normalize_angular_profile,
)
from src.evaluation.blas_free_harmonic import (
    recognize_harmonic_state,
)
from src.physics.receiver_noise import (
    SUPPORTED_SNR_DB,
    add_deterministic_awgn,
)


ROOT = Path(__file__).resolve().parents[2]

H5_PATH = (
    ROOT
    / "data"
    / "generated"
    / "occlusion_clean_v2.h5"
)

SPLIT_PATH = (
    ROOT
    / "data"
    / "manifest"
    / "sample_split_v1.npz"
)

RUNTIME_CSV_PATH = (
    ROOT
    / "results"
    / "csv"
    / "mc_vwls_runtime_benchmark.csv"
)

COMPLEXITY_CSV_PATH = (
    ROOT
    / "results"
    / "csv"
    / "mc_vwls_complexity_summary.csv"
)

PNG_PATH = (
    ROOT
    / "results"
    / "figures"
    / "fig_mc_vwls_runtime.png"
)

PDF_PATH = (
    ROOT
    / "results"
    / "figures"
    / "fig_mc_vwls_runtime.pdf"
)

REPORT_PATH = (
    ROOT
    / "results"
    / "validation"
    / "mc_vwls_runtime_complexity_report.txt"
)

PREDICTION_PATH = (
    ROOT
    / "results"
    / "csv"
    / "mc_vwls_ablation_full_test_predictions.csv"
)

BENCHMARK_SAMPLE_COUNT = 256
BENCHMARK_SEED = 20260804
WARMUP_COUNT = 16
REPEAT_COUNT = 5

ANGULAR_SAMPLES = 180
RADIAL_SAMPLES = 64
PHASE_BINS = 8

CANDIDATE_ORDERS = (
    1,
    2,
    3,
    4,
)

VISIBILITY_THRESHOLD = 0.0
REGULARIZATION = 0.0
WEIGHT_POWER = 0.5

METHOD_NAMES = (
    "A0_DAF",
    "A1_MASK_ULS",
    "A2_RAW_VWLS",
    "A3_MC_VWLS",
)

METHOD_DISPLAY_NAMES = {
    "A0_DAF": "DAF",
    "A1_MASK_ULS": "Mask-ULS",
    "A2_RAW_VWLS": "Raw-VWLS",
    "A3_MC_VWLS": "MC-VWLS",
}

EPSILON = 1.0e-12


@dataclass(frozen=True)
class BenchmarkObservation:
    sample_index: int
    snr_db: float
    clean_intensity: np.ndarray
    visible_mask: np.ndarray


@dataclass(frozen=True)
class PreparedObservation:
    sample_index: int
    snr_db: float
    theta: np.ndarray
    raw_profile: np.ndarray
    normalized_profile: np.ndarray
    angular_visibility: np.ndarray
    valid_angles: np.ndarray


@dataclass(frozen=True)
class RuntimeSummary:
    method: str
    timing_scope: str
    repeat_count: int
    observation_count: int
    total_times_seconds: tuple[float, ...]
    mean_total_seconds: float
    standard_deviation_total_seconds: float
    mean_time_ms_per_image: float
    standard_deviation_ms_per_image: float
    median_time_ms_per_image: float
    minimum_time_ms_per_image: float
    maximum_time_ms_per_image: float
    throughput_images_per_second: float


def stage(
    message: str,
) -> None:
    print(
        message,
        flush=True,
    )


def load_test_indices() -> np.ndarray:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Split file does not exist: {SPLIT_PATH}"
        )

    split = np.load(
        SPLIT_PATH,
        allow_pickle=False,
    )

    if "sample_split_codes" not in split.files:
        raise KeyError(
            "sample_split_codes was not found in split file."
        )

    split_codes = np.asarray(
        split[
            "sample_split_codes"
        ],
        dtype=np.int64,
    )

    test_indices = np.flatnonzero(
        split_codes == 2
    ).astype(
        np.int64
    )

    if len(test_indices) == 0:
        raise ValueError(
            "No frozen test samples were found."
        )

    return test_indices


def choose_benchmark_indices(
    test_indices: np.ndarray,
) -> np.ndarray:
    if len(test_indices) < BENCHMARK_SAMPLE_COUNT:
        raise ValueError(
            "The frozen test set is smaller than "
            f"BENCHMARK_SAMPLE_COUNT={BENCHMARK_SAMPLE_COUNT}."
        )

    random_generator = np.random.default_rng(
        BENCHMARK_SEED
    )

    selected = random_generator.choice(
        test_indices,
        size=BENCHMARK_SAMPLE_COUNT,
        replace=False,
    )

    return np.sort(
        np.asarray(
            selected,
            dtype=np.int64,
        )
    )


def build_benchmark_observations(
    selected_indices: np.ndarray,
) -> List[BenchmarkObservation]:
    observations: List[
        BenchmarkObservation
    ] = []

    supported_snr = tuple(
        float(value)
        for value in SUPPORTED_SNR_DB
    )

    if not supported_snr:
        raise ValueError(
            "SUPPORTED_SNR_DB is empty."
        )

    with h5py.File(
        H5_PATH,
        "r",
    ) as h5_file:
        intensity_dataset = h5_file[
            "intensity"
        ]

        mask_dataset = h5_file[
            "visible_mask"
        ]

        for position, sample_index in enumerate(
            selected_indices
        ):
            snr_db = supported_snr[
                position
                % len(
                    supported_snr
                )
            ]

            clean_intensity = np.asarray(
                intensity_dataset[
                    int(
                        sample_index
                    )
                ],
                dtype=np.float32,
            ).copy()

            visible_mask = np.asarray(
                mask_dataset[
                    int(
                        sample_index
                    )
                ],
                dtype=np.float32,
            ).copy()

            observations.append(
                BenchmarkObservation(
                    sample_index=int(
                        sample_index
                    ),
                    snr_db=float(
                        snr_db
                    ),
                    clean_intensity=clean_intensity,
                    visible_mask=visible_mask,
                )
            )

    return observations


def build_raw_angular_profile(
    polar_intensity: np.ndarray,
    radius: np.ndarray,
) -> np.ndarray:
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

    if intensity.shape[0] != len(
        radius_array
    ):
        raise ValueError(
            "Radius count does not match polar intensity."
        )

    radial_weights = radius_array.copy()

    if np.all(
        radial_weights <= EPSILON
    ):
        radial_weights = np.ones_like(
            radial_weights
        )

    raw_profile = np.sum(
        intensity
        * radial_weights[:, None],
        axis=0,
        dtype=np.float64,
    )

    raw_profile = (
        raw_profile
        - float(
            np.mean(
                raw_profile,
                dtype=np.float64,
            )
        )
    )

    return np.asarray(
        raw_profile,
        dtype=np.float64,
    )


def prepare_observation(
    observation: BenchmarkObservation,
) -> PreparedObservation:
    noisy_observation = add_deterministic_awgn(
        clean_intensity=observation.clean_intensity,
        sample_index=observation.sample_index,
        snr_db=observation.snr_db,
    )

    polar = extract_polar_profile(
        intensity=noisy_observation.intensity,
        visible_mask=observation.visible_mask,
        angular_samples=ANGULAR_SAMPLES,
        radial_samples=RADIAL_SAMPLES,
        visibility_threshold=VISIBILITY_THRESHOLD,
    )

    raw_profile = build_raw_angular_profile(
        polar_intensity=polar.polar_intensity,
        radius=polar.radius,
    )

    normalized_profile = normalize_angular_profile(
        angular_profile=polar.angular_profile,
        valid_angles=polar.valid_angles,
        remove_mean=True,
        unit_norm=False,
    )

    return PreparedObservation(
        sample_index=observation.sample_index,
        snr_db=observation.snr_db,
        theta=np.asarray(
            polar.theta,
            dtype=np.float64,
        ).copy(),
        raw_profile=np.asarray(
            raw_profile,
            dtype=np.float64,
        ).copy(),
        normalized_profile=np.asarray(
            normalized_profile,
            dtype=np.float64,
        ).copy(),
        angular_visibility=np.asarray(
            polar.angular_visibility,
            dtype=np.float64,
        ).copy(),
        valid_angles=np.asarray(
            polar.valid_angles,
            dtype=bool,
        ).copy(),
    )


def prepare_all_observations(
    observations: Sequence[BenchmarkObservation],
) -> List[PreparedObservation]:
    prepared: List[
        PreparedObservation
    ] = []

    for index, observation in enumerate(
        observations,
        start=1,
    ):
        if (
            index == 1
            or index % 32 == 0
            or index == len(
                observations
            )
        ):
            stage(
                f"    prepared {index}/{len(observations)}"
            )

        prepared.append(
            prepare_observation(
                observation
            )
        )

    return prepared


def recognize_daf(
    prepared: PreparedObservation,
) -> int:
    result = recognize_daf_state(
        theta=prepared.theta,
        angular_profile=prepared.raw_profile,
        candidate_orders=CANDIDATE_ORDERS,
        phase_bins=PHASE_BINS,
    )

    return int(
        result.predicted_label
    )


def recognize_mask_uls(
    prepared: PreparedObservation,
) -> int:
    unit_visibility = np.ones_like(
        prepared.angular_visibility,
        dtype=np.float64,
    )

    all_valid = np.ones_like(
        prepared.valid_angles,
        dtype=bool,
    )

    result = recognize_harmonic_state(
        theta=prepared.theta,
        angular_profile=prepared.normalized_profile,
        angular_visibility=unit_visibility,
        valid_angles=all_valid,
        candidate_orders=CANDIDATE_ORDERS,
        phase_bins=PHASE_BINS,
        regularization=REGULARIZATION,
        weight_power=0.0,
    )

    return int(
        result.predicted_label
    )


def recognize_raw_vwls(
    prepared: PreparedObservation,
) -> int:
    result = recognize_harmonic_state(
        theta=prepared.theta,
        angular_profile=prepared.raw_profile,
        angular_visibility=prepared.angular_visibility,
        valid_angles=prepared.valid_angles,
        candidate_orders=CANDIDATE_ORDERS,
        phase_bins=PHASE_BINS,
        regularization=REGULARIZATION,
        weight_power=WEIGHT_POWER,
    )

    return int(
        result.predicted_label
    )


def recognize_mc_vwls(
    prepared: PreparedObservation,
) -> int:
    result = recognize_harmonic_state(
        theta=prepared.theta,
        angular_profile=prepared.normalized_profile,
        angular_visibility=prepared.angular_visibility,
        valid_angles=prepared.valid_angles,
        candidate_orders=CANDIDATE_ORDERS,
        phase_bins=PHASE_BINS,
        regularization=REGULARIZATION,
        weight_power=WEIGHT_POWER,
    )

    return int(
        result.predicted_label
    )


RECOGNITION_FUNCTIONS: Dict[
    str,
    Callable[
        [
            PreparedObservation
        ],
        int,
    ],
] = {
    "A0_DAF": recognize_daf,
    "A1_MASK_ULS": recognize_mask_uls,
    "A2_RAW_VWLS": recognize_raw_vwls,
    "A3_MC_VWLS": recognize_mc_vwls,
}


def end_to_end_predict(
    method: str,
    observation: BenchmarkObservation,
) -> int:
    prepared = prepare_observation(
        observation
    )

    return RECOGNITION_FUNCTIONS[
        method
    ](
        prepared
    )


def run_warmup(
    observations: Sequence[BenchmarkObservation],
    prepared_observations: Sequence[PreparedObservation],
) -> None:
    warmup_count = min(
        WARMUP_COUNT,
        len(
            observations
        ),
    )

    stage(
        f"[4] Warm-up: {warmup_count} observations per method"
    )

    checksum = 0

    for method in METHOD_NAMES:
        for index in range(
            warmup_count
        ):
            checksum += RECOGNITION_FUNCTIONS[
                method
            ](
                prepared_observations[
                    index
                ]
            )

            checksum += end_to_end_predict(
                method=method,
                observation=observations[
                    index
                ],
            )

    stage(
        f"[4] PASS: checksum={checksum}"
    )


def summarize_times(
    *,
    method: str,
    timing_scope: str,
    total_times: Sequence[float],
    observation_count: int,
) -> RuntimeSummary:
    total_time_array = np.asarray(
        total_times,
        dtype=np.float64,
    )

    per_image_ms = (
        total_time_array
        / float(
            observation_count
        )
        * 1000.0
    )

    mean_total = float(
        np.mean(
            total_time_array,
            dtype=np.float64,
        )
    )

    standard_deviation_total = float(
        np.std(
            total_time_array,
            ddof=1,
        )
        if len(
            total_time_array
        ) > 1
        else 0.0
    )

    mean_ms = float(
        np.mean(
            per_image_ms,
            dtype=np.float64,
        )
    )

    standard_deviation_ms = float(
        np.std(
            per_image_ms,
            ddof=1,
        )
        if len(
            per_image_ms
        ) > 1
        else 0.0
    )

    median_ms = float(
        np.median(
            per_image_ms
        )
    )

    minimum_ms = float(
        np.min(
            per_image_ms
        )
    )

    maximum_ms = float(
        np.max(
            per_image_ms
        )
    )

    throughput = float(
        1000.0
        / max(
            mean_ms,
            EPSILON,
        )
    )

    return RuntimeSummary(
        method=method,
        timing_scope=timing_scope,
        repeat_count=len(
            total_times
        ),
        observation_count=observation_count,
        total_times_seconds=tuple(
            float(value)
            for value in total_times
        ),
        mean_total_seconds=mean_total,
        standard_deviation_total_seconds=(
            standard_deviation_total
        ),
        mean_time_ms_per_image=mean_ms,
        standard_deviation_ms_per_image=(
            standard_deviation_ms
        ),
        median_time_ms_per_image=median_ms,
        minimum_time_ms_per_image=minimum_ms,
        maximum_time_ms_per_image=maximum_ms,
        throughput_images_per_second=throughput,
    )


def benchmark_recognition(
    method: str,
    prepared_observations: Sequence[PreparedObservation],
) -> RuntimeSummary:
    function = RECOGNITION_FUNCTIONS[
        method
    ]

    total_times: List[
        float
    ] = []

    checksum = 0

    for repeat_index in range(
        1,
        REPEAT_COUNT + 1,
    ):
        gc.collect()

        start_time = time.perf_counter_ns()

        for prepared in prepared_observations:
            checksum += function(
                prepared
            )

        end_time = time.perf_counter_ns()

        elapsed_seconds = (
            end_time
            - start_time
        ) / 1.0e9

        total_times.append(
            float(
                elapsed_seconds
            )
        )

        stage(
            f"    {method} recognition repeat "
            f"{repeat_index}/{REPEAT_COUNT}: "
            f"{elapsed_seconds:.6f} s"
        )

    if checksum < 0:
        raise RuntimeError(
            "Unexpected benchmark checksum."
        )

    return summarize_times(
        method=method,
        timing_scope="recognition_only",
        total_times=total_times,
        observation_count=len(
            prepared_observations
        ),
    )


def benchmark_end_to_end(
    method: str,
    observations: Sequence[BenchmarkObservation],
) -> RuntimeSummary:
    total_times: List[
        float
    ] = []

    checksum = 0

    for repeat_index in range(
        1,
        REPEAT_COUNT + 1,
    ):
        gc.collect()

        start_time = time.perf_counter_ns()

        for observation in observations:
            checksum += end_to_end_predict(
                method=method,
                observation=observation,
            )

        end_time = time.perf_counter_ns()

        elapsed_seconds = (
            end_time
            - start_time
        ) / 1.0e9

        total_times.append(
            float(
                elapsed_seconds
            )
        )

        stage(
            f"    {method} end-to-end repeat "
            f"{repeat_index}/{REPEAT_COUNT}: "
            f"{elapsed_seconds:.6f} s"
        )

    if checksum < 0:
        raise RuntimeError(
            "Unexpected benchmark checksum."
        )

    return summarize_times(
        method=method,
        timing_scope="end_to_end",
        total_times=total_times,
        observation_count=len(
            observations
        ),
    )


def save_runtime_csv(
    summaries: Sequence[RuntimeSummary],
) -> None:
    RUNTIME_CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "method",
        "method_display_name",
        "timing_scope",
        "observation_count",
        "repeat_count",
        "mean_total_seconds",
        "std_total_seconds",
        "mean_ms_per_image",
        "std_ms_per_image",
        "median_ms_per_image",
        "minimum_ms_per_image",
        "maximum_ms_per_image",
        "throughput_images_per_second",
        "repeat_times_seconds",
    ]

    with RUNTIME_CSV_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for summary in summaries:
            writer.writerow(
                {
                    "method": summary.method,
                    "method_display_name": (
                        METHOD_DISPLAY_NAMES[
                            summary.method
                        ]
                    ),
                    "timing_scope": summary.timing_scope,
                    "observation_count": (
                        summary.observation_count
                    ),
                    "repeat_count": (
                        summary.repeat_count
                    ),
                    "mean_total_seconds": (
                        f"{summary.mean_total_seconds:.12f}"
                    ),
                    "std_total_seconds": (
                        f"{summary.standard_deviation_total_seconds:.12f}"
                    ),
                    "mean_ms_per_image": (
                        f"{summary.mean_time_ms_per_image:.9f}"
                    ),
                    "std_ms_per_image": (
                        f"{summary.standard_deviation_ms_per_image:.9f}"
                    ),
                    "median_ms_per_image": (
                        f"{summary.median_time_ms_per_image:.9f}"
                    ),
                    "minimum_ms_per_image": (
                        f"{summary.minimum_time_ms_per_image:.9f}"
                    ),
                    "maximum_ms_per_image": (
                        f"{summary.maximum_time_ms_per_image:.9f}"
                    ),
                    "throughput_images_per_second": (
                        f"{summary.throughput_images_per_second:.6f}"
                    ),
                    "repeat_times_seconds": ";".join(
                        f"{value:.12f}"
                        for value in summary.total_times_seconds
                    ),
                }
            )


def build_complexity_rows() -> List[Dict[str, object]]:
    n_theta = ANGULAR_SAMPLES
    n_radius = RADIAL_SAMPLES
    candidate_count = len(
        CANDIDATE_ORDERS
    )

    common_polar_samples = (
        n_theta
        * n_radius
    )

    rows = [
        {
            "method": "A0_DAF",
            "method_display_name": "DAF",
            "polar_sampling_complexity": "O(N_theta*N_r)",
            "profile_complexity": "O(N_theta*N_r)",
            "recognition_complexity": "O(K*N_theta)",
            "total_complexity": (
                "O(N_theta*N_r + K*N_theta)"
            ),
            "dominant_memory": "O(N_theta*N_r)",
            "candidate_count_K": candidate_count,
            "angular_samples_N_theta": n_theta,
            "radial_samples_N_r": n_radius,
            "polar_grid_elements": common_polar_samples,
            "weighted_fit": 0,
            "mask_support_normalization": 0,
            "small_system_size": 0,
        },
        {
            "method": "A1_MASK_ULS",
            "method_display_name": "Mask-ULS",
            "polar_sampling_complexity": "O(N_theta*N_r)",
            "profile_complexity": "O(N_theta*N_r)",
            "recognition_complexity": (
                "O(K*N_theta*p^2 + K*p^3)"
            ),
            "total_complexity": (
                "O(N_theta*N_r + K*N_theta*p^2 + K*p^3)"
            ),
            "dominant_memory": "O(N_theta*N_r)",
            "candidate_count_K": candidate_count,
            "angular_samples_N_theta": n_theta,
            "radial_samples_N_r": n_radius,
            "polar_grid_elements": common_polar_samples,
            "weighted_fit": 0,
            "mask_support_normalization": 1,
            "small_system_size": 3,
        },
        {
            "method": "A2_RAW_VWLS",
            "method_display_name": "Raw-VWLS",
            "polar_sampling_complexity": "O(N_theta*N_r)",
            "profile_complexity": "O(N_theta*N_r)",
            "recognition_complexity": (
                "O(K*N_theta*p^2 + K*p^3)"
            ),
            "total_complexity": (
                "O(N_theta*N_r + K*N_theta*p^2 + K*p^3)"
            ),
            "dominant_memory": "O(N_theta*N_r)",
            "candidate_count_K": candidate_count,
            "angular_samples_N_theta": n_theta,
            "radial_samples_N_r": n_radius,
            "polar_grid_elements": common_polar_samples,
            "weighted_fit": 1,
            "mask_support_normalization": 0,
            "small_system_size": 3,
        },
        {
            "method": "A3_MC_VWLS",
            "method_display_name": "MC-VWLS",
            "polar_sampling_complexity": "O(N_theta*N_r)",
            "profile_complexity": "O(N_theta*N_r)",
            "recognition_complexity": (
                "O(K*N_theta*p^2 + K*p^3)"
            ),
            "total_complexity": (
                "O(N_theta*N_r + K*N_theta*p^2 + K*p^3)"
            ),
            "dominant_memory": "O(N_theta*N_r)",
            "candidate_count_K": candidate_count,
            "angular_samples_N_theta": n_theta,
            "radial_samples_N_r": n_radius,
            "polar_grid_elements": common_polar_samples,
            "weighted_fit": 1,
            "mask_support_normalization": 1,
            "small_system_size": 3,
        },
    ]

    return rows


def save_complexity_csv(
    rows: Sequence[Dict[str, object]],
) -> None:
    COMPLEXITY_CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise ValueError(
            "Complexity table is empty."
        )

    fieldnames = list(
        rows[0].keys()
    )

    with COMPLEXITY_CSV_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def get_font(
    size: int,
) -> ImageFont.ImageFont:
    candidate_paths = (
        Path(
            r"C:\Windows\Fonts\arial.ttf"
        ),
        Path(
            r"C:\Windows\Fonts\calibri.ttf"
        ),
        Path(
            r"C:\Windows\Fonts\segoeui.ttf"
        ),
    )

    for font_path in candidate_paths:
        if font_path.exists():
            return ImageFont.truetype(
                str(
                    font_path
                ),
                size=size,
            )

    return ImageFont.load_default()


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    left, top, right, bottom = box

    text_box = draw.multiline_textbbox(
        (
            0,
            0,
        ),
        text,
        font=font,
        spacing=4,
        align="center",
    )

    width = (
        text_box[2]
        - text_box[0]
    )

    height = (
        text_box[3]
        - text_box[1]
    )

    x = (
        left
        + (
            right
            - left
            - width
        )
        // 2
    )

    y = (
        top
        + (
            bottom
            - top
            - height
        )
        // 2
    )

    draw.multiline_text(
        (
            x,
            y,
        ),
        text,
        font=font,
        fill=fill,
        spacing=4,
        align="center",
    )


def generate_runtime_figure(
    summaries: Sequence[RuntimeSummary],
) -> None:
    end_to_end_summaries = [
        summary
        for summary in summaries
        if summary.timing_scope
        == "end_to_end"
    ]

    recognition_summaries = [
        summary
        for summary in summaries
        if summary.timing_scope
        == "recognition_only"
    ]

    end_to_end_by_method = {
        summary.method: summary
        for summary in end_to_end_summaries
    }

    recognition_by_method = {
        summary.method: summary
        for summary in recognition_summaries
    }

    canvas_width = 1500
    canvas_height = 900

    background = (
        255,
        255,
        255,
    )

    text_color = (
        25,
        25,
        25,
    )

    axis_color = (
        50,
        50,
        50,
    )

    grid_color = (
        215,
        215,
        215,
    )

    recognition_color = (
        80,
        125,
        175,
    )

    end_to_end_color = (
        210,
        120,
        55,
    )

    canvas = Image.new(
        "RGB",
        (
            canvas_width,
            canvas_height,
        ),
        background,
    )

    draw = ImageDraw.Draw(
        canvas
    )

    title_font = get_font(
        30
    )

    label_font = get_font(
        21
    )

    small_font = get_font(
        17
    )

    value_font = get_font(
        15
    )

    draw_centered_text(
        draw,
        (
            0,
            15,
            canvas_width,
            80,
        ),
        "Single-image runtime of harmonic-recognition methods",
        title_font,
        text_color,
    )

    chart_left = 130
    chart_right = 1430
    chart_top = 130
    chart_bottom = 750

    all_values = [
        summary.mean_time_ms_per_image
        for summary in summaries
    ]

    maximum_value = max(
        max(
            all_values
        ),
        EPSILON,
    )

    axis_maximum = (
        maximum_value
        * 1.22
    )

    grid_count = 5

    for grid_index in range(
        grid_count + 1
    ):
        fraction = (
            grid_index
            / grid_count
        )

        y = int(
            chart_bottom
            - fraction
            * (
                chart_bottom
                - chart_top
            )
        )

        value = (
            axis_maximum
            * fraction
        )

        draw.line(
            (
                chart_left,
                y,
                chart_right,
                y,
            ),
            fill=grid_color,
            width=1,
        )

        label = (
            f"{value:.2f}"
        )

        text_box = draw.textbbox(
            (
                0,
                0,
            ),
            label,
            font=small_font,
        )

        draw.text(
            (
                chart_left
                - 18
                - (
                    text_box[2]
                    - text_box[0]
                ),
                y
                - (
                    text_box[3]
                    - text_box[1]
                )
                // 2,
            ),
            label,
            font=small_font,
            fill=text_color,
        )

    draw.line(
        (
            chart_left,
            chart_top,
            chart_left,
            chart_bottom,
        ),
        fill=axis_color,
        width=2,
    )

    draw.line(
        (
            chart_left,
            chart_bottom,
            chart_right,
            chart_bottom,
        ),
        fill=axis_color,
        width=2,
    )

    method_slot_width = (
        chart_right
        - chart_left
    ) / len(
        METHOD_NAMES
    )

    bar_width = 82
    bar_gap = 18

    for method_index, method in enumerate(
        METHOD_NAMES
    ):
        center_x = (
            chart_left
            + method_slot_width
            * (
                method_index
                + 0.5
            )
        )

        recognition_summary = recognition_by_method[
            method
        ]

        end_to_end_summary = end_to_end_by_method[
            method
        ]

        values = (
            (
                recognition_summary.mean_time_ms_per_image,
                recognition_color,
                -bar_width
                - bar_gap
                / 2,
            ),
            (
                end_to_end_summary.mean_time_ms_per_image,
                end_to_end_color,
                bar_gap
                / 2,
            ),
        )

        for value, fill, offset in values:
            bar_left = int(
                center_x
                + offset
            )

            bar_right = (
                bar_left
                + bar_width
            )

            bar_height = int(
                value
                / axis_maximum
                * (
                    chart_bottom
                    - chart_top
                )
            )

            bar_top = (
                chart_bottom
                - bar_height
            )

            draw.rectangle(
                (
                    bar_left,
                    bar_top,
                    bar_right,
                    chart_bottom,
                ),
                fill=fill,
                outline=axis_color,
                width=1,
            )

            value_text = (
                f"{value:.3f}"
            )

            text_box = draw.textbbox(
                (
                    0,
                    0,
                ),
                value_text,
                font=value_font,
            )

            text_width = (
                text_box[2]
                - text_box[0]
            )

            draw.text(
                (
                    (
                        bar_left
                        + bar_right
                        - text_width
                    )
                    // 2,
                    max(
                        chart_top,
                        bar_top - 25,
                    ),
                ),
                value_text,
                font=value_font,
                fill=text_color,
            )

        method_label = METHOD_DISPLAY_NAMES[
            method
        ]

        text_box = draw.textbbox(
            (
                0,
                0,
            ),
            method_label,
            font=label_font,
        )

        text_width = (
            text_box[2]
            - text_box[0]
        )

        draw.text(
            (
                int(
                    center_x
                    - text_width
                    / 2
                ),
                chart_bottom + 22,
            ),
            method_label,
            font=label_font,
            fill=text_color,
        )

    y_axis_label = "Mean runtime (ms/image)"

    rotated_label = Image.new(
        "RGBA",
        (
            400,
            50,
        ),
        (
            255,
            255,
            255,
            0,
        ),
    )

    rotated_draw = ImageDraw.Draw(
        rotated_label
    )

    rotated_draw.text(
        (
            0,
            5,
        ),
        y_axis_label,
        font=label_font,
        fill=text_color,
    )

    rotated_label = rotated_label.rotate(
        90,
        expand=True,
    )

    canvas.paste(
        rotated_label,
        (
            25,
            270,
        ),
        rotated_label,
    )

    legend_y = 830

    draw.rectangle(
        (
            430,
            legend_y,
            465,
            legend_y + 24,
        ),
        fill=recognition_color,
        outline=axis_color,
        width=1,
    )

    draw.text(
        (
            480,
            legend_y,
        ),
        "Recognition only",
        font=small_font,
        fill=text_color,
    )

    draw.rectangle(
        (
            760,
            legend_y,
            795,
            legend_y + 24,
        ),
        fill=end_to_end_color,
        outline=axis_color,
        width=1,
    )

    draw.text(
        (
            810,
            legend_y,
        ),
        "End-to-end",
        font=small_font,
        fill=text_color,
    )

    PNG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        PNG_PATH,
        format="PNG",
        optimize=True,
    )

    canvas.save(
        PDF_PATH,
        format="PDF",
        resolution=300.0,
    )


def build_report(
    summaries: Sequence[RuntimeSummary],
    selected_indices: np.ndarray,
) -> str:
    end_to_end = {
        summary.method: summary
        for summary in summaries
        if summary.timing_scope
        == "end_to_end"
    }

    recognition_only = {
        summary.method: summary
        for summary in summaries
        if summary.timing_scope
        == "recognition_only"
    }

    fastest_end_to_end_method = min(
        METHOD_NAMES,
        key=lambda method: (
            end_to_end[
                method
            ].mean_time_ms_per_image
        ),
    )

    fastest_recognition_method = min(
        METHOD_NAMES,
        key=lambda method: (
            recognition_only[
                method
            ].mean_time_ms_per_image
        ),
    )

    mc_end_to_end = end_to_end[
        "A3_MC_VWLS"
    ].mean_time_ms_per_image

    daf_end_to_end = end_to_end[
        "A0_DAF"
    ].mean_time_ms_per_image

    uls_end_to_end = end_to_end[
        "A1_MASK_ULS"
    ].mean_time_ms_per_image

    mc_to_daf_ratio = (
        mc_end_to_end
        / max(
            daf_end_to_end,
            EPSILON,
        )
    )

    mc_to_uls_ratio = (
        mc_end_to_end
        / max(
            uls_end_to_end,
            EPSILON,
        )
    )

    lines = [
        "MC-VWLS runtime and computational-complexity benchmark",
        "",
        "[ENVIRONMENT]",
        f"Python={sys.version.replace(chr(10), ' ')}",
        f"Platform={platform.platform()}",
        f"Processor={platform.processor()}",
        f"NumPy={np.__version__}",
        f"Timer=time.perf_counter_ns",
        "",
        "[BENCHMARK PROTOCOL]",
        f"Frozen test subset size={len(selected_indices)}",
        f"Subset seed={BENCHMARK_SEED}",
        f"Warm-up observations per method={WARMUP_COUNT}",
        f"Repeated measurements={REPEAT_COUNT}",
        (
            "SNR assignment="
            + ",".join(
                str(
                    float(value)
                )
                for value in SUPPORTED_SNR_DB
            )
            + " dB, assigned cyclically"
        ),
        f"Angular samples={ANGULAR_SAMPLES}",
        f"Radial samples={RADIAL_SAMPLES}",
        f"Candidate orders={CANDIDATE_ORDERS}",
        f"Phase bins={PHASE_BINS}",
        f"Visibility threshold={VISIBILITY_THRESHOLD}",
        f"Regularization={REGULARIZATION}",
        f"Weight power={WEIGHT_POWER}",
        (
            "Timing is single-process Python CPU timing. "
            "File loading and result serialization are excluded."
        ),
        (
            "End-to-end timing includes deterministic noise, polar "
            "sampling, angular-profile construction, and recognition."
        ),
        (
            "Recognition-only timing starts from frozen prepared polar "
            "and angular representations."
        ),
        "",
        "[RUNTIME RESULTS]",
    ]

    for method in METHOD_NAMES:
        recognition = recognition_only[
            method
        ]

        complete = end_to_end[
            method
        ]

        lines.extend(
            [
                (
                    f"{method} ({METHOD_DISPLAY_NAMES[method]}):"
                ),
                (
                    "  recognition_only="
                    f"{recognition.mean_time_ms_per_image:.6f} "
                    "ms/image, "
                    f"std={recognition.standard_deviation_ms_per_image:.6f}, "
                    f"throughput={recognition.throughput_images_per_second:.3f} "
                    "images/s"
                ),
                (
                    "  end_to_end="
                    f"{complete.mean_time_ms_per_image:.6f} "
                    "ms/image, "
                    f"std={complete.standard_deviation_ms_per_image:.6f}, "
                    f"throughput={complete.throughput_images_per_second:.3f} "
                    "images/s"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "[RUNTIME COMPARISONS]",
            (
                "Fastest recognition-only method="
                f"{fastest_recognition_method}"
            ),
            (
                "Fastest end-to-end method="
                f"{fastest_end_to_end_method}"
            ),
            (
                "MC-VWLS / DAF end-to-end runtime ratio="
                f"{mc_to_daf_ratio:.6f}"
            ),
            (
                "MC-VWLS / Mask-ULS end-to-end runtime ratio="
                f"{mc_to_uls_ratio:.6f}"
            ),
            "",
            "[COMPUTATIONAL COMPLEXITY]",
            (
                "Let N_theta denote angular samples, N_r radial samples, "
                "K candidate OAM orders, and p=3 fitted coefficients."
            ),
            (
                "Polar sampling and radial aggregation require "
                "O(N_theta*N_r) time and dominate stored intermediate "
                "memory with O(N_theta*N_r)."
            ),
            (
                "DAF recognition evaluates K angular harmonics and has "
                "O(K*N_theta) recognition complexity."
            ),
            (
                "ULS and VWLS construct one p-by-p normal system for each "
                "candidate, giving "
                "O(K*N_theta*p^2 + K*p^3)."
            ),
            (
                "Because p=3 and K=4 are fixed, all four complete methods "
                "are asymptotically O(N_theta*N_r) for the present "
                "implementation."
            ),
            (
                f"Current polar grid size="
                f"{ANGULAR_SAMPLES * RADIAL_SAMPLES} samples."
            ),
            "",
            "[ENVIRONMENT NOTE]",
            (
                "The timing script uses the validated BLAS-free 3x3 "
                "harmonic solver because the current Windows environment "
                "terminates natively for specific BLAS/LAPACK operations. "
                "The frozen harmonic model and parameters are unchanged."
            ),
            "",
            "[OUTPUTS]",
            f"Runtime CSV: {RUNTIME_CSV_PATH}",
            f"Complexity CSV: {COMPLEXITY_CSV_PATH}",
            f"PNG figure: {PNG_PATH}",
            f"PDF figure: {PDF_PATH}",
            f"Report: {REPORT_PATH}",
        ]
    )

    return "\n".join(
        lines
    )


def main() -> None:
    stage(
        "=" * 78
    )

    stage(
        "MC-VWLS RUNTIME AND COMPUTATIONAL-COMPLEXITY BENCHMARK"
    )

    stage(
        "=" * 78
    )

    if not H5_PATH.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {H5_PATH}"
        )

    stage(
        "[1] Load frozen test indices"
    )

    test_indices = load_test_indices()

    stage(
        f"[1] PASS: test samples={len(test_indices)}"
    )

    stage(
        "[2] Select deterministic benchmark subset"
    )

    selected_indices = choose_benchmark_indices(
        test_indices
    )

    stage(
        f"[2] PASS: selected={len(selected_indices)}"
    )

    stage(
        "[3] Load benchmark observations"
    )

    observations = build_benchmark_observations(
        selected_indices
    )

    stage(
        f"[3] PASS: observations={len(observations)}"
    )

    stage(
        "[3.1] Prepare frozen polar representations"
    )

    prepared_observations = prepare_all_observations(
        observations
    )

    stage(
        f"[3.1] PASS: prepared={len(prepared_observations)}"
    )

    run_warmup(
        observations=observations,
        prepared_observations=prepared_observations,
    )

    summaries: List[
        RuntimeSummary
    ] = []

    stage(
        "[5] Recognition-only benchmark"
    )

    for method in METHOD_NAMES:
        stage(
            f"[5] Method: {method}"
        )

        summaries.append(
            benchmark_recognition(
                method=method,
                prepared_observations=prepared_observations,
            )
        )

    stage(
        "[5] PASS"
    )

    stage(
        "[6] End-to-end benchmark"
    )

    for method in METHOD_NAMES:
        stage(
            f"[6] Method: {method}"
        )

        summaries.append(
            benchmark_end_to_end(
                method=method,
                observations=observations,
            )
        )

    stage(
        "[6] PASS"
    )

    stage(
        "[7] Save runtime CSV"
    )

    save_runtime_csv(
        summaries
    )

    stage(
        f"[7] PASS: {RUNTIME_CSV_PATH}"
    )

    stage(
        "[8] Save complexity CSV"
    )

    complexity_rows = build_complexity_rows()

    save_complexity_csv(
        complexity_rows
    )

    stage(
        f"[8] PASS: {COMPLEXITY_CSV_PATH}"
    )

    stage(
        "[9] Generate runtime figure with Pillow"
    )

    generate_runtime_figure(
        summaries
    )

    stage(
        f"[9] PASS: {PNG_PATH}"
    )

    stage(
        "[10] Generate validation report"
    )

    report_text = build_report(
        summaries=summaries,
        selected_indices=selected_indices,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    stage(
        "[10] PASS"
    )

    stage(
        ""
    )

    stage(
        report_text
    )

    stage(
        ""
    )

    stage(
        "=" * 78
    )

    stage(
        "RUNTIME AND COMPLEXITY BENCHMARK COMPLETE"
    )

    stage(
        "=" * 78
    )


if __name__ == "__main__":
    main()