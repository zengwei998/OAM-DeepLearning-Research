"""
Generate representative MC-VWLS success and failure cases.

Inputs:
    results/csv/mc_vwls_ablation_full_test_predictions.csv
    data/generated/occlusion_clean_v2.h5

Outputs:
    results/figures/fig_mc_vwls_failure_cases.png
    results/figures/fig_mc_vwls_failure_cases.pdf
    results/csv/mc_vwls_selected_failure_cases.csv
    results/validation/mc_vwls_failure_cases_report.txt
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.algorithms.harmonic_fit import HarmonicRecognition
from src.algorithms.polar_sampling import (
    PolarProfile,
    extract_polar_profile,
    normalize_angular_profile,
)
from src.evaluation.blas_free_harmonic import (
    recognize_harmonic_state,
)
from src.physics.receiver_noise import (
    add_deterministic_awgn,
)


ROOT = Path(__file__).resolve().parents[2]

PREDICTION_PATH = (
    ROOT
    / "results"
    / "csv"
    / "mc_vwls_ablation_full_test_predictions.csv"
)

DATASET_PATH = (
    ROOT
    / "data"
    / "generated"
    / "occlusion_clean_v2.h5"
)

FIGURE_DIRECTORY = (
    ROOT
    / "results"
    / "figures"
)

CSV_DIRECTORY = (
    ROOT
    / "results"
    / "csv"
)

REPORT_PATH = (
    ROOT
    / "results"
    / "validation"
    / "mc_vwls_failure_cases_report.txt"
)

PNG_PATH = (
    FIGURE_DIRECTORY
    / "fig_mc_vwls_failure_cases.png"
)

PDF_PATH = (
    FIGURE_DIRECTORY
    / "fig_mc_vwls_failure_cases.pdf"
)

SELECTED_CASES_PATH = (
    CSV_DIRECTORY
    / "mc_vwls_selected_failure_cases.csv"
)

TARGET_METHOD = "A3_MC_VWLS"

ANGULAR_SAMPLES = 180
RADIAL_SAMPLES = 64

CANDIDATE_ORDERS = (
    1,
    2,
    3,
    4,
)

PHASE_BINS = 8

VISIBILITY_THRESHOLD = 0.0
REGULARIZATION = 0.0
WEIGHT_POWER = 0.5

CONDITION_CN2_COLUMN = 0
CONDITION_DISTANCE_COLUMN = 1
CONDITION_OCCLUSION_COLUMN = 3

REQUIRED_COLUMNS = (
    "method",
    "sample_index",
    "target_snr_db",
    "target_occlusion",
    "true_label",
    "predicted_label",
    "true_order",
    "predicted_order",
    "true_phase_bin",
    "predicted_phase_bin",
    "label_correct",
    "order_correct",
    "phase_correct",
    "confidence",
    "harmonic_margin",
    "best_score",
    "second_best_score",
)


def load_prediction_rows(
    path: Path,
) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction CSV does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file
        )

        fieldnames = list(
            reader.fieldnames
            or []
        )

        missing_columns = (
            set(REQUIRED_COLUMNS)
            - set(fieldnames)
        )

        if missing_columns:
            raise KeyError(
                "Prediction CSV is missing columns: "
                f"{sorted(missing_columns)}"
            )

        rows = [
            row
            for row in reader
            if row["method"] == TARGET_METHOD
        ]

    if not rows:
        raise ValueError(
            f"No rows found for method: {TARGET_METHOD}"
        )

    return rows


def parse_int(
    row: Dict[str, str],
    column: str,
) -> int:
    return int(
        round(
            float(
                row[column]
            )
        )
    )


def parse_float(
    row: Dict[str, str],
    column: str,
) -> float:
    return float(
        row[column]
    )


def circular_phase_distance(
    true_phase: int,
    predicted_phase: int,
) -> int:
    direct_distance = abs(
        true_phase
        - predicted_phase
    )

    return min(
        direct_distance,
        PHASE_BINS
        - direct_distance,
    )


def attach_conditions(
    rows: Sequence[Dict[str, str]],
    conditions: np.ndarray,
) -> List[Dict[str, object]]:
    enriched_rows: List[
        Dict[str, object]
    ] = []

    for row in rows:
        sample_index = parse_int(
            row,
            "sample_index",
        )

        if not (
            0
            <= sample_index
            < conditions.shape[0]
        ):
            raise IndexError(
                "Sample index is outside conditions array: "
                f"{sample_index}"
            )

        condition = conditions[
            sample_index
        ]

        enriched: Dict[
            str,
            object,
        ] = {
            key: value
            for key, value in row.items()
        }

        enriched.update(
            {
                "sample_index_int": sample_index,
                "target_snr_db_float": parse_float(
                    row,
                    "target_snr_db",
                ),
                "target_occlusion_float": parse_float(
                    row,
                    "target_occlusion",
                ),
                "true_order_int": parse_int(
                    row,
                    "true_order",
                ),
                "predicted_order_int": parse_int(
                    row,
                    "predicted_order",
                ),
                "true_phase_bin_int": parse_int(
                    row,
                    "true_phase_bin",
                ),
                "predicted_phase_bin_int": parse_int(
                    row,
                    "predicted_phase_bin",
                ),
                "label_correct_int": parse_int(
                    row,
                    "label_correct",
                ),
                "order_correct_int": parse_int(
                    row,
                    "order_correct",
                ),
                "phase_correct_int": parse_int(
                    row,
                    "phase_correct",
                ),
                "confidence_float": parse_float(
                    row,
                    "confidence",
                ),
                "cn2": float(
                    condition[
                        CONDITION_CN2_COLUMN
                    ]
                ),
                "distance": float(
                    condition[
                        CONDITION_DISTANCE_COLUMN
                    ]
                ),
                "condition_occlusion": float(
                    condition[
                        CONDITION_OCCLUSION_COLUMN
                    ]
                ),
            }
        )

        if not np.isclose(
            float(
                enriched[
                    "target_occlusion_float"
                ]
            ),
            float(
                enriched[
                    "condition_occlusion"
                ]
            ),
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise ValueError(
                "Occlusion mismatch for sample "
                f"{sample_index}: "
                f"CSV={enriched['target_occlusion_float']}, "
                f"HDF5={enriched['condition_occlusion']}"
            )

        enriched_rows.append(
            enriched
        )

    return enriched_rows


def choose_median_confidence_case(
    rows: Sequence[Dict[str, object]],
    *,
    case_name: str,
) -> Dict[str, object]:
    if not rows:
        raise ValueError(
            f"No candidate rows found for case: {case_name}"
        )

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            float(
                row[
                    "confidence_float"
                ]
            ),
            int(
                row[
                    "sample_index_int"
                ]
            ),
            float(
                row[
                    "target_snr_db_float"
                ]
            ),
        ),
    )

    return dict(
        sorted_rows[
            len(sorted_rows) // 2
        ]
    )


def select_cases(
    rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    weak_correct_candidates = [
        row
        for row in rows
        if (
            int(
                row[
                    "label_correct_int"
                ]
            ) == 1
            and float(
                row[
                    "cn2"
                ]
            ) <= 1.0e-14
            and float(
                row[
                    "target_occlusion_float"
                ]
            ) >= 0.3
            and float(
                row[
                    "target_snr_db_float"
                ]
            ) <= 5.0
        )
    ]

    severe_correct_candidates = [
        row
        for row in rows
        if (
            int(
                row[
                    "label_correct_int"
                ]
            ) == 1
            and float(
                row[
                    "cn2"
                ]
            ) >= 5.0e-14
            and float(
                row[
                    "distance"
                ]
            ) >= 750.0
        )
    ]

    same_order_phase_error_candidates = [
        row
        for row in rows
        if (
            int(
                row[
                    "label_correct_int"
                ]
            ) == 0
            and int(
                row[
                    "order_correct_int"
                ]
            ) == 1
            and circular_phase_distance(
                int(
                    row[
                        "true_phase_bin_int"
                    ]
                ),
                int(
                    row[
                        "predicted_phase_bin_int"
                    ]
                ),
            ) == 1
        )
    ]

    preferred_cross_order_candidates = [
        row
        for row in rows
        if (
            int(
                row[
                    "order_correct_int"
                ]
            ) == 0
            and int(
                row[
                    "predicted_order_int"
                ]
            ) == 1
            and int(
                row[
                    "predicted_phase_bin_int"
                ]
            ) == 4
            and float(
                row[
                    "cn2"
                ]
            ) >= 5.0e-14
            and float(
                row[
                    "distance"
                ]
            ) >= 750.0
        )
    ]

    if preferred_cross_order_candidates:
        cross_order_candidates = (
            preferred_cross_order_candidates
        )
    else:
        cross_order_candidates = [
            row
            for row in rows
            if int(
                row[
                    "order_correct_int"
                ]
            ) == 0
        ]

    selected = [
        choose_median_confidence_case(
            weak_correct_candidates,
            case_name="weak_turbulence_correct",
        ),
        choose_median_confidence_case(
            severe_correct_candidates,
            case_name="severe_turbulence_correct",
        ),
        choose_median_confidence_case(
            same_order_phase_error_candidates,
            case_name="same_order_adjacent_phase_error",
        ),
        choose_median_confidence_case(
            cross_order_candidates,
            case_name="cross_order_error",
        ),
    ]

    case_names = (
        "Correct: weak turbulence",
        "Correct: severe turbulence",
        "Failure: adjacent phase",
        "Failure: cross order",
    )

    for case_name, row in zip(
        case_names,
        selected,
    ):
        row[
            "case_name"
        ] = case_name

    sample_snr_pairs = {
        (
            int(
                row[
                    "sample_index_int"
                ]
            ),
            float(
                row[
                    "target_snr_db_float"
                ]
            ),
        )
        for row in selected
    }

    if len(sample_snr_pairs) != len(
        selected
    ):
        raise RuntimeError(
            "Duplicate selected observations were found."
        )

    return selected


def build_fitted_curve(
    theta: np.ndarray,
    recognition: HarmonicRecognition,
) -> np.ndarray:
    candidate = recognition.candidates[
        recognition.predicted_order
    ]

    return (
        candidate.intercept
        + candidate.cosine_coefficient
        * np.cos(
            candidate.harmonic_order
            * theta
        )
        + candidate.sine_coefficient
        * np.sin(
            candidate.harmonic_order
            * theta
        )
    )


def reconstruct_case(
    row: Dict[str, object],
    intensity_dataset: h5py.Dataset,
    mask_dataset: h5py.Dataset,
) -> Dict[str, object]:
    sample_index = int(
        row[
            "sample_index_int"
        ]
    )

    snr_db = float(
        row[
            "target_snr_db_float"
        ]
    )

    print(
        "    [A] Read intensity",
        flush=True,
    )

    clean_intensity = np.asarray(
        intensity_dataset[
            sample_index
        ],
        dtype=np.float32,
    ).copy()

    print(
        "    [B] Read visible mask",
        flush=True,
    )

    visible_mask = np.asarray(
        mask_dataset[
            sample_index
        ],
        dtype=np.float32,
    ).copy()

    print(
        "    [C] Generate noisy observation",
        flush=True,
    )

    observation = add_deterministic_awgn(
        clean_intensity=clean_intensity,
        sample_index=sample_index,
        snr_db=snr_db,
    )

    print(
        "    [D] Extract polar profile",
        flush=True,
    )

    polar = extract_polar_profile(
        intensity=observation.intensity,
        visible_mask=visible_mask,
        angular_samples=ANGULAR_SAMPLES,
        radial_samples=RADIAL_SAMPLES,
        visibility_threshold=VISIBILITY_THRESHOLD,
    )

    print(
        "    [E] Normalize angular profile",
        flush=True,
    )

    normalized_profile = normalize_angular_profile(
        angular_profile=polar.angular_profile,
        valid_angles=polar.valid_angles,
        remove_mean=True,
        unit_norm=False,
    )

    print(
        "    [F] BLAS-free harmonic recognition",
        flush=True,
    )

    recognition = recognize_harmonic_state(
        theta=polar.theta,
        angular_profile=normalized_profile,
        angular_visibility=polar.angular_visibility,
        valid_angles=polar.valid_angles,
        candidate_orders=CANDIDATE_ORDERS,
        phase_bins=PHASE_BINS,
        regularization=REGULARIZATION,
        weight_power=WEIGHT_POWER,
    )

    print(
        "    [G] Validate reconstructed prediction",
        flush=True,
    )

    expected_order = int(
        row[
            "predicted_order_int"
        ]
    )

    expected_phase = int(
        row[
            "predicted_phase_bin_int"
        ]
    )

    if recognition.predicted_order != expected_order:
        raise ValueError(
            "Reconstructed order does not match prediction CSV: "
            f"sample={sample_index}, "
            f"expected={expected_order}, "
            f"actual={recognition.predicted_order}"
        )

    if recognition.predicted_phase_bin != expected_phase:
        raise ValueError(
            "Reconstructed phase bin does not match prediction CSV: "
            f"sample={sample_index}, "
            f"expected={expected_phase}, "
            f"actual={recognition.predicted_phase_bin}"
        )

    print(
        "    [H] Build fitted curve",
        flush=True,
    )

    fitted_curve = build_fitted_curve(
        theta=polar.theta,
        recognition=recognition,
    )

    return {
        "row": row,
        "clean_intensity": clean_intensity,
        "noisy_intensity": np.asarray(
            observation.intensity,
            dtype=np.float64,
        ),
        "visible_mask": visible_mask,
        "polar": polar,
        "normalized_profile": normalized_profile,
        "recognition": recognition,
        "fitted_curve": fitted_curve,
    }


def format_case_title(
    row: Dict[str, object],
) -> str:
    return (
        f"{row['case_name']}\n"
        f"true=({row['true_order_int']},"
        f"{row['true_phase_bin_int']}), "
        f"pred=({row['predicted_order_int']},"
        f"{row['predicted_phase_bin_int']}), "
        f"SNR={float(row['target_snr_db_float']):.0f} dB\n"
        f"Cn2={float(row['cn2']):.1e}, "
        f"d={float(row['distance']):.0f} m, "
        f"occ={float(row['target_occlusion_float']):.1f}"
    )


def plot_cases(
    reconstructed_cases: Sequence[Dict[str, object]],
) -> None:
    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    row_count = len(
        reconstructed_cases
    )

    print(
        "    [P1] Create figure",
        flush=True,
    )

    figure, axes = plt.subplots(
        nrows=row_count,
        ncols=5,
        figsize=(
            18.0,
            3.8 * row_count,
        ),
        squeeze=False,
    )

    for row_index, case in enumerate(
        reconstructed_cases
    ):
        print(
            f"    [P2.{row_index + 1}] Plot case row",
            flush=True,
        )

        row = case[
            "row"
        ]

        clean_intensity = np.asarray(
            case[
                "clean_intensity"
            ],
            dtype=np.float64,
        )

        noisy_intensity = np.asarray(
            case[
                "noisy_intensity"
            ],
            dtype=np.float64,
        )

        visible_mask = np.asarray(
            case[
                "visible_mask"
            ],
            dtype=np.float64,
        )

        polar = case[
            "polar"
        ]

        if not isinstance(
            polar,
            PolarProfile,
        ):
            raise TypeError(
                "Unexpected polar-profile object."
            )

        normalized_profile = np.asarray(
            case[
                "normalized_profile"
            ],
            dtype=np.float64,
        )

        recognition = case[
            "recognition"
        ]

        if not isinstance(
            recognition,
            HarmonicRecognition,
        ):
            raise TypeError(
                "Unexpected recognition object."
            )

        fitted_curve = np.asarray(
            case[
                "fitted_curve"
            ],
            dtype=np.float64,
        )

        clean_axis = axes[
            row_index,
            0,
        ]

        noisy_axis = axes[
            row_index,
            1,
        ]

        mask_axis = axes[
            row_index,
            2,
        ]

        polar_axis = axes[
            row_index,
            3,
        ]

        result_axis = axes[
            row_index,
            4,
        ]

        clean_image = clean_axis.imshow(
            clean_intensity,
            origin="lower",
            aspect="equal",
        )

        clean_axis.set_title(
            format_case_title(
                row
            ),
            fontsize=9,
        )

        clean_axis.set_xlabel(
            "x"
        )

        clean_axis.set_ylabel(
            "y"
        )

        figure.colorbar(
            clean_image,
            ax=clean_axis,
            fraction=0.046,
            pad=0.04,
        )

        noisy_image = noisy_axis.imshow(
            noisy_intensity,
            origin="lower",
            aspect="equal",
        )

        noisy_axis.set_title(
            "Noisy receiver intensity"
        )

        noisy_axis.set_xlabel(
            "x"
        )

        noisy_axis.set_ylabel(
            "y"
        )

        figure.colorbar(
            noisy_image,
            ax=noisy_axis,
            fraction=0.046,
            pad=0.04,
        )

        mask_image = mask_axis.imshow(
            visible_mask,
            origin="lower",
            aspect="equal",
            vmin=0.0,
            vmax=1.0,
            cmap="gray",
        )

        mask_axis.set_title(
            "Visible mask"
        )

        mask_axis.set_xlabel(
            "x"
        )

        mask_axis.set_ylabel(
            "y"
        )

        figure.colorbar(
            mask_image,
            ax=mask_axis,
            fraction=0.046,
            pad=0.04,
        )

        polar_image = polar_axis.imshow(
            polar.polar_intensity,
            origin="lower",
            aspect="auto",
            extent=(
                0.0,
                2.0 * np.pi,
                float(
                    polar.radius[0]
                ),
                float(
                    polar.radius[-1]
                ),
            ),
        )

        polar_axis.set_title(
            "Polar intensity"
        )

        polar_axis.set_xlabel(
            "Angle (rad)"
        )

        polar_axis.set_ylabel(
            "Radius (pixel)"
        )

        figure.colorbar(
            polar_image,
            ax=polar_axis,
            fraction=0.046,
            pad=0.04,
        )

        candidate_orders = list(
            CANDIDATE_ORDERS
        )

        candidate_scores = [
            float(
                recognition.candidates[
                    order
                ].score
            )
            for order in candidate_orders
        ]

        result_axis.bar(
            candidate_orders,
            candidate_scores,
        )

        result_axis.set_title(
            (
                "Candidate harmonic scores\n"
                f"confidence={recognition.confidence:.4f}, "
                f"margin={recognition.harmonic_margin:.4f}"
            ),
            fontsize=9,
        )

        result_axis.set_xlabel(
            "Candidate OAM order"
        )

        result_axis.set_ylabel(
            "Score"
        )

        result_axis.set_xticks(
            candidate_orders
        )

        result_axis.grid(
            axis="y",
            alpha=0.3,
        )

        for order, score in zip(
            candidate_orders,
            candidate_scores,
        ):
            result_axis.text(
                order,
                score,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        inset_axis = result_axis.inset_axes(
            [
                0.10,
                0.52,
                0.82,
                0.38,
            ]
        )

        inset_axis.plot(
            polar.theta,
            normalized_profile,
            linewidth=1.0,
            label="Observed",
        )

        inset_axis.plot(
            polar.theta,
            fitted_curve,
            linestyle="--",
            linewidth=1.0,
            label="Best fit",
        )

        inset_axis.set_xlim(
            0.0,
            2.0 * np.pi,
        )

        inset_axis.set_xticks(
            [
                0.0,
                np.pi,
                2.0 * np.pi,
            ]
        )

        inset_axis.set_xticklabels(
            [
                "0",
                "π",
                "2π",
            ],
            fontsize=7,
        )

        inset_axis.tick_params(
            axis="y",
            labelsize=7,
        )

        inset_axis.grid(
            alpha=0.2,
        )

    print(
        "    [P3] Apply tight layout",
        flush=True,
    )

    figure.tight_layout()

    print(
        "    [P4] Save PNG",
        flush=True,
    )

    figure.savefig(
        PNG_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        "    [P5] Save PDF",
        flush=True,
    )

    figure.savefig(
        PDF_PATH,
        bbox_inches="tight",
    )

    print(
        "    [P6] Close figure",
        flush=True,
    )

    plt.close(
        figure
    )


def save_selected_cases(
    reconstructed_cases: Sequence[Dict[str, object]],
) -> None:
    SELECTED_CASES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_rows: List[
        Dict[str, object]
    ] = []

    for case in reconstructed_cases:
        row = case[
            "row"
        ]

        recognition = case[
            "recognition"
        ]

        if not isinstance(
            recognition,
            HarmonicRecognition,
        ):
            raise TypeError(
                "Unexpected recognition object."
            )

        output_row: Dict[
            str,
            object,
        ] = {
            "case_name": row[
                "case_name"
            ],
            "sample_index": row[
                "sample_index_int"
            ],
            "target_snr_db": row[
                "target_snr_db_float"
            ],
            "target_occlusion": row[
                "target_occlusion_float"
            ],
            "cn2": row[
                "cn2"
            ],
            "distance": row[
                "distance"
            ],
            "true_order": row[
                "true_order_int"
            ],
            "true_phase_bin": row[
                "true_phase_bin_int"
            ],
            "predicted_order": recognition.predicted_order,
            "predicted_phase_bin": recognition.predicted_phase_bin,
            "label_correct": row[
                "label_correct_int"
            ],
            "order_correct": row[
                "order_correct_int"
            ],
            "phase_correct": row[
                "phase_correct_int"
            ],
            "confidence": recognition.confidence,
            "harmonic_margin": recognition.harmonic_margin,
            "best_score": recognition.best_score,
            "second_best_score": recognition.second_best_score,
        }

        for order in CANDIDATE_ORDERS:
            candidate = recognition.candidates[
                order
            ]

            output_row[
                f"order_{order}_score"
            ] = candidate.score

            output_row[
                f"order_{order}_phase_rad"
            ] = candidate.phase_rad

            output_row[
                f"order_{order}_residual_mse"
            ] = candidate.residual_mse

        output_rows.append(
            output_row
        )

    if not output_rows:
        raise ValueError(
            "No selected cases are available for CSV output."
        )

    fieldnames = list(
        output_rows[0].keys()
    )

    with SELECTED_CASES_PATH.open(
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
            output_rows
        )


def build_report(
    reconstructed_cases: Sequence[Dict[str, object]],
) -> str:
    lines = [
        "MC-VWLS representative success and failure cases",
        f"Prediction CSV: {PREDICTION_PATH}",
        f"Dataset: {DATASET_PATH}",
        "",
        "[FROZEN PARAMETERS]",
        f"angular_samples={ANGULAR_SAMPLES}",
        f"radial_samples={RADIAL_SAMPLES}",
        f"weight_power={WEIGHT_POWER}",
        f"visibility_threshold={VISIBILITY_THRESHOLD}",
        f"regularization={REGULARIZATION}",
        "",
        "[SELECTED CASES]",
    ]

    for index, case in enumerate(
        reconstructed_cases,
        start=1,
    ):
        row = case[
            "row"
        ]

        recognition = case[
            "recognition"
        ]

        if not isinstance(
            recognition,
            HarmonicRecognition,
        ):
            raise TypeError(
                "Unexpected recognition object."
            )

        score_text = ", ".join(
            (
                f"l={order}:"
                f"{recognition.candidates[order].score:.8f}"
            )
            for order in CANDIDATE_ORDERS
        )

        lines.append(
            (
                f"{index}. case={row['case_name']}, "
                f"sample_index={row['sample_index_int']}, "
                f"SNR={float(row['target_snr_db_float']):.1f}, "
                f"occlusion={float(row['target_occlusion_float']):.1f}, "
                f"Cn2={float(row['cn2']):.8e}, "
                f"distance={float(row['distance']):.1f}, "
                f"true=({row['true_order_int']},"
                f"{row['true_phase_bin_int']}), "
                f"predicted=({recognition.predicted_order},"
                f"{recognition.predicted_phase_bin}), "
                f"confidence={recognition.confidence:.8f}, "
                f"scores=[{score_text}]"
            )
        )

    lines.extend(
        [
            "",
            "[OUTPUTS]",
            f"Selected-case CSV: {SELECTED_CASES_PATH}",
            f"PNG figure: {PNG_PATH}",
            f"PDF figure: {PDF_PATH}",
            f"Report: {REPORT_PATH}",
        ]
    )

    return "\n".join(
        lines
    )


def main() -> None:
    print(
        "=" * 78,
        flush=True,
    )

    print(
        "GENERATE MC-VWLS REPRESENTATIVE FAILURE CASES",
        flush=True,
    )

    print(
        "=" * 78,
        flush=True,
    )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {DATASET_PATH}"
        )

    print(
        "[1] Load prediction CSV",
        flush=True,
    )

    prediction_rows = load_prediction_rows(
        PREDICTION_PATH
    )

    print(
        "MC-VWLS prediction rows:",
        len(
            prediction_rows
        ),
        flush=True,
    )

    print(
        "[2] Open HDF5 file",
        flush=True,
    )

    with h5py.File(
        DATASET_PATH,
        "r",
    ) as h5_file:
        print(
            "[2] PASS",
            flush=True,
        )

        required_datasets = {
            "intensity",
            "visible_mask",
            "conditions",
        }

        missing_datasets = (
            required_datasets
            - set(
                h5_file.keys()
            )
        )

        if missing_datasets:
            raise KeyError(
                "HDF5 file is missing datasets: "
                f"{sorted(missing_datasets)}"
            )

        print(
            "[3] Load complete conditions array",
            flush=True,
        )

        conditions = np.asarray(
            h5_file[
                "conditions"
            ][
                :
            ],
            dtype=np.float64,
        )

        print(
            f"[3] PASS: shape={conditions.shape}",
            flush=True,
        )

        print(
            "[4] Attach conditions to prediction rows",
            flush=True,
        )

        enriched_rows = attach_conditions(
            prediction_rows,
            conditions,
        )

        print(
            f"[4] PASS: rows={len(enriched_rows)}",
            flush=True,
        )

        print(
            "[5] Select representative cases",
            flush=True,
        )

        selected_cases = select_cases(
            enriched_rows
        )

        print(
            f"[5] PASS: cases={len(selected_cases)}",
            flush=True,
        )

        for selected_index, selected_row in enumerate(
            selected_cases,
            start=1,
        ):
            print(
                (
                    f"    case {selected_index}: "
                    f"{selected_row['case_name']}, "
                    f"sample={selected_row['sample_index_int']}, "
                    f"SNR={selected_row['target_snr_db_float']}"
                ),
                flush=True,
            )

        reconstructed_cases: List[
            Dict[str, object]
        ] = []

        for case_index, row in enumerate(
            selected_cases,
            start=1,
        ):
            print(
                (
                    f"[6.{case_index}] Reconstruct "
                    f"sample={row['sample_index_int']}, "
                    f"SNR={row['target_snr_db_float']}"
                ),
                flush=True,
            )

            reconstructed_case = reconstruct_case(
                row=row,
                intensity_dataset=h5_file[
                    "intensity"
                ],
                mask_dataset=h5_file[
                    "visible_mask"
                ],
            )

            reconstructed_cases.append(
                reconstructed_case
            )

            print(
                f"[6.{case_index}] PASS",
                flush=True,
            )

    print(
        "[7] HDF5 closed",
        flush=True,
    )

    print(
        "[8] Save selected-case CSV",
        flush=True,
    )

    save_selected_cases(
        reconstructed_cases
    )

    print(
        "[8] PASS",
        flush=True,
    )

    print(
        "[9] Generate figure",
        flush=True,
    )

    plot_cases(
        reconstructed_cases
    )

    print(
        "[9] PASS",
        flush=True,
    )

    print(
        "[10] Generate report",
        flush=True,
    )

    report_text = build_report(
        reconstructed_cases
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    print(
        "[10] PASS",
        flush=True,
    )

    print(
        "",
        flush=True,
    )

    print(
        report_text,
        flush=True,
    )

    print(
        "",
        flush=True,
    )

    print(
        "=" * 78,
        flush=True,
    )

    print(
        "MC-VWLS FAILURE CASE FIGURE COMPLETE",
        flush=True,
    )

    print(
        "=" * 78,
        flush=True,
    )

    print(
        "Selected cases:",
        SELECTED_CASES_PATH,
        flush=True,
    )

    print(
        "PNG figure:",
        PNG_PATH,
        flush=True,
    )

    print(
        "PDF figure:",
        PDF_PATH,
        flush=True,
    )

    print(
        "Report:",
        REPORT_PATH,
        flush=True,
    )


if __name__ == "__main__":
    main()