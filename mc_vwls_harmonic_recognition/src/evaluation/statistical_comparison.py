"""
Paired statistical comparison of the four ablation methods.

Input:
    results/csv/mc_vwls_ablation_full_test_predictions.csv

Outputs:
    results/csv/ablation_pairwise_statistics.csv
    results/csv/ablation_accuracy_confidence_intervals.csv
    results/validation/ablation_statistical_report.txt
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

PREDICTION_PATH = (
    ROOT
    / "results"
    / "csv"
    / "mc_vwls_ablation_full_test_predictions.csv"
)

PAIRWISE_CSV_PATH = (
    ROOT
    / "results"
    / "csv"
    / "ablation_pairwise_statistics.csv"
)

CI_CSV_PATH = (
    ROOT
    / "results"
    / "csv"
    / "ablation_accuracy_confidence_intervals.csv"
)

REPORT_PATH = (
    ROOT
    / "results"
    / "validation"
    / "ablation_statistical_report.txt"
)

METHOD_NAMES = (
    "A0_DAF",
    "A1_MASK_ULS",
    "A2_RAW_VWLS",
    "A3_MC_VWLS",
)

EXPECTED_OBSERVATIONS_PER_METHOD = 33600

BOOTSTRAP_REPETITIONS = 10000
BOOTSTRAP_SEED = 20260804
CONFIDENCE_LEVEL = 0.95

EPSILON = 1.0e-15


def load_predictions() -> Dict[
    str,
    Dict[Tuple[int, float], Dict[str, int]],
]:
    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(
            f"Prediction CSV does not exist: {PREDICTION_PATH}"
        )

    predictions: Dict[
        str,
        Dict[Tuple[int, float], Dict[str, int]],
    ] = {
        method: {}
        for method in METHOD_NAMES
    }

    with PREDICTION_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        required_columns = {
            "method",
            "sample_index",
            "target_snr_db",
            "label_correct",
            "order_correct",
            "phase_correct",
        }

        missing_columns = (
            required_columns
            - set(reader.fieldnames or [])
        )

        if missing_columns:
            raise KeyError(
                "Prediction CSV is missing columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:
            method = row["method"]

            if method not in predictions:
                raise ValueError(
                    f"Unexpected method: {method}"
                )

            key = (
                int(row["sample_index"]),
                float(row["target_snr_db"]),
            )

            if key in predictions[method]:
                raise ValueError(
                    "Duplicate observation detected: "
                    f"method={method}, key={key}"
                )

            predictions[method][key] = {
                "label_correct": int(
                    row["label_correct"]
                ),
                "order_correct": int(
                    row["order_correct"]
                ),
                "phase_correct": int(
                    row["phase_correct"]
                ),
            }

    for method in METHOD_NAMES:
        count = len(predictions[method])

        if count != EXPECTED_OBSERVATIONS_PER_METHOD:
            raise ValueError(
                f"Unexpected observation count for {method}: "
                f"expected {EXPECTED_OBSERVATIONS_PER_METHOD}, "
                f"found {count}"
            )

    reference_keys = set(
        predictions[METHOD_NAMES[0]].keys()
    )

    for method in METHOD_NAMES[1:]:
        method_keys = set(
            predictions[method].keys()
        )

        if method_keys != reference_keys:
            missing = reference_keys - method_keys
            extra = method_keys - reference_keys

            raise ValueError(
                f"Observation mismatch for {method}. "
                f"Missing={len(missing)}, extra={len(extra)}"
            )

    return predictions


def normal_survival_function(
    z_value: float,
) -> float:
    return 0.5 * math.erfc(
        float(z_value)
        / math.sqrt(2.0)
    )


def exact_two_sided_binomial_pvalue(
    successes: int,
    trials: int,
) -> float:
    if trials <= 0:
        return 1.0

    lower_tail_limit = min(
        successes,
        trials - successes,
    )

    probability = 0.0

    for k in range(
        lower_tail_limit + 1
    ):
        probability += (
            math.comb(trials, k)
            * (0.5 ** trials)
        )

    return float(
        min(
            1.0,
            2.0 * probability,
        )
    )


def mcnemar_test(
    first_correct: np.ndarray,
    second_correct: np.ndarray,
) -> Dict[str, float]:
    first = np.asarray(
        first_correct,
        dtype=np.int8,
    )

    second = np.asarray(
        second_correct,
        dtype=np.int8,
    )

    if first.shape != second.shape:
        raise ValueError(
            "Paired arrays must have identical shapes."
        )

    first_only = int(
        np.count_nonzero(
            (first == 1)
            & (second == 0)
        )
    )

    second_only = int(
        np.count_nonzero(
            (first == 0)
            & (second == 1)
        )
    )

    discordant = (
        first_only
        + second_only
    )

    if discordant == 0:
        statistic = 0.0
        asymptotic_pvalue = 1.0
        exact_pvalue = 1.0
    else:
        statistic = float(
            (
                abs(
                    first_only
                    - second_only
                )
                - 1.0
            ) ** 2
            / discordant
        )

        asymptotic_pvalue = float(
            2.0
            * normal_survival_function(
                math.sqrt(statistic)
            )
        )

        exact_pvalue = (
            exact_two_sided_binomial_pvalue(
                successes=min(
                    first_only,
                    second_only,
                ),
                trials=discordant,
            )
        )

    return {
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "discordant_count": discordant,
        "mcnemar_statistic": statistic,
        "asymptotic_pvalue": asymptotic_pvalue,
        "exact_pvalue": exact_pvalue,
    }


def wilson_interval(
    successes: int,
    total: int,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> Tuple[float, float]:
    if total <= 0:
        return (
            float("nan"),
            float("nan"),
        )

    if not np.isclose(
        confidence_level,
        0.95,
    ):
        raise ValueError(
            "This script currently uses the fixed "
            "95% normal quantile."
        )

    z_value = 1.959963984540054

    proportion = (
        successes
        / total
    )

    denominator = (
        1.0
        + z_value ** 2
        / total
    )

    center = (
        proportion
        + z_value ** 2
        / (
            2.0
            * total
        )
    ) / denominator

    half_width = (
        z_value
        / denominator
        * math.sqrt(
            (
                proportion
                * (
                    1.0
                    - proportion
                )
                / total
            )
            + (
                z_value ** 2
                / (
                    4.0
                    * total ** 2
                )
            )
        )
    )

    return (
        float(
            max(
                0.0,
                center - half_width,
            )
        ),
        float(
            min(
                1.0,
                center + half_width,
            )
        ),
    )


def paired_bootstrap_difference(
    first_correct: np.ndarray,
    second_correct: np.ndarray,
) -> Dict[str, float]:
    first = np.asarray(
        first_correct,
        dtype=np.float64,
    )

    second = np.asarray(
        second_correct,
        dtype=np.float64,
    )

    if first.shape != second.shape:
        raise ValueError(
            "Paired arrays must have identical shapes."
        )

    paired_difference = (
        second
        - first
    )

    observed_difference = float(
        np.mean(
            paired_difference,
            dtype=np.float64,
        )
    )

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    observation_count = len(
        paired_difference
    )

    bootstrap_means = np.empty(
        BOOTSTRAP_REPETITIONS,
        dtype=np.float64,
    )

    batch_size = 100

    completed = 0

    while completed < BOOTSTRAP_REPETITIONS:
        current_batch = min(
            batch_size,
            BOOTSTRAP_REPETITIONS
            - completed,
        )

        indices = rng.integers(
            low=0,
            high=observation_count,
            size=(
                current_batch,
                observation_count,
            ),
            endpoint=False,
        )

        bootstrap_means[
            completed:
            completed + current_batch
        ] = np.mean(
            paired_difference[indices],
            axis=1,
            dtype=np.float64,
        )

        completed += current_batch

    alpha = (
        1.0
        - CONFIDENCE_LEVEL
    )

    lower = float(
        np.quantile(
            bootstrap_means,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            bootstrap_means,
            1.0 - alpha / 2.0,
        )
    )

    probability_positive = float(
        np.mean(
            bootstrap_means > 0.0,
            dtype=np.float64,
        )
    )

    probability_negative = float(
        np.mean(
            bootstrap_means < 0.0,
            dtype=np.float64,
        )
    )

    return {
        "accuracy_difference_second_minus_first": (
            observed_difference
        ),
        "bootstrap_ci_lower": lower,
        "bootstrap_ci_upper": upper,
        "bootstrap_probability_positive": (
            probability_positive
        ),
        "bootstrap_probability_negative": (
            probability_negative
        ),
    }


def benjamini_hochberg_adjustment(
    pvalues: List[float],
) -> List[float]:
    if not pvalues:
        return []

    values = np.asarray(
        pvalues,
        dtype=np.float64,
    )

    count = len(values)

    order = np.argsort(values)

    ranked = values[order]

    adjusted_ranked = np.empty_like(
        ranked
    )

    running_minimum = 1.0

    for reverse_index in range(
        count - 1,
        -1,
        -1,
    ):
        rank = reverse_index + 1

        adjusted_value = (
            ranked[reverse_index]
            * count
            / rank
        )

        running_minimum = min(
            running_minimum,
            adjusted_value,
        )

        adjusted_ranked[
            reverse_index
        ] = min(
            1.0,
            running_minimum,
        )

    adjusted = np.empty_like(
        adjusted_ranked
    )

    adjusted[order] = (
        adjusted_ranked
    )

    return adjusted.tolist()


def save_csv(
    path: Path,
    rows: List[Dict[str, object]],
) -> None:
    if not rows:
        raise ValueError(
            f"No rows available for {path}."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print("=" * 78)
    print("ABLATION PAIRED STATISTICAL COMPARISON")
    print("=" * 78)

    predictions = load_predictions()

    ordered_keys = sorted(
        predictions[METHOD_NAMES[0]].keys()
    )

    print(
        "Paired observations:",
        len(ordered_keys),
    )

    metrics = (
        "label_correct",
        "order_correct",
        "phase_correct",
    )

    method_arrays: Dict[
        str,
        Dict[str, np.ndarray],
    ] = defaultdict(dict)

    confidence_interval_rows: List[
        Dict[str, object]
    ] = []

    for method in METHOD_NAMES:
        for metric in metrics:
            values = np.asarray(
                [
                    predictions[
                        method
                    ][key][metric]
                    for key in ordered_keys
                ],
                dtype=np.int8,
            )

            method_arrays[
                method
            ][metric] = values

            successes = int(
                np.sum(
                    values,
                    dtype=np.int64,
                )
            )

            total = len(values)

            accuracy = float(
                successes
                / total
            )

            lower, upper = wilson_interval(
                successes=successes,
                total=total,
            )

            confidence_interval_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "successes": successes,
                    "total": total,
                    "accuracy": accuracy,
                    "wilson_95_ci_lower": lower,
                    "wilson_95_ci_upper": upper,
                    "wilson_95_ci_width": (
                        upper - lower
                    ),
                }
            )

    pairwise_rows: List[
        Dict[str, object]
    ] = []

    for metric in metrics:
        for first_index in range(
            len(METHOD_NAMES)
        ):
            for second_index in range(
                first_index + 1,
                len(METHOD_NAMES),
            ):
                first_method = (
                    METHOD_NAMES[
                        first_index
                    ]
                )

                second_method = (
                    METHOD_NAMES[
                        second_index
                    ]
                )

                first_values = (
                    method_arrays[
                        first_method
                    ][metric]
                )

                second_values = (
                    method_arrays[
                        second_method
                    ][metric]
                )

                first_accuracy = float(
                    np.mean(
                        first_values,
                        dtype=np.float64,
                    )
                )

                second_accuracy = float(
                    np.mean(
                        second_values,
                        dtype=np.float64,
                    )
                )

                mcnemar = mcnemar_test(
                    first_correct=first_values,
                    second_correct=second_values,
                )

                bootstrap = (
                    paired_bootstrap_difference(
                        first_correct=first_values,
                        second_correct=second_values,
                    )
                )

                pairwise_rows.append(
                    {
                        "metric": metric,
                        "first_method": first_method,
                        "second_method": second_method,
                        "sample_count": len(
                            first_values
                        ),
                        "first_accuracy": first_accuracy,
                        "second_accuracy": second_accuracy,
                        **mcnemar,
                        **bootstrap,
                    }
                )

    adjusted_pvalues = (
        benjamini_hochberg_adjustment(
            [
                float(
                    row["exact_pvalue"]
                )
                for row in pairwise_rows
            ]
        )
    )

    for row, adjusted_pvalue in zip(
        pairwise_rows,
        adjusted_pvalues,
    ):
        row[
            "bh_adjusted_exact_pvalue"
        ] = float(
            adjusted_pvalue
        )

        row[
            "significant_at_0_05"
        ] = int(
            adjusted_pvalue < 0.05
        )

    save_csv(
        CI_CSV_PATH,
        confidence_interval_rows,
    )

    save_csv(
        PAIRWISE_CSV_PATH,
        pairwise_rows,
    )

    report_lines = [
        "Ablation paired statistical comparison report",
        f"Prediction CSV: {PREDICTION_PATH}",
        f"Paired observations: {len(ordered_keys)}",
        (
            "Bootstrap repetitions: "
            f"{BOOTSTRAP_REPETITIONS}"
        ),
        (
            "Bootstrap seed: "
            f"{BOOTSTRAP_SEED}"
        ),
        "",
        "[95% WILSON CONFIDENCE INTERVALS]",
    ]

    for row in confidence_interval_rows:
        report_lines.append(
            (
                f"method={row['method']}, "
                f"metric={row['metric']}, "
                f"accuracy="
                f"{float(row['accuracy']):.8f}, "
                f"CI=["
                f"{float(row['wilson_95_ci_lower']):.8f}, "
                f"{float(row['wilson_95_ci_upper']):.8f}]"
            )
        )

    report_lines.extend(
        [
            "",
            "[PAIRWISE LABEL-ACCURACY COMPARISONS]",
        ]
    )

    for row in pairwise_rows:
        if row["metric"] != "label_correct":
            continue

        report_lines.append(
            (
                f"{row['first_method']} vs "
                f"{row['second_method']}: "
                f"first_accuracy="
                f"{float(row['first_accuracy']):.8f}, "
                f"second_accuracy="
                f"{float(row['second_accuracy']):.8f}, "
                f"difference(second-first)="
                f"{float(row['accuracy_difference_second_minus_first']):+.8f}, "
                f"bootstrap_95_CI=["
                f"{float(row['bootstrap_ci_lower']):+.8f}, "
                f"{float(row['bootstrap_ci_upper']):+.8f}], "
                f"first_only_correct="
                f"{int(row['first_only_correct'])}, "
                f"second_only_correct="
                f"{int(row['second_only_correct'])}, "
                f"exact_p="
                f"{float(row['exact_pvalue']):.10e}, "
                f"BH_adjusted_p="
                f"{float(row['bh_adjusted_exact_pvalue']):.10e}, "
                f"significant="
                f"{int(row['significant_at_0_05'])}"
            )
        )

    report_lines.extend(
        [
            "",
            f"Confidence interval CSV: {CI_CSV_PATH}",
            f"Pairwise statistics CSV: {PAIRWISE_CSV_PATH}",
        ]
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_text = "\n".join(
        report_lines
    )

    REPORT_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    print("")
    print(report_text)

    print("")
    print("=" * 78)
    print(
        "ABLATION STATISTICAL COMPARISON COMPLETE"
    )
    print("=" * 78)

    print(
        "Confidence intervals:",
        CI_CSV_PATH,
    )

    print(
        "Pairwise statistics:",
        PAIRWISE_CSV_PATH,
    )

    print(
        "Report:",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()