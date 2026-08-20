"""
Final consistency validation for all frozen MC-VWLS experiment outputs.

Checks:
    1. Full ablation prediction row counts and method coverage.
    2. A3 MC-VWLS label, order, and phase accuracies.
    3. Confusion-matrix dimensions, total count, and diagonal count.
    4. Confusion-matrix accuracy against A3 prediction accuracy.
    5. Selected failure-case count and reconstructed labels.
    6. Runtime benchmark method/scope coverage and positive timing values.
    7. Computational-complexity table method coverage.
    8. Critical figure, CSV, report, dataset, and split files.
    9. No critical output is empty.

Outputs:
    results/csv/final_experiment_consistency_checks.csv
    results/validation/final_experiment_consistency_report.txt
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[2]

PREDICTION_PATH = (
    ROOT
    / "results"
    / "csv"
    / "mc_vwls_ablation_full_test_predictions.csv"
)

CONFUSION_COUNT_PATH = (
    ROOT
    / "results"
    / "csv"
    / "mc_vwls_confusion_matrix_counts.csv"
)

CONFUSION_NORMALIZED_PATH = (
    ROOT
    / "results"
    / "csv"
    / "mc_vwls_confusion_matrix_normalized.csv"
)

FAILURE_CASE_PATH = (
    ROOT
    / "results"
    / "csv"
    / "mc_vwls_selected_failure_cases.csv"
)

RUNTIME_PATH = (
    ROOT
    / "results"
    / "csv"
    / "mc_vwls_runtime_benchmark.csv"
)

COMPLEXITY_PATH = (
    ROOT
    / "results"
    / "csv"
    / "mc_vwls_complexity_summary.csv"
)

OUTPUT_CHECK_CSV_PATH = (
    ROOT
    / "results"
    / "csv"
    / "final_experiment_consistency_checks.csv"
)

REPORT_PATH = (
    ROOT
    / "results"
    / "validation"
    / "final_experiment_consistency_report.txt"
)

EXPECTED_METHODS = (
    "A0_DAF",
    "A1_MASK_ULS",
    "A2_RAW_VWLS",
    "A3_MC_VWLS",
)

EXPECTED_TIMING_SCOPES = (
    "recognition_only",
    "end_to_end",
)

EXPECTED_CLASS_COUNT = 32
EXPECTED_OBSERVATIONS_PER_METHOD = 33600
EXPECTED_TOTAL_PREDICTION_ROWS = 134400
EXPECTED_FAILURE_CASE_COUNT = 4

EXPECTED_A3_LABEL_ACCURACY = 0.848125
EXPECTED_A3_ORDER_ACCURACY = 0.8934523809523809
EXPECTED_A3_PHASE_ACCURACY = 0.8579166666666667

EXPECTED_CONFUSION_TOTAL = 33600
EXPECTED_CONFUSION_DIAGONAL = 28497

ABSOLUTE_TOLERANCE = 1.0e-12


CRITICAL_OUTPUTS = (
    "data/generated/turbulence_base_v1.h5",
    "data/generated/occlusion_clean_v2.h5",
    "data/manifest/sample_split_v1.npz",
    (
        "results/csv/"
        "mc_vwls_ablation_full_test_predictions.csv"
    ),
    (
        "results/csv/"
        "mc_vwls_ablation_summary.csv"
    ),
    (
        "results/csv/"
        "mc_vwls_confusion_matrix_counts.csv"
    ),
    (
        "results/csv/"
        "mc_vwls_confusion_matrix_normalized.csv"
    ),
    (
        "results/csv/"
        "mc_vwls_selected_failure_cases.csv"
    ),
    (
        "results/csv/"
        "mc_vwls_runtime_benchmark.csv"
    ),
    (
        "results/csv/"
        "mc_vwls_complexity_summary.csv"
    ),
    (
        "results/figures/"
        "fig_mc_vwls_confusion_matrix.png"
    ),
    (
        "results/figures/"
        "fig_mc_vwls_confusion_matrix.pdf"
    ),
    (
        "results/figures/"
        "fig_mc_vwls_failure_cases.png"
    ),
    (
        "results/figures/"
        "fig_mc_vwls_failure_cases.pdf"
    ),
    (
        "results/figures/"
        "fig_mc_vwls_runtime.png"
    ),
    (
        "results/figures/"
        "fig_mc_vwls_runtime.pdf"
    ),
    (
        "results/validation/"
        "mc_vwls_ablation_full_test_report.txt"
    ),
    (
        "results/validation/"
        "mc_vwls_failure_cases_report.txt"
    ),
    (
        "results/validation/"
        "mc_vwls_runtime_complexity_report.txt"
    ),
    (
        "results/validation/"
        "experiment_results_inventory_report.txt"
    ),
    "docs/experiment_results_draft.md",
)


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    category: str
    description: str
    expected: str
    actual: str
    status: str
    details: str


def stage(
    message: str,
) -> None:
    print(
        message,
        flush=True,
    )


def make_check(
    *,
    check_id: str,
    category: str,
    description: str,
    expected: object,
    actual: object,
    passed: bool,
    details: str = "",
) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        category=category,
        description=description,
        expected=str(
            expected
        ),
        actual=str(
            actual
        ),
        status=(
            "PASS"
            if passed
            else "FAIL"
        ),
        details=details,
    )


def load_csv_rows(
    path: Path,
) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"CSV file does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file
        )

        rows = list(
            reader
        )

    return rows


def parse_integer(
    value: object,
) -> int:
    return int(
        round(
            float(
                value
            )
        )
    )


def parse_float(
    value: object,
) -> float:
    return float(
        value
    )


def nearly_equal(
    actual: float,
    expected: float,
    *,
    tolerance: float = ABSOLUTE_TOLERANCE,
) -> bool:
    return math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def calculate_binary_accuracy(
    rows: Sequence[Dict[str, str]],
    column: str,
) -> float:
    if not rows:
        raise ValueError(
            "Cannot calculate accuracy from an empty row set."
        )

    correct_count = sum(
        parse_integer(
            row[
                column
            ]
        )
        for row in rows
    )

    return (
        correct_count
        / len(
            rows
        )
    )


def validate_prediction_table() -> tuple[
    List[CheckResult],
    Dict[str, float],
]:
    rows = load_csv_rows(
        PREDICTION_PATH
    )

    checks: List[
        CheckResult
    ] = []

    checks.append(
        make_check(
            check_id="PRED-001",
            category="prediction",
            description="Total ablation prediction rows",
            expected=EXPECTED_TOTAL_PREDICTION_ROWS,
            actual=len(
                rows
            ),
            passed=(
                len(
                    rows
                )
                == EXPECTED_TOTAL_PREDICTION_ROWS
            ),
        )
    )

    method_counts: Dict[
        str,
        int,
    ] = {}

    for row in rows:
        method = row[
            "method"
        ]

        method_counts[
            method
        ] = (
            method_counts.get(
                method,
                0,
            )
            + 1
        )

    observed_methods = tuple(
        sorted(
            method_counts
        )
    )

    expected_methods = tuple(
        sorted(
            EXPECTED_METHODS
        )
    )

    checks.append(
        make_check(
            check_id="PRED-002",
            category="prediction",
            description="Prediction method coverage",
            expected=expected_methods,
            actual=observed_methods,
            passed=(
                observed_methods
                == expected_methods
            ),
        )
    )

    for method in EXPECTED_METHODS:
        actual_count = method_counts.get(
            method,
            0,
        )

        checks.append(
            make_check(
                check_id=(
                    "PRED-COUNT-"
                    + method
                ),
                category="prediction",
                description=(
                    f"Observation count for {method}"
                ),
                expected=EXPECTED_OBSERVATIONS_PER_METHOD,
                actual=actual_count,
                passed=(
                    actual_count
                    == EXPECTED_OBSERVATIONS_PER_METHOD
                ),
            )
        )

    a3_rows = [
        row
        for row in rows
        if row[
            "method"
        ] == "A3_MC_VWLS"
    ]

    label_accuracy = calculate_binary_accuracy(
        a3_rows,
        "label_correct",
    )

    order_accuracy = calculate_binary_accuracy(
        a3_rows,
        "order_correct",
    )

    phase_accuracy = calculate_binary_accuracy(
        a3_rows,
        "phase_correct",
    )

    checks.append(
        make_check(
            check_id="PRED-A3-LABEL",
            category="prediction",
            description="A3 joint-label accuracy",
            expected=(
                f"{EXPECTED_A3_LABEL_ACCURACY:.12f}"
            ),
            actual=(
                f"{label_accuracy:.12f}"
            ),
            passed=nearly_equal(
                label_accuracy,
                EXPECTED_A3_LABEL_ACCURACY,
            ),
        )
    )

    checks.append(
        make_check(
            check_id="PRED-A3-ORDER",
            category="prediction",
            description="A3 OAM-order accuracy",
            expected=(
                f"{EXPECTED_A3_ORDER_ACCURACY:.12f}"
            ),
            actual=(
                f"{order_accuracy:.12f}"
            ),
            passed=nearly_equal(
                order_accuracy,
                EXPECTED_A3_ORDER_ACCURACY,
            ),
        )
    )

    checks.append(
        make_check(
            check_id="PRED-A3-PHASE",
            category="prediction",
            description="A3 phase-bin accuracy",
            expected=(
                f"{EXPECTED_A3_PHASE_ACCURACY:.12f}"
            ),
            actual=(
                f"{phase_accuracy:.12f}"
            ),
            passed=nearly_equal(
                phase_accuracy,
                EXPECTED_A3_PHASE_ACCURACY,
            ),
        )
    )

    duplicate_keys: set[
        tuple[str, int, float]
    ] = set()

    observed_keys: set[
        tuple[str, int, float]
    ] = set()

    for row in rows:
        key = (
            row[
                "method"
            ],
            parse_integer(
                row[
                    "sample_index"
                ]
            ),
            parse_float(
                row[
                    "target_snr_db"
                ]
            ),
        )

        if key in observed_keys:
            duplicate_keys.add(
                key
            )

        observed_keys.add(
            key
        )

    checks.append(
        make_check(
            check_id="PRED-003",
            category="prediction",
            description=(
                "Unique method/sample/SNR prediction keys"
            ),
            expected=0,
            actual=len(
                duplicate_keys
            ),
            passed=(
                len(
                    duplicate_keys
                )
                == 0
            ),
            details=(
                ""
                if not duplicate_keys
                else str(
                    sorted(
                        duplicate_keys
                    )[
                        :10
                    ]
                )
            ),
        )
    )

    metrics = {
        "a3_label_accuracy": label_accuracy,
        "a3_order_accuracy": order_accuracy,
        "a3_phase_accuracy": phase_accuracy,
        "a3_row_count": float(
            len(
                a3_rows
            )
        ),
    }

    return (
        checks,
        metrics,
    )


def identify_numeric_columns(
    rows: Sequence[Dict[str, str]],
) -> List[str]:
    if not rows:
        return []

    columns = list(
        rows[0].keys()
    )

    numeric_columns: List[
        str
    ] = []

    for column in columns:
        is_numeric = True

        for row in rows:
            value = row.get(
                column,
                "",
            )

            if value is None or value.strip() == "":
                is_numeric = False
                break

            try:
                float(
                    value
                )
            except ValueError:
                is_numeric = False
                break

        if is_numeric:
            numeric_columns.append(
                column
            )

    return numeric_columns


def load_confusion_matrix(
    path: Path,
) -> List[List[float]]:
    rows = load_csv_rows(
        path
    )

    if len(
        rows
    ) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"Confusion matrix must contain "
            f"{EXPECTED_CLASS_COUNT} rows: {path}"
        )

    numeric_columns = identify_numeric_columns(
        rows
    )

    if len(
        numeric_columns
    ) < EXPECTED_CLASS_COUNT:
        raise ValueError(
            "Could not identify 32 numeric confusion-matrix "
            f"columns in {path}. "
            f"Numeric columns={numeric_columns}"
        )

    matrix_columns = numeric_columns[
        -EXPECTED_CLASS_COUNT:
    ]

    matrix: List[
        List[float]
    ] = []

    for row in rows:
        matrix.append(
            [
                float(
                    row[
                        column
                    ]
                )
                for column in matrix_columns
            ]
        )

    return matrix


def validate_confusion_matrices(
    prediction_metrics: Dict[str, float],
) -> List[CheckResult]:
    checks: List[
        CheckResult
    ] = []

    count_matrix = load_confusion_matrix(
        CONFUSION_COUNT_PATH
    )

    normalized_matrix = load_confusion_matrix(
        CONFUSION_NORMALIZED_PATH
    )

    count_shape = (
        len(
            count_matrix
        ),
        len(
            count_matrix[0]
        ),
    )

    normalized_shape = (
        len(
            normalized_matrix
        ),
        len(
            normalized_matrix[0]
        ),
    )

    checks.append(
        make_check(
            check_id="CONF-001",
            category="confusion",
            description="Count matrix dimensions",
            expected=(
                EXPECTED_CLASS_COUNT,
                EXPECTED_CLASS_COUNT,
            ),
            actual=count_shape,
            passed=(
                count_shape
                == (
                    EXPECTED_CLASS_COUNT,
                    EXPECTED_CLASS_COUNT,
                )
            ),
        )
    )

    checks.append(
        make_check(
            check_id="CONF-002",
            category="confusion",
            description="Normalized matrix dimensions",
            expected=(
                EXPECTED_CLASS_COUNT,
                EXPECTED_CLASS_COUNT,
            ),
            actual=normalized_shape,
            passed=(
                normalized_shape
                == (
                    EXPECTED_CLASS_COUNT,
                    EXPECTED_CLASS_COUNT,
                )
            ),
        )
    )

    total_count_float = sum(
        sum(
            row
        )
        for row in count_matrix
    )

    total_count = int(
        round(
            total_count_float
        )
    )

    diagonal_count = int(
        round(
            sum(
                count_matrix[
                    class_index
                ][
                    class_index
                ]
                for class_index in range(
                    EXPECTED_CLASS_COUNT
                )
            )
        )
    )

    checks.append(
        make_check(
            check_id="CONF-003",
            category="confusion",
            description="Confusion-matrix total count",
            expected=EXPECTED_CONFUSION_TOTAL,
            actual=total_count,
            passed=(
                total_count
                == EXPECTED_CONFUSION_TOTAL
            ),
        )
    )

    checks.append(
        make_check(
            check_id="CONF-004",
            category="confusion",
            description="Confusion-matrix diagonal count",
            expected=EXPECTED_CONFUSION_DIAGONAL,
            actual=diagonal_count,
            passed=(
                diagonal_count
                == EXPECTED_CONFUSION_DIAGONAL
            ),
        )
    )

    confusion_accuracy = (
        diagonal_count
        / max(
            total_count,
            1,
        )
    )

    prediction_accuracy = prediction_metrics[
        "a3_label_accuracy"
    ]

    checks.append(
        make_check(
            check_id="CONF-005",
            category="confusion",
            description=(
                "Confusion accuracy matches A3 prediction accuracy"
            ),
            expected=(
                f"{prediction_accuracy:.12f}"
            ),
            actual=(
                f"{confusion_accuracy:.12f}"
            ),
            passed=nearly_equal(
                confusion_accuracy,
                prediction_accuracy,
            ),
        )
    )

    invalid_normalized_values = 0

    for row in normalized_matrix:
        for value in row:
            if (
                not math.isfinite(
                    value
                )
                or value < -ABSOLUTE_TOLERANCE
                or value > 1.0 + ABSOLUTE_TOLERANCE
            ):
                invalid_normalized_values += 1

    checks.append(
        make_check(
            check_id="CONF-006",
            category="confusion",
            description=(
                "Normalized confusion values lie in [0, 1]"
            ),
            expected=0,
            actual=invalid_normalized_values,
            passed=(
                invalid_normalized_values
                == 0
            ),
        )
    )

    invalid_row_sums: List[
        tuple[int, float]
    ] = []

    for row_index, row in enumerate(
        normalized_matrix
    ):
        row_sum = sum(
            row
        )

        if not math.isclose(
            row_sum,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            invalid_row_sums.append(
                (
                    row_index,
                    row_sum,
                )
            )

    checks.append(
        make_check(
            check_id="CONF-007",
            category="confusion",
            description=(
                "Normalized confusion rows sum to one"
            ),
            expected=0,
            actual=len(
                invalid_row_sums
            ),
            passed=(
                len(
                    invalid_row_sums
                )
                == 0
            ),
            details=str(
                invalid_row_sums[
                    :10
                ]
            ),
        )
    )

    return checks


def validate_failure_cases() -> List[CheckResult]:
    rows = load_csv_rows(
        FAILURE_CASE_PATH
    )

    checks: List[
        CheckResult
    ] = []

    checks.append(
        make_check(
            check_id="FAILCASE-001",
            category="failure_case",
            description="Selected representative-case count",
            expected=EXPECTED_FAILURE_CASE_COUNT,
            actual=len(
                rows
            ),
            passed=(
                len(
                    rows
                )
                == EXPECTED_FAILURE_CASE_COUNT
            ),
        )
    )

    expected_case_names = {
        "Correct: weak turbulence",
        "Correct: severe turbulence",
        "Failure: adjacent phase",
        "Failure: cross order",
    }

    actual_case_names = {
        row[
            "case_name"
        ]
        for row in rows
    }

    checks.append(
        make_check(
            check_id="FAILCASE-002",
            category="failure_case",
            description="Representative-case category coverage",
            expected=sorted(
                expected_case_names
            ),
            actual=sorted(
                actual_case_names
            ),
            passed=(
                actual_case_names
                == expected_case_names
            ),
        )
    )

    invalid_label_consistency = 0

    for row in rows:
        true_order = parse_integer(
            row[
                "true_order"
            ]
        )

        true_phase = parse_integer(
            row[
                "true_phase_bin"
            ]
        )

        predicted_order = parse_integer(
            row[
                "predicted_order"
            ]
        )

        predicted_phase = parse_integer(
            row[
                "predicted_phase_bin"
            ]
        )

        label_correct = parse_integer(
            row[
                "label_correct"
            ]
        )

        calculated_label_correct = int(
            true_order
            == predicted_order
            and true_phase
            == predicted_phase
        )

        if (
            label_correct
            != calculated_label_correct
        ):
            invalid_label_consistency += 1

    checks.append(
        make_check(
            check_id="FAILCASE-003",
            category="failure_case",
            description=(
                "Selected-case correctness flags match labels"
            ),
            expected=0,
            actual=invalid_label_consistency,
            passed=(
                invalid_label_consistency
                == 0
            ),
        )
    )

    return checks


def validate_runtime_table() -> List[CheckResult]:
    rows = load_csv_rows(
        RUNTIME_PATH
    )

    checks: List[
        CheckResult
    ] = []

    expected_row_count = (
        len(
            EXPECTED_METHODS
        )
        * len(
            EXPECTED_TIMING_SCOPES
        )
    )

    checks.append(
        make_check(
            check_id="RUNTIME-001",
            category="runtime",
            description="Runtime benchmark row count",
            expected=expected_row_count,
            actual=len(
                rows
            ),
            passed=(
                len(
                    rows
                )
                == expected_row_count
            ),
        )
    )

    observed_pairs = {
        (
            row[
                "method"
            ],
            row[
                "timing_scope"
            ],
        )
        for row in rows
    }

    expected_pairs = {
        (
            method,
            timing_scope,
        )
        for method in EXPECTED_METHODS
        for timing_scope in EXPECTED_TIMING_SCOPES
    }

    checks.append(
        make_check(
            check_id="RUNTIME-002",
            category="runtime",
            description="Runtime method/scope coverage",
            expected=sorted(
                expected_pairs
            ),
            actual=sorted(
                observed_pairs
            ),
            passed=(
                observed_pairs
                == expected_pairs
            ),
        )
    )

    invalid_runtime_rows: List[
        str
    ] = []

    for row in rows:
        mean_ms = parse_float(
            row[
                "mean_ms_per_image"
            ]
        )

        standard_deviation_ms = parse_float(
            row[
                "std_ms_per_image"
            ]
        )

        throughput = parse_float(
            row[
                "throughput_images_per_second"
            ]
        )

        observation_count = parse_integer(
            row[
                "observation_count"
            ]
        )

        repeat_count = parse_integer(
            row[
                "repeat_count"
            ]
        )

        if (
            not math.isfinite(
                mean_ms
            )
            or mean_ms <= 0.0
            or not math.isfinite(
                standard_deviation_ms
            )
            or standard_deviation_ms < 0.0
            or not math.isfinite(
                throughput
            )
            or throughput <= 0.0
            or observation_count != 256
            or repeat_count != 5
        ):
            invalid_runtime_rows.append(
                (
                    f"{row['method']}/"
                    f"{row['timing_scope']}"
                )
            )

    checks.append(
        make_check(
            check_id="RUNTIME-003",
            category="runtime",
            description="Runtime values and protocol fields",
            expected=0,
            actual=len(
                invalid_runtime_rows
            ),
            passed=(
                len(
                    invalid_runtime_rows
                )
                == 0
            ),
            details=str(
                invalid_runtime_rows
            ),
        )
    )

    runtime_map = {
        (
            row[
                "method"
            ],
            row[
                "timing_scope"
            ],
        ): parse_float(
            row[
                "mean_ms_per_image"
            ]
        )
        for row in rows
    }

    scope_order_failures: List[
        str
    ] = []

    for method in EXPECTED_METHODS:
        recognition_time = runtime_map[
            (
                method,
                "recognition_only",
            )
        ]

        end_to_end_time = runtime_map[
            (
                method,
                "end_to_end",
            )
        ]

        if (
            end_to_end_time
            < recognition_time
        ):
            scope_order_failures.append(
                method
            )

    checks.append(
        make_check(
            check_id="RUNTIME-004",
            category="runtime",
            description=(
                "End-to-end time is not below recognition-only time"
            ),
            expected=0,
            actual=len(
                scope_order_failures
            ),
            passed=(
                len(
                    scope_order_failures
                )
                == 0
            ),
            details=str(
                scope_order_failures
            ),
        )
    )

    return checks


def validate_complexity_table() -> List[CheckResult]:
    rows = load_csv_rows(
        COMPLEXITY_PATH
    )

    checks: List[
        CheckResult
    ] = []

    observed_methods = {
        row[
            "method"
        ]
        for row in rows
    }

    checks.append(
        make_check(
            check_id="COMPLEX-001",
            category="complexity",
            description="Complexity-table row count",
            expected=len(
                EXPECTED_METHODS
            ),
            actual=len(
                rows
            ),
            passed=(
                len(
                    rows
                )
                == len(
                    EXPECTED_METHODS
                )
            ),
        )
    )

    checks.append(
        make_check(
            check_id="COMPLEX-002",
            category="complexity",
            description="Complexity-table method coverage",
            expected=sorted(
                EXPECTED_METHODS
            ),
            actual=sorted(
                observed_methods
            ),
            passed=(
                observed_methods
                == set(
                    EXPECTED_METHODS
                )
            ),
        )
    )

    invalid_parameter_rows: List[
        str
    ] = []

    for row in rows:
        candidate_count = parse_integer(
            row[
                "candidate_count_K"
            ]
        )

        angular_samples = parse_integer(
            row[
                "angular_samples_N_theta"
            ]
        )

        radial_samples = parse_integer(
            row[
                "radial_samples_N_r"
            ]
        )

        polar_grid_elements = parse_integer(
            row[
                "polar_grid_elements"
            ]
        )

        if (
            candidate_count != 4
            or angular_samples != 180
            or radial_samples != 64
            or polar_grid_elements != 11520
        ):
            invalid_parameter_rows.append(
                row[
                    "method"
                ]
            )

    checks.append(
        make_check(
            check_id="COMPLEX-003",
            category="complexity",
            description=(
                "Complexity-table frozen parameter values"
            ),
            expected=0,
            actual=len(
                invalid_parameter_rows
            ),
            passed=(
                len(
                    invalid_parameter_rows
                )
                == 0
            ),
            details=str(
                invalid_parameter_rows
            ),
        )
    )

    return checks


def validate_critical_outputs() -> List[CheckResult]:
    checks: List[
        CheckResult
    ] = []

    missing_files: List[
        str
    ] = []

    empty_files: List[
        str
    ] = []

    for relative_path in CRITICAL_OUTPUTS:
        path = (
            ROOT
            / relative_path
        )

        if not path.exists():
            missing_files.append(
                relative_path
            )

            continue

        if (
            not path.is_file()
            or path.stat().st_size <= 0
        ):
            empty_files.append(
                relative_path
            )

    checks.append(
        make_check(
            check_id="FILES-001",
            category="files",
            description="Missing critical experiment outputs",
            expected=0,
            actual=len(
                missing_files
            ),
            passed=(
                len(
                    missing_files
                )
                == 0
            ),
            details=str(
                missing_files
            ),
        )
    )

    checks.append(
        make_check(
            check_id="FILES-002",
            category="files",
            description="Empty critical experiment outputs",
            expected=0,
            actual=len(
                empty_files
            ),
            passed=(
                len(
                    empty_files
                )
                == 0
            ),
            details=str(
                empty_files
            ),
        )
    )

    return checks


def save_check_csv(
    checks: Sequence[CheckResult],
) -> None:
    OUTPUT_CHECK_CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "check_id",
        "category",
        "description",
        "expected",
        "actual",
        "status",
        "details",
    ]

    with OUTPUT_CHECK_CSV_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for check in checks:
            writer.writerow(
                {
                    "check_id": check.check_id,
                    "category": check.category,
                    "description": check.description,
                    "expected": check.expected,
                    "actual": check.actual,
                    "status": check.status,
                    "details": check.details,
                }
            )


def build_report(
    checks: Sequence[CheckResult],
    prediction_metrics: Dict[str, float],
) -> str:
    passed_checks = [
        check
        for check in checks
        if check.status == "PASS"
    ]

    failed_checks = [
        check
        for check in checks
        if check.status == "FAIL"
    ]

    category_counts: Dict[
        str,
        Dict[str, int],
    ] = {}

    for check in checks:
        if check.category not in category_counts:
            category_counts[
                check.category
            ] = {
                "PASS": 0,
                "FAIL": 0,
            }

        category_counts[
            check.category
        ][
            check.status
        ] += 1

    overall_status = (
        "PASS"
        if not failed_checks
        else "FAIL"
    )

    lines = [
        "Final MC-VWLS experiment consistency validation",
        "",
        "[SUMMARY]",
        f"Status={overall_status}",
        f"Total checks={len(checks)}",
        f"Passed checks={len(passed_checks)}",
        f"Failed checks={len(failed_checks)}",
        "",
        "[FROZEN A3 RESULTS]",
        (
            "Joint-label accuracy="
            f"{prediction_metrics['a3_label_accuracy']:.12f}"
        ),
        (
            "OAM-order accuracy="
            f"{prediction_metrics['a3_order_accuracy']:.12f}"
        ),
        (
            "Phase-bin accuracy="
            f"{prediction_metrics['a3_phase_accuracy']:.12f}"
        ),
        (
            "A3 prediction observations="
            f"{int(prediction_metrics['a3_row_count'])}"
        ),
        "",
        "[CATEGORY RESULTS]",
    ]

    for category in sorted(
        category_counts
    ):
        counts = category_counts[
            category
        ]

        lines.append(
            (
                f"{category}: "
                f"PASS={counts['PASS']}, "
                f"FAIL={counts['FAIL']}"
            )
        )

    lines.extend(
        [
            "",
            "[FAILED CHECKS]",
        ]
    )

    if failed_checks:
        for check in failed_checks:
            lines.append(
                (
                    f"{check.check_id}: "
                    f"{check.description}; "
                    f"expected={check.expected}; "
                    f"actual={check.actual}; "
                    f"details={check.details}"
                )
            )
    else:
        lines.append(
            "None"
        )

    lines.extend(
        [
            "",
            "[ALL CHECKS]",
        ]
    )

    for check in checks:
        lines.append(
            (
                f"{check.status}: "
                f"{check.check_id}, "
                f"{check.description}, "
                f"expected={check.expected}, "
                f"actual={check.actual}"
            )
        )

    lines.extend(
        [
            "",
            "[OUTPUTS]",
            f"Check CSV: {OUTPUT_CHECK_CSV_PATH}",
            f"Validation report: {REPORT_PATH}",
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
        "FINAL MC-VWLS EXPERIMENT CONSISTENCY VALIDATION"
    )

    stage(
        "=" * 78
    )

    all_checks: List[
        CheckResult
    ] = []

    stage(
        "[1] Validate full prediction table"
    )

    (
        prediction_checks,
        prediction_metrics,
    ) = validate_prediction_table()

    all_checks.extend(
        prediction_checks
    )

    stage(
        f"[1] PASS: checks={len(prediction_checks)}"
    )

    stage(
        "[2] Validate confusion matrices"
    )

    confusion_checks = validate_confusion_matrices(
        prediction_metrics
    )

    all_checks.extend(
        confusion_checks
    )

    stage(
        f"[2] PASS: checks={len(confusion_checks)}"
    )

    stage(
        "[3] Validate representative failure cases"
    )

    failure_case_checks = validate_failure_cases()

    all_checks.extend(
        failure_case_checks
    )

    stage(
        f"[3] PASS: checks={len(failure_case_checks)}"
    )

    stage(
        "[4] Validate runtime benchmark"
    )

    runtime_checks = validate_runtime_table()

    all_checks.extend(
        runtime_checks
    )

    stage(
        f"[4] PASS: checks={len(runtime_checks)}"
    )

    stage(
        "[5] Validate complexity table"
    )

    complexity_checks = validate_complexity_table()

    all_checks.extend(
        complexity_checks
    )

    stage(
        f"[5] PASS: checks={len(complexity_checks)}"
    )

    stage(
        "[6] Validate critical output files"
    )

    file_checks = validate_critical_outputs()

    all_checks.extend(
        file_checks
    )

    stage(
        f"[6] PASS: checks={len(file_checks)}"
    )

    stage(
        "[7] Save check CSV"
    )

    save_check_csv(
        all_checks
    )

    stage(
        f"[7] PASS: {OUTPUT_CHECK_CSV_PATH}"
    )

    stage(
        "[8] Generate final validation report"
    )

    report_text = build_report(
        checks=all_checks,
        prediction_metrics=prediction_metrics,
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
        "[8] PASS"
    )

    stage(
        ""
    )

    stage(
        report_text
    )

    failed_checks = [
        check
        for check in all_checks
        if check.status == "FAIL"
    ]

    stage(
        ""
    )

    stage(
        "=" * 78
    )

    if failed_checks:
        stage(
            "FINAL EXPERIMENT CONSISTENCY VALIDATION FAILED"
        )

        stage(
            "=" * 78
        )

        raise SystemExit(
            1
        )

    stage(
        "FINAL EXPERIMENT CONSISTENCY VALIDATION PASSED"
    )

    stage(
        "=" * 78
    )


if __name__ == "__main__":
    main()