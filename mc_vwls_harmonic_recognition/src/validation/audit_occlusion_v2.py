"""
Audit corrected random local-occlusion dataset v2.

Checks:
1. HDF5 structure and metadata.
2. Label and condition balance.
3. Binary-mask validity.
4. Target versus achieved energy-occlusion error.
5. Intensity-mask consistency.
6. Nested-mask consistency.
7. Obstacle-center and radius distributions.
8. Sample power and finite-value checks.

The source file is opened read-only.
"""

from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

H5_PATH = (
    ROOT
    / "data"
    / "generated"
    / "occlusion_clean_v2.h5"
)

REPORT_PATH = (
    ROOT
    / "results"
    / "validation"
    / "occlusion_v2_audit.txt"
)

EXPECTED_SAMPLES = 44_800
EXPECTED_BASE_SAMPLES = 8_960
EXPECTED_STATES = 32
EXPECTED_LEVELS = 5

TARGET_LEVELS = np.asarray(
    [0.0, 0.1, 0.2, 0.3, 0.4],
    dtype=np.float64,
)

AUDIT_SAMPLE_COUNT = 512
AUDIT_BASE_GROUP_COUNT = 256
AUDIT_SEED = 20260804

ZERO_TOLERANCE = 1.0e-12


def summary(values: np.ndarray) -> Dict[str, float]:
    """Return descriptive statistics."""

    array = np.asarray(values, dtype=np.float64)

    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
    }


def format_summary(
    name: str,
    values: np.ndarray,
) -> str:
    """Format descriptive statistics."""

    stats = summary(values)

    return (
        f"{name}: "
        f"n={stats['count']}, "
        f"min={stats['min']:.10f}, "
        f"mean={stats['mean']:.10f}, "
        f"median={stats['median']:.10f}, "
        f"max={stats['max']:.10f}, "
        f"std={stats['std']:.10f}"
    )


def validate_structure(
    h5: h5py.File,
    report: List[str],
) -> bool:
    """Validate required datasets and shapes."""

    structure_ok = True

    required_keys = {
        "intensity",
        "visible_mask",
        "labels",
        "conditions",
        "base_indices",
        "legacy_source_indices",
    }

    missing_keys = required_keys.difference(h5.keys())

    report.append("SECTION 1 — Structure and metadata")
    report.append(f"HDF5 keys: {list(h5.keys())}")

    if missing_keys:
        report.append(
            f"[FAIL] Missing datasets: {sorted(missing_keys)}"
        )
        return False

    expected_shapes = {
        "intensity": (
            EXPECTED_SAMPLES,
            256,
            256,
        ),
        "visible_mask": (
            EXPECTED_SAMPLES,
            256,
            256,
        ),
        "labels": (
            EXPECTED_SAMPLES,
        ),
        "conditions": (
            EXPECTED_SAMPLES,
            9,
        ),
        "base_indices": (
            EXPECTED_SAMPLES,
        ),
        "legacy_source_indices": (
            EXPECTED_SAMPLES,
        ),
    }

    for name, expected_shape in expected_shapes.items():
        dataset = h5[name]

        report.append(
            f"{name}: "
            f"shape={dataset.shape}, "
            f"dtype={dataset.dtype}, "
            f"chunks={dataset.chunks}, "
            f"compression={dataset.compression}"
        )

        if dataset.shape != expected_shape:
            structure_ok = False
            report.append(
                f"[FAIL] {name} expected shape "
                f"{expected_shape}, found {dataset.shape}."
            )

    report.append("")
    report.append("HDF5 attributes:")

    for key in sorted(h5.attrs.keys()):
        report.append(
            f"  {key}: {h5.attrs[key]}"
        )

    return structure_ok


def validate_balance(
    labels: np.ndarray,
    conditions: np.ndarray,
    base_indices: np.ndarray,
    report: List[str],
) -> bool:
    """Validate class, condition, and base-field balance."""

    balance_ok = True

    report.append("")
    report.append("SECTION 2 — Dataset balance")

    unique_labels, label_counts = np.unique(
        labels,
        return_counts=True,
    )

    report.append(
        f"Unique labels: {unique_labels.tolist()}"
    )
    report.append(
        f"Label counts min/max: "
        f"{int(label_counts.min())}/"
        f"{int(label_counts.max())}"
    )

    expected_per_label = (
        EXPECTED_SAMPLES // EXPECTED_STATES
    )

    if (
        len(unique_labels) != EXPECTED_STATES
        or not np.all(
            label_counts == expected_per_label
        )
    ):
        balance_ok = False
        report.append(
            "[FAIL] Label distribution is not balanced."
        )

    target_ratio = conditions[:, 3]

    unique_targets, target_counts = np.unique(
        target_ratio,
        return_counts=True,
    )

    report.append(
        f"Target levels: {unique_targets.tolist()}"
    )
    report.append(
        f"Target counts: {target_counts.tolist()}"
    )

    if (
        len(unique_targets) != EXPECTED_LEVELS
        or not np.allclose(
            unique_targets,
            TARGET_LEVELS,
            atol=1.0e-12,
        )
        or not np.all(
            target_counts == EXPECTED_BASE_SAMPLES
        )
    ):
        balance_ok = False
        report.append(
            "[FAIL] Occlusion-level distribution is incorrect."
        )

    unique_base, base_counts = np.unique(
        base_indices,
        return_counts=True,
    )

    report.append(
        f"Unique base indices: {len(unique_base)}"
    )
    report.append(
        f"Samples per base min/max: "
        f"{int(base_counts.min())}/"
        f"{int(base_counts.max())}"
    )

    if (
        len(unique_base) != EXPECTED_BASE_SAMPLES
        or not np.all(
            base_counts == EXPECTED_LEVELS
        )
    ):
        balance_ok = False
        report.append(
            "[FAIL] Each base field must have exactly five "
            "occlusion levels."
        )

    return balance_ok


def validate_occlusion_error(
    conditions: np.ndarray,
    report: List[str],
) -> bool:
    """Validate target-versus-achieved ratios."""

    error_ok = True

    target = conditions[:, 3]
    achieved = conditions[:, 4]

    absolute_error = np.abs(
        achieved - target
    )

    report.append("")
    report.append(
        "SECTION 3 — Energy-occlusion accuracy"
    )

    report.append(
        format_summary(
            "Absolute energy-ratio error",
            absolute_error,
        )
    )

    for level in TARGET_LEVELS:
        selected = np.isclose(
            target,
            level,
            atol=1.0e-12,
        )

        report.append(
            f"target={level:.1f}"
        )
        report.append(
            "  "
            + format_summary(
                "achieved ratio",
                achieved[selected],
            )
        )
        report.append(
            "  "
            + format_summary(
                "absolute error",
                absolute_error[selected],
            )
        )

    if not np.all(
        np.isfinite(achieved)
    ):
        error_ok = False
        report.append(
            "[FAIL] Non-finite achieved ratios detected."
        )

    if np.any(achieved < -1.0e-12):
        error_ok = False
        report.append(
            "[FAIL] Negative achieved ratio detected."
        )

    if np.any(achieved > 1.0):
        error_ok = False
        report.append(
            "[FAIL] Achieved ratio above one detected."
        )

    # Conservative acceptance limit:
    # no sample may differ from target by more than 0.2 percentage points.
    if float(np.max(absolute_error)) > 0.002:
        error_ok = False
        report.append(
            "[FAIL] Maximum energy-ratio error exceeds 0.002."
        )

    return error_ok


def validate_sample_content(
    intensity_ds: h5py.Dataset,
    mask_ds: h5py.Dataset,
    conditions: np.ndarray,
    rng: np.random.Generator,
    report: List[str],
) -> bool:
    """Validate representative intensity and mask samples."""

    content_ok = True

    report.append("")
    report.append(
        "SECTION 4 — Sample intensity and mask content"
    )

    selected_indices = rng.choice(
        EXPECTED_SAMPLES,
        size=min(
            AUDIT_SAMPLE_COUNT,
            EXPECTED_SAMPLES,
        ),
        replace=False,
    )

    powers = []
    visible_fractions = []
    blocked_maxima = []
    visible_negative_counts = []
    finite_flags = []
    mask_binary_flags = []
    ratio_recalculation_errors = []

    for index in selected_indices:
        index = int(index)

        intensity = np.asarray(
            intensity_ds[index],
            dtype=np.float64,
        )

        mask = np.asarray(
            mask_ds[index],
        )

        finite_flags.append(
            bool(np.all(np.isfinite(intensity)))
        )

        mask_binary_flags.append(
            bool(
                np.all(
                    (mask == 0)
                    | (mask == 1)
                )
            )
        )

        powers.append(
            float(
                np.sum(
                    intensity,
                    dtype=np.float64,
                )
            )
        )

        visible_fractions.append(
            float(np.mean(mask))
        )

        blocked_values = intensity[
            mask == 0
        ]

        if blocked_values.size == 0:
            blocked_maximum = 0.0
        else:
            blocked_maximum = float(
                np.max(
                    np.abs(blocked_values)
                )
            )

        blocked_maxima.append(
            blocked_maximum
        )

        visible_negative_counts.append(
            int(
                np.count_nonzero(
                    intensity[mask == 1] < 0
                )
            )
        )

        target_ratio = float(
            conditions[index, 3]
        )

        achieved_ratio = float(
            conditions[index, 4]
        )

        # The clean field was normalized to approximately unit power.
        # Recalculated blocked-energy fraction is therefore
        # approximately 1 - remaining power.
        remaining_power = float(
            np.sum(
                intensity,
                dtype=np.float64,
            )
        )

        recalculated_ratio = (
            1.0 - remaining_power
        )

        ratio_recalculation_errors.append(
            abs(
                recalculated_ratio
                - achieved_ratio
            )
        )

        if (
            target_ratio == 0.0
            and not np.all(mask == 1)
        ):
            content_ok = False
            report.append(
                f"[FAIL] Unoccluded sample {index} "
                "contains blocked pixels."
            )

    report.append(
        format_summary(
            "Remaining intensity power",
            np.asarray(powers),
        )
    )

    report.append(
        format_summary(
            "Visible pixel fraction",
            np.asarray(visible_fractions),
        )
    )

    report.append(
        format_summary(
            "Maximum blocked-region intensity",
            np.asarray(blocked_maxima),
        )
    )

    report.append(
        format_summary(
            "Ratio recalculation error",
            np.asarray(
                ratio_recalculation_errors
            ),
        )
    )

    report.append(
        "All sampled intensities finite: "
        + str(bool(np.all(finite_flags)))
    )

    report.append(
        "All sampled masks binary: "
        + str(bool(np.all(mask_binary_flags)))
    )

    report.append(
        "Total negative visible pixels: "
        + str(
            int(
                np.sum(
                    visible_negative_counts
                )
            )
        )
    )

    if not np.all(finite_flags):
        content_ok = False
        report.append(
            "[FAIL] Non-finite intensity values detected."
        )

    if not np.all(mask_binary_flags):
        content_ok = False
        report.append(
            "[FAIL] Non-binary mask values detected."
        )

    if float(np.max(blocked_maxima)) > ZERO_TOLERANCE:
        content_ok = False
        report.append(
            "[FAIL] Blocked-region intensity is not zero."
        )

    if int(np.sum(visible_negative_counts)) > 0:
        content_ok = False
        report.append(
            "[FAIL] Negative intensity detected."
        )

    return content_ok


def validate_nested_masks(
    mask_ds: h5py.Dataset,
    conditions: np.ndarray,
    base_indices: np.ndarray,
    rng: np.random.Generator,
    report: List[str],
) -> bool:
    """Validate monotonic nesting across five occlusion levels."""

    nested_ok = True

    report.append("")
    report.append(
        "SECTION 5 — Nested-mask consistency"
    )

    selected_base_indices = rng.choice(
        EXPECTED_BASE_SAMPLES,
        size=min(
            AUDIT_BASE_GROUP_COUNT,
            EXPECTED_BASE_SAMPLES,
        ),
        replace=False,
    )

    nesting_failures = 0
    radius_failures = 0
    achieved_failures = 0

    radii_all = []
    centers_x = []
    centers_y = []

    for base_index in selected_base_indices:
        group_indices = np.where(
            base_indices == int(base_index)
        )[0]

        if len(group_indices) != EXPECTED_LEVELS:
            nesting_failures += 1
            continue

        order = np.argsort(
            conditions[group_indices, 3]
        )

        group_indices = group_indices[order]

        target = conditions[
            group_indices,
            3,
        ]

        achieved = conditions[
            group_indices,
            4,
        ]

        center_x = conditions[
            group_indices,
            5,
        ]

        center_y = conditions[
            group_indices,
            6,
        ]

        radii = conditions[
            group_indices,
            7,
        ]

        centers_x.append(
            float(center_x[0])
        )
        centers_y.append(
            float(center_y[0])
        )
        radii_all.extend(
            radii.tolist()
        )

        if not np.allclose(
            target,
            TARGET_LEVELS,
            atol=1.0e-12,
        ):
            nesting_failures += 1
            continue

        if not (
            np.allclose(
                center_x,
                center_x[0],
                atol=1.0e-12,
            )
            and np.allclose(
                center_y,
                center_y[0],
                atol=1.0e-12,
            )
        ):
            nesting_failures += 1

        if np.any(
            np.diff(radii) < -1.0e-12
        ):
            radius_failures += 1

        if np.any(
            np.diff(achieved) < -1.0e-12
        ):
            achieved_failures += 1

        previous_blocked = None

        for sample_index in group_indices:
            visible_mask = np.asarray(
                mask_ds[int(sample_index)],
                dtype=np.uint8,
            )

            blocked = (
                visible_mask == 0
            )

            if previous_blocked is not None:
                # Every previously blocked pixel must remain blocked
                # at the next stronger level.
                if not np.all(
                    previous_blocked <= blocked
                ):
                    nesting_failures += 1
                    break

            previous_blocked = blocked

    report.append(
        f"Audited base groups: "
        f"{len(selected_base_indices)}"
    )

    report.append(
        f"Mask nesting failures: "
        f"{nesting_failures}"
    )

    report.append(
        f"Radius monotonicity failures: "
        f"{radius_failures}"
    )

    report.append(
        f"Achieved-ratio monotonicity failures: "
        f"{achieved_failures}"
    )

    report.append(
        format_summary(
            "Obstacle center x",
            np.asarray(centers_x),
        )
    )

    report.append(
        format_summary(
            "Obstacle center y",
            np.asarray(centers_y),
        )
    )

    report.append(
        format_summary(
            "Obstacle radius",
            np.asarray(radii_all),
        )
    )

    unique_center_pairs = len(
        {
            (
                round(x, 6),
                round(y, 6),
            )
            for x, y in zip(
                centers_x,
                centers_y,
            )
        }
    )

    report.append(
        f"Unique sampled center pairs: "
        f"{unique_center_pairs}"
    )

    if (
        nesting_failures > 0
        or radius_failures > 0
        or achieved_failures > 0
    ):
        nested_ok = False
        report.append(
            "[FAIL] Nested occlusion sequence is inconsistent."
        )

    if unique_center_pairs < 10:
        nested_ok = False
        report.append(
            "[FAIL] Obstacle-center diversity is too low."
        )

    return nested_ok


def main() -> None:
    """Run complete v2 dataset audit."""

    print("=" * 78)
    print("CORRECTED OCCLUSION DATASET V2 AUDIT")
    print("=" * 78)
    print("Source:", H5_PATH)

    if not H5_PATH.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {H5_PATH}"
        )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        AUDIT_SEED
    )

    report: List[str] = []

    report.append(
        "Corrected occlusion dataset v2 audit"
    )
    report.append(
        f"Source: {H5_PATH}"
    )
    report.append(
        f"Audit seed: {AUDIT_SEED}"
    )
    report.append("")

    with h5py.File(
        H5_PATH,
        "r",
    ) as h5:
        structure_ok = validate_structure(
            h5,
            report,
        )

        labels = h5["labels"][:]
        conditions = h5["conditions"][:]
        base_indices = h5["base_indices"][:]

        balance_ok = validate_balance(
            labels=labels,
            conditions=conditions,
            base_indices=base_indices,
            report=report,
        )

        error_ok = validate_occlusion_error(
            conditions=conditions,
            report=report,
        )

        content_ok = validate_sample_content(
            intensity_ds=h5["intensity"],
            mask_ds=h5["visible_mask"],
            conditions=conditions,
            rng=rng,
            report=report,
        )

        nested_ok = validate_nested_masks(
            mask_ds=h5["visible_mask"],
            conditions=conditions,
            base_indices=base_indices,
            rng=rng,
            report=report,
        )

    all_ok = (
        structure_ok
        and balance_ok
        and error_ok
        and content_ok
        and nested_ok
    )

    status = (
        "PASS"
        if all_ok
        else "FAIL"
    )

    report.append("")
    report.append("=" * 78)
    report.append(
        f"AUDIT STATUS: {status}"
    )
    report.append("=" * 78)

    report_text = "\n".join(
        report
    )

    REPORT_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    print(report_text)
    print("")
    print("Saved report:")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()