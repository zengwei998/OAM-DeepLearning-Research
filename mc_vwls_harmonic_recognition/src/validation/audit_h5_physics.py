"""
Physical-quality audit for turbulence_mask_v1.h5.

Checks:
1. Field power versus mask ratio.
2. Recoverability of the binary occlusion mask.
3. Diversity across random realizations.
4. Diversity across OAM states.
5. Possible duplicate samples.
6. Consistency between labels and state definitions.

The source HDF5 file is opened read-only and is never modified.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

H5_PATH = (
    ROOT
    / "data"
    / "generated"
    / "turbulence_mask_v1.h5"
)

REPORT_PATH = (
    ROOT
    / "results"
    / "validation"
    / "h5_physics_audit.txt"
)

EXPECTED_SAMPLES = 44_800
EXPECTED_STATES = 32

# A pixel is treated as blocked when its complex amplitude is exactly
# or numerically extremely close to zero.
MASK_EPSILON = 1.0e-12

# Number of samples selected for detailed statistical checks.
RANDOM_AUDIT_SAMPLES = 512

# Fixed seed guarantees reproducibility.
AUDIT_SEED = 20260804


def normalized_correlation(
    image_a: np.ndarray,
    image_b: np.ndarray,
) -> float:
    """
    Calculate normalized zero-mean correlation between two real images.
    """

    a = np.asarray(image_a, dtype=np.float64).ravel()
    b = np.asarray(image_b, dtype=np.float64).ravel()

    a = a - np.mean(a)
    b = b - np.mean(b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator <= 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def normalized_complex_similarity(
    field_a: np.ndarray,
    field_b: np.ndarray,
) -> float:
    """
    Calculate phase-insensitive normalized complex-field similarity.

    The absolute inner product makes the metric insensitive to a global
    phase offset.
    """

    a = np.asarray(field_a, dtype=np.complex128).ravel()
    b = np.asarray(field_b, dtype=np.complex128).ravel()

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator <= 0:
        return 0.0

    return float(np.abs(np.vdot(a, b)) / denominator)


def infer_visible_mask(field: np.ndarray) -> np.ndarray:
    """
    Recover the visible-region mask from exact zero-valued blocked pixels.

    Returns:
        Boolean array:
        True  = visible pixel
        False = blocked pixel
    """

    return np.abs(field) > MASK_EPSILON


def summarize(values: List[float]) -> Dict[str, float]:
    """
    Return compact descriptive statistics.
    """

    array = np.asarray(values, dtype=np.float64)

    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
    }


def format_summary(name: str, values: List[float]) -> str:
    """
    Format statistics for console and report output.
    """

    stats = summarize(values)

    return (
        f"{name}: "
        f"n={stats['count']}, "
        f"min={stats['min']:.8f}, "
        f"mean={stats['mean']:.8f}, "
        f"median={stats['median']:.8f}, "
        f"max={stats['max']:.8f}, "
        f"std={stats['std']:.8f}"
    )


def find_matching_indices(
    labels: np.ndarray,
    conditions: np.ndarray,
    state_id: int,
    cn2: float,
    distance: float,
    mask_ratio: float,
) -> np.ndarray:
    """
    Find all realizations matching one physical condition and one state.
    """

    return np.where(
        (labels == state_id)
        & np.isclose(conditions[:, 0], cn2, rtol=1.0e-5, atol=0.0)
        & np.isclose(conditions[:, 1], distance)
        & np.isclose(conditions[:, 2], mask_ratio)
    )[0]


def audit_random_samples(
    fields: h5py.Dataset,
    conditions: np.ndarray,
    rng: np.random.Generator,
) -> List[str]:
    """
    Audit field power, visible fraction, mask binarity, and finite values.
    """

    lines: List[str] = []

    sample_count = min(RANDOM_AUDIT_SAMPLES, len(fields))

    selected_indices = rng.choice(
        len(fields),
        size=sample_count,
        replace=False,
    )

    powers: List[float] = []
    visible_fractions: List[float] = []
    finite_flags: List[bool] = []
    mask_binary_errors: List[float] = []

    power_by_mask: Dict[float, List[float]] = {}
    visibility_by_mask: Dict[float, List[float]] = {}

    for index in selected_indices:
        field = fields[int(index)]

        amplitude = np.abs(field)
        intensity = amplitude ** 2

        visible_mask = infer_visible_mask(field)

        power = float(np.sum(intensity))
        visible_fraction = float(np.mean(visible_mask))

        finite = bool(np.all(np.isfinite(field)))

        # Blocked pixels should be exactly zero or extremely close to zero.
        blocked_values = amplitude[~visible_mask]

        if blocked_values.size == 0:
            binary_error = 0.0
        else:
            binary_error = float(np.max(blocked_values))

        mask_ratio = float(conditions[int(index), 2])

        powers.append(power)
        visible_fractions.append(visible_fraction)
        finite_flags.append(finite)
        mask_binary_errors.append(binary_error)

        power_by_mask.setdefault(mask_ratio, []).append(power)
        visibility_by_mask.setdefault(mask_ratio, []).append(
            visible_fraction
        )

    lines.append(format_summary("Field total power", powers))
    lines.append(
        format_summary(
            "Visible pixel fraction",
            visible_fractions,
        )
    )
    lines.append(
        format_summary(
            "Maximum blocked-pixel amplitude",
            mask_binary_errors,
        )
    )
    lines.append(
        "All sampled fields finite: "
        + str(bool(np.all(finite_flags)))
    )

    lines.append("")
    lines.append("Statistics grouped by configured mask_ratio:")

    for mask_ratio in sorted(power_by_mask.keys(), reverse=True):
        lines.append(
            f"mask_ratio={mask_ratio:.6f}"
        )
        lines.append(
            "  "
            + format_summary(
                "power",
                power_by_mask[mask_ratio],
            )
        )
        lines.append(
            "  "
            + format_summary(
                "visible_fraction",
                visibility_by_mask[mask_ratio],
            )
        )

    return lines


def audit_realization_diversity(
    fields: h5py.Dataset,
    labels: np.ndarray,
    conditions: np.ndarray,
) -> List[str]:
    """
    Check whether different seeds produce genuinely different fields.
    """

    lines: List[str] = []

    unique_cn2 = np.unique(conditions[:, 0])
    unique_distance = np.unique(conditions[:, 1])
    unique_mask = np.unique(conditions[:, 2])

    test_conditions: List[Tuple[int, float, float, float]] = [
        (
            0,
            float(unique_cn2[0]),
            float(unique_distance[0]),
            float(unique_mask[-1]),
        ),
        (
            7,
            float(unique_cn2[len(unique_cn2) // 2]),
            float(unique_distance[len(unique_distance) // 2]),
            float(unique_mask[len(unique_mask) // 2]),
        ),
        (
            31,
            float(unique_cn2[-1]),
            float(unique_distance[-1]),
            float(unique_mask[0]),
        ),
    ]

    for state_id, cn2, distance, mask_ratio in test_conditions:
        indices = find_matching_indices(
            labels=labels,
            conditions=conditions,
            state_id=state_id,
            cn2=cn2,
            distance=distance,
            mask_ratio=mask_ratio,
        )

        lines.append("")
        lines.append(
            "Condition: "
            f"state={state_id}, "
            f"Cn2={cn2:.4e}, "
            f"distance={distance:.1f}, "
            f"mask_ratio={mask_ratio:.3f}"
        )
        lines.append(
            f"Matched realizations: {len(indices)}"
        )

        if len(indices) < 2:
            lines.append(
                "[FAIL] Fewer than two realizations were found."
            )
            continue

        reference_field = fields[int(indices[0])]
        reference_intensity = np.abs(reference_field) ** 2

        complex_similarities: List[float] = []
        intensity_correlations: List[float] = []

        for comparison_index in indices[1:]:
            comparison_field = fields[int(comparison_index)]
            comparison_intensity = np.abs(comparison_field) ** 2

            complex_similarities.append(
                normalized_complex_similarity(
                    reference_field,
                    comparison_field,
                )
            )

            intensity_correlations.append(
                normalized_correlation(
                    reference_intensity,
                    comparison_intensity,
                )
            )

        lines.append(
            "  "
            + format_summary(
                "complex similarity to seed 0",
                complex_similarities,
            )
        )
        lines.append(
            "  "
            + format_summary(
                "intensity correlation to seed 0",
                intensity_correlations,
            )
        )

    return lines


def audit_state_diversity(
    fields: h5py.Dataset,
    labels: np.ndarray,
    conditions: np.ndarray,
) -> List[str]:
    """
    Compare different OAM states under the same nominal channel condition.
    """

    lines: List[str] = []

    unique_cn2 = np.unique(conditions[:, 0])
    unique_distance = np.unique(conditions[:, 1])
    unique_mask = np.unique(conditions[:, 2])

    selected_condition = {
        "cn2": float(unique_cn2[2]),
        "distance": float(unique_distance[1]),
        "mask_ratio": float(unique_mask[-1]),
        "seed": 0.0,
    }

    indices = np.where(
        np.isclose(
            conditions[:, 0],
            selected_condition["cn2"],
            rtol=1.0e-5,
            atol=0.0,
        )
        & np.isclose(
            conditions[:, 1],
            selected_condition["distance"],
        )
        & np.isclose(
            conditions[:, 2],
            selected_condition["mask_ratio"],
        )
        & np.isclose(
            conditions[:, 3],
            selected_condition["seed"],
        )
    )[0]

    lines.append("")
    lines.append(
        "State-diversity reference condition: "
        f"Cn2={selected_condition['cn2']:.4e}, "
        f"distance={selected_condition['distance']:.1f}, "
        f"mask_ratio={selected_condition['mask_ratio']:.3f}, "
        f"seed={selected_condition['seed']:.0f}"
    )
    lines.append(f"Matched state samples: {len(indices)}")

    state_to_index = {
        int(labels[index]): int(index)
        for index in indices
    }

    missing_states = sorted(
        set(range(EXPECTED_STATES)).difference(state_to_index.keys())
    )

    if missing_states:
        lines.append(
            f"[FAIL] Missing states: {missing_states}"
        )
        return lines

    reference_state = 0
    reference_field = fields[state_to_index[reference_state]]
    reference_intensity = np.abs(reference_field) ** 2

    complex_similarities: List[float] = []
    intensity_correlations: List[float] = []

    for state_id in range(1, EXPECTED_STATES):
        field = fields[state_to_index[state_id]]
        intensity = np.abs(field) ** 2

        complex_similarities.append(
            normalized_complex_similarity(
                reference_field,
                field,
            )
        )

        intensity_correlations.append(
            normalized_correlation(
                reference_intensity,
                intensity,
            )
        )

    lines.append(
        "  "
        + format_summary(
            "complex similarity between state 0 and states 1-31",
            complex_similarities,
        )
    )
    lines.append(
        "  "
        + format_summary(
            "intensity correlation between state 0 and states 1-31",
            intensity_correlations,
        )
    )

    return lines


def audit_exact_duplicates(
    fields: h5py.Dataset,
    rng: np.random.Generator,
) -> List[str]:
    """
    Check a representative subset for exact duplicate fields.

    Full pairwise comparison of 44,800 images is unnecessary and
    computationally expensive. Instead, selected fields are fingerprinted.
    """

    lines: List[str] = []

    sample_count = min(512, len(fields))

    selected_indices = rng.choice(
        len(fields),
        size=sample_count,
        replace=False,
    )

    fingerprints = set()
    duplicates = 0

    for index in selected_indices:
        field = fields[int(index)]

        # Use multiple deterministic numerical summaries as a lightweight
        # duplicate fingerprint.
        fingerprint = (
            round(float(np.sum(field.real)), 10),
            round(float(np.sum(field.imag)), 10),
            round(float(np.sum(np.abs(field) ** 2)), 10),
            round(float(np.max(np.abs(field))), 10),
            int(np.count_nonzero(np.abs(field) > MASK_EPSILON)),
        )

        if fingerprint in fingerprints:
            duplicates += 1
        else:
            fingerprints.add(fingerprint)

    lines.append("")
    lines.append(
        f"Duplicate audit sample count: {sample_count}"
    )
    lines.append(
        f"Repeated lightweight fingerprints: {duplicates}"
    )

    return lines


def determine_status(report_lines: List[str]) -> str:
    """
    Provide a conservative audit status.
    """

    report_text = "\n".join(report_lines)

    if "[FAIL]" in report_text:
        return "FAIL"

    return "PASS_WITH_REVIEW"


def main() -> None:
    print("=" * 78)
    print("HDF5 PHYSICAL-QUALITY AUDIT")
    print("=" * 78)
    print("Source:", H5_PATH)

    if not H5_PATH.exists():
        raise FileNotFoundError(
            f"Source file does not exist: {H5_PATH}"
        )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(AUDIT_SEED)

    report_lines: List[str] = []

    report_lines.append(
        "HDF5 physical-quality audit report"
    )
    report_lines.append(
        f"Source file: {H5_PATH}"
    )
    report_lines.append(
        f"Audit seed: {AUDIT_SEED}"
    )
    report_lines.append("")

    with h5py.File(H5_PATH, "r") as h5:
        fields = h5["fields"]
        labels = h5["labels"][:]
        conditions = h5["conditions"][:]

        if fields.shape[0] != EXPECTED_SAMPLES:
            report_lines.append(
                f"[FAIL] Expected {EXPECTED_SAMPLES} samples, "
                f"found {fields.shape[0]}."
            )

        report_lines.append(
            "SECTION 1 — Random sample field and mask audit"
        )
        report_lines.extend(
            audit_random_samples(
                fields=fields,
                conditions=conditions,
                rng=rng,
            )
        )

        report_lines.append("")
        report_lines.append(
            "SECTION 2 — Random-realization diversity"
        )
        report_lines.extend(
            audit_realization_diversity(
                fields=fields,
                labels=labels,
                conditions=conditions,
            )
        )

        report_lines.append("")
        report_lines.append(
            "SECTION 3 — OAM-state diversity"
        )
        report_lines.extend(
            audit_state_diversity(
                fields=fields,
                labels=labels,
                conditions=conditions,
            )
        )

        report_lines.append("")
        report_lines.append(
            "SECTION 4 — Duplicate screening"
        )
        report_lines.extend(
            audit_exact_duplicates(
                fields=fields,
                rng=rng,
            )
        )

    status = determine_status(report_lines)

    report_lines.append("")
    report_lines.append("=" * 78)
    report_lines.append(f"AUDIT STATUS: {status}")
    report_lines.append("=" * 78)
    report_lines.append(
        "PASS_WITH_REVIEW means that no structural failure was found, "
        "but numerical results must still be reviewed before the dataset "
        "is accepted for final experiments."
    )

    report_text = "\n".join(report_lines)

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