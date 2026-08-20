"""
Inspect every column of the conditions dataset.

Output:
    results/validation/condition_columns_report.txt
"""

from __future__ import annotations

from pathlib import Path

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
    / "condition_columns_report.txt"
)

MAX_UNIQUE_VALUES_TO_PRINT = 30


def format_number(value: float) -> str:
    value = float(value)

    if value == 0.0:
        return "0"

    if abs(value) < 1.0e-3 or abs(value) >= 1.0e4:
        return f"{value:.10e}"

    return f"{value:.10f}".rstrip("0").rstrip(".")


def main() -> None:
    print("=" * 78)
    print("CONDITIONS DATASET INSPECTION")
    print("=" * 78)

    if not H5_PATH.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {H5_PATH}"
        )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_lines = [
        "Conditions dataset inspection report",
        f"Dataset: {H5_PATH}",
        "",
    ]

    with h5py.File(
        H5_PATH,
        "r",
    ) as h5:
        if "conditions" not in h5:
            raise KeyError(
                "conditions dataset was not found."
            )

        conditions_ds = h5["conditions"]

        conditions = np.asarray(
            conditions_ds[:],
            dtype=np.float64,
        )

        print(
            "conditions shape:",
            conditions.shape,
        )

        report_lines.append(
            f"conditions_shape={conditions.shape}"
        )

        report_lines.append(
            f"conditions_dtype={conditions_ds.dtype}"
        )

        report_lines.append("")

        print(
            "conditions dtype:",
            conditions_ds.dtype,
        )

        print("")
        print("HDF5 attributes:")
        print("-" * 78)

        report_lines.append(
            "[HDF5 ATTRIBUTES]"
        )

        if len(h5.attrs) == 0:
            print("No root attributes.")
            report_lines.append(
                "No root attributes."
            )
        else:
            for key in sorted(
                h5.attrs.keys()
            ):
                value = h5.attrs[key]

                print(
                    f"{key} = {value}"
                )

                report_lines.append(
                    f"{key}={value}"
                )

        print("")
        print("conditions dataset attributes:")
        print("-" * 78)

        report_lines.extend(
            [
                "",
                "[CONDITIONS DATASET ATTRIBUTES]",
            ]
        )

        if len(conditions_ds.attrs) == 0:
            print(
                "No conditions dataset attributes."
            )

            report_lines.append(
                "No conditions dataset attributes."
            )
        else:
            for key in sorted(
                conditions_ds.attrs.keys()
            ):
                value = conditions_ds.attrs[
                    key
                ]

                print(
                    f"{key} = {value}"
                )

                report_lines.append(
                    f"{key}={value}"
                )

        if conditions.ndim != 2:
            raise ValueError(
                "conditions must be a two-dimensional array."
            )

        if not np.all(
            np.isfinite(conditions)
        ):
            raise ValueError(
                "conditions contains NaN or Inf."
            )

        print("")
        print("Column statistics:")
        print("=" * 78)

        report_lines.extend(
            [
                "",
                "[COLUMN STATISTICS]",
            ]
        )

        for column_index in range(
            conditions.shape[1]
        ):
            column = conditions[
                :,
                column_index,
            ]

            unique_values, counts = np.unique(
                column,
                return_counts=True,
            )

            print("")
            print(
                f"column_{column_index}"
            )

            print(
                f"  min={format_number(np.min(column))}"
            )

            print(
                f"  max={format_number(np.max(column))}"
            )

            print(
                f"  mean={format_number(np.mean(column))}"
            )

            print(
                f"  unique_count={len(unique_values)}"
            )

            report_lines.extend(
                [
                    "",
                    f"column_{column_index}",
                    (
                        "min="
                        f"{format_number(np.min(column))}"
                    ),
                    (
                        "max="
                        f"{format_number(np.max(column))}"
                    ),
                    (
                        "mean="
                        f"{format_number(np.mean(column))}"
                    ),
                    (
                        "unique_count="
                        f"{len(unique_values)}"
                    ),
                ]
            )

            if (
                len(unique_values)
                <= MAX_UNIQUE_VALUES_TO_PRINT
            ):
                print(
                    "  unique values and counts:"
                )

                report_lines.append(
                    "unique_values_and_counts:"
                )

                for value, count in zip(
                    unique_values,
                    counts,
                ):
                    line = (
                        f"    {format_number(value)}: "
                        f"{int(count)}"
                    )

                    print(line)
                    report_lines.append(line)
            else:
                print(
                    "  first 10 unique values:"
                )

                report_lines.append(
                    "first_10_unique_values:"
                )

                for value in unique_values[:10]:
                    line = (
                        f"    {format_number(value)}"
                    )

                    print(line)
                    report_lines.append(line)

                print(
                    "  last 10 unique values:"
                )

                report_lines.append(
                    "last_10_unique_values:"
                )

                for value in unique_values[-10:]:
                    line = (
                        f"    {format_number(value)}"
                    )

                    print(line)
                    report_lines.append(line)

    report_text = "\n".join(
        report_lines
    )

    REPORT_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    print("")
    print("=" * 78)
    print("CONDITIONS DATASET INSPECTION COMPLETE")
    print("=" * 78)

    print(
        "Report:",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()