"""
Quick integrity check for turbulence_mask_v1.h5

This script does not modify the HDF5 file.
"""

from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

H5_PATH = (
    ROOT
    / "data"
    / "generated"
    / "turbulence_mask_v1.h5"
)

EXPECTED_SAMPLES = 44_800
EXPECTED_IMAGE_SHAPE = (256, 256)
EXPECTED_STATES = 32


def describe_field(field: np.ndarray, sample_index: int) -> None:
    """Print basic statistics for one complex optical field."""

    amplitude = np.abs(field)

    print(f"\n--- Field sample {sample_index} ---")
    print("shape:", field.shape)
    print("dtype:", field.dtype)
    print("finite:", bool(np.all(np.isfinite(field))))
    print("real min/max:", float(np.min(field.real)), float(np.max(field.real)))
    print("imag min/max:", float(np.min(field.imag)), float(np.max(field.imag)))
    print("amplitude min/max:", float(np.min(amplitude)), float(np.max(amplitude)))
    print("amplitude mean:", float(np.mean(amplitude)))
    print("power sum:", float(np.sum(amplitude ** 2)))
    print("nonzero ratio:", float(np.count_nonzero(amplitude) / amplitude.size))


def main() -> None:
    print("=" * 70)
    print("HDF5 QUICK INTEGRITY CHECK")
    print("=" * 70)

    print("File:", H5_PATH)
    print("Exists:", H5_PATH.exists())

    if not H5_PATH.exists():
        raise FileNotFoundError(f"HDF5 file does not exist: {H5_PATH}")

    file_size_gb = H5_PATH.stat().st_size / (1024 ** 3)
    print(f"File size: {file_size_gb:.3f} GB")

    # "r" means read-only. This script cannot modify the HDF5 file.
    with h5py.File(H5_PATH, "r") as h5:
        print("\nHDF5 keys:", list(h5.keys()))

        required_keys = {"fields", "labels", "conditions"}
        missing_keys = required_keys.difference(h5.keys())

        if missing_keys:
            raise KeyError(f"Missing datasets: {sorted(missing_keys)}")

        fields = h5["fields"]
        labels_ds = h5["labels"]
        conditions_ds = h5["conditions"]

        print("\n--- Dataset structure ---")
        print("fields shape:", fields.shape)
        print("fields dtype:", fields.dtype)
        print("fields chunks:", fields.chunks)
        print("fields compression:", fields.compression)
        print("fields compression options:", fields.compression_opts)

        print("labels shape:", labels_ds.shape)
        print("labels dtype:", labels_ds.dtype)

        print("conditions shape:", conditions_ds.shape)
        print("conditions dtype:", conditions_ds.dtype)

        structure_ok = True

        if fields.shape != (
            EXPECTED_SAMPLES,
            *EXPECTED_IMAGE_SHAPE,
        ):
            structure_ok = False
            print(
                "\n[ERROR] Unexpected fields shape:",
                fields.shape,
            )

        if labels_ds.shape != (EXPECTED_SAMPLES,):
            structure_ok = False
            print(
                "\n[ERROR] Unexpected labels shape:",
                labels_ds.shape,
            )

        if conditions_ds.shape != (EXPECTED_SAMPLES, 4):
            structure_ok = False
            print(
                "\n[ERROR] Unexpected conditions shape:",
                conditions_ds.shape,
            )

        # labels and conditions are small enough to read fully.
        labels = labels_ds[:]
        conditions = conditions_ds[:]

        print("\n--- Label check ---")
        unique_labels, label_counts = np.unique(
            labels,
            return_counts=True,
        )

        print("unique labels:", unique_labels.tolist())
        print("number of unique labels:", len(unique_labels))
        print("label min/max:", int(labels.min()), int(labels.max()))

        print("\nlabel counts:")
        for label, count in zip(unique_labels, label_counts):
            print(f"  state {int(label):02d}: {int(count)}")

        expected_per_state = EXPECTED_SAMPLES // EXPECTED_STATES
        print("expected samples per state:", expected_per_state)

        label_ok = (
            len(unique_labels) == EXPECTED_STATES
            and int(labels.min()) == 0
            and int(labels.max()) == EXPECTED_STATES - 1
            and np.all(label_counts == expected_per_state)
        )

        if not label_ok:
            print("[ERROR] Label distribution is not balanced or complete.")

        print("\n--- Condition check ---")

        condition_names = [
            "Cn2",
            "distance",
            "mask_ratio",
            "seed",
        ]

        expected_unique_counts = {
            "Cn2": 7,
            "distance": 4,
            "mask_ratio": 5,
            "seed": 10,
        }

        condition_ok = True

        for column_index, name in enumerate(condition_names):
            values, counts = np.unique(
                conditions[:, column_index],
                return_counts=True,
            )

            print(f"\n{name}:")
            print("  unique count:", len(values))
            print("  values:", values.tolist())
            print("  min count:", int(counts.min()))
            print("  max count:", int(counts.max()))

            if len(values) != expected_unique_counts[name]:
                condition_ok = False
                print(
                    f"  [ERROR] Expected "
                    f"{expected_unique_counts[name]} unique values."
                )

        print("\nFirst condition:", conditions[0].tolist())
        print("Middle condition:", conditions[len(conditions) // 2].tolist())
        print("Last condition:", conditions[-1].tolist())

        # Read representative fields without loading the entire 11.2 GB file.
        sample_indices = [
            0,
            1,
            EXPECTED_SAMPLES // 2,
            EXPECTED_SAMPLES - 2,
            EXPECTED_SAMPLES - 1,
        ]

        field_ok = True

        for index in sample_indices:
            field = fields[index]

            describe_field(field, index)

            if not np.all(np.isfinite(field)):
                field_ok = False
                print("[ERROR] NaN or Inf detected.")

            if np.sum(np.abs(field) ** 2) <= 0:
                field_ok = False
                print("[ERROR] Field power is zero.")

        print("\n" + "=" * 70)
        print("QUICK CHECK RESULT")
        print("=" * 70)
        print("Structure:", "PASS" if structure_ok else "FAIL")
        print("Labels:", "PASS" if label_ok else "FAIL")
        print("Conditions:", "PASS" if condition_ok else "FAIL")
        print("Sample fields:", "PASS" if field_ok else "FAIL")

        all_ok = (
            structure_ok
            and label_ok
            and condition_ok
            and field_ok
        )

        print("\nOVERALL:", "PASS" if all_ok else "FAIL")

        if all_ok:
            print(
                "\nThe HDF5 file passed the quick integrity check."
            )
        else:
            print(
                "\nThe file requires further diagnosis. "
                "Do not regenerate it yet."
            )


if __name__ == "__main__":
    main()