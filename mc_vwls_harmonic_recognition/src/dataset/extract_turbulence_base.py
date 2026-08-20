"""
Extract the unoccluded turbulence fields from the legacy HDF5 dataset.

Source:
    data/generated/turbulence_mask_v1.h5

Output:
    data/generated/turbulence_base_v1.h5

Only samples with mask_ratio == 1.0 are copied.

The source file is opened read-only and is never modified.
"""

from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = (
    ROOT
    / "data"
    / "generated"
    / "turbulence_mask_v1.h5"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "generated"
    / "turbulence_base_v1.h5"
)

TARGET_MASK_RATIO = 1.0
EXPECTED_SAMPLES = 8_960
BATCH_SIZE = 64


def main() -> None:
    print("=" * 78)
    print("EXTRACT UNOCCLUDED TURBULENCE BASE FIELDS")
    print("=" * 78)

    print("Source:", SOURCE_PATH)
    print("Output:", OUTPUT_PATH)

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Source file does not exist: {SOURCE_PATH}"
        )

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_PATH}\n"
            "Do not overwrite it automatically."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with h5py.File(SOURCE_PATH, "r") as source:
        source_fields = source["fields"]
        source_labels = source["labels"]
        source_conditions = source["conditions"]

        conditions_array = source_conditions[:]

        selected_indices = np.where(
            np.isclose(
                conditions_array[:, 2],
                TARGET_MASK_RATIO,
                rtol=0.0,
                atol=1.0e-6,
            )
        )[0]

        print("Selected samples:", len(selected_indices))

        if len(selected_indices) != EXPECTED_SAMPLES:
            raise ValueError(
                f"Expected {EXPECTED_SAMPLES} unoccluded samples, "
                f"but found {len(selected_indices)}."
            )

        with h5py.File(OUTPUT_PATH, "w") as output:
            output_fields = output.create_dataset(
                "fields",
                shape=(
                    EXPECTED_SAMPLES,
                    256,
                    256,
                ),
                dtype=np.complex64,
                chunks=(
                    BATCH_SIZE,
                    256,
                    256,
                ),
                compression="gzip",
                compression_opts=4,
            )

            output_labels = output.create_dataset(
                "labels",
                shape=(EXPECTED_SAMPLES,),
                dtype=np.int32,
            )

            # New base conditions contain only:
            # Cn2, distance, seed
            output_conditions = output.create_dataset(
                "conditions",
                shape=(EXPECTED_SAMPLES, 3),
                dtype=np.float32,
            )

            output_source_indices = output.create_dataset(
                "source_indices",
                shape=(EXPECTED_SAMPLES,),
                dtype=np.int64,
            )

            output.attrs["dataset_name"] = (
                "unoccluded_turbulence_base_v1"
            )
            output.attrs["source_file"] = SOURCE_PATH.name
            output.attrs["source_mask_ratio"] = TARGET_MASK_RATIO
            output.attrs["condition_columns"] = (
                "Cn2,distance,seed"
            )
            output.attrs["field_definition"] = (
                "complex field after turbulence propagation "
                "and before corrected occlusion/noise generation"
            )

            write_index = 0

            with tqdm(
                total=EXPECTED_SAMPLES,
                desc="Extracting",
                unit="field",
            ) as progress:
                for start in range(
                    0,
                    EXPECTED_SAMPLES,
                    BATCH_SIZE,
                ):
                    stop = min(
                        start + BATCH_SIZE,
                        EXPECTED_SAMPLES,
                    )

                    batch_indices = selected_indices[start:stop]

                    batch_fields = source_fields[batch_indices]
                    batch_labels = source_labels[batch_indices]
                    batch_conditions = source_conditions[
                        batch_indices
                    ]

                    batch_size = len(batch_indices)
                    destination_slice = slice(
                        write_index,
                        write_index + batch_size,
                    )

                    output_fields[
                        destination_slice
                    ] = batch_fields

                    output_labels[
                        destination_slice
                    ] = batch_labels

                    # Keep Cn2, distance, and seed.
                    output_conditions[
                        destination_slice,
                        0,
                    ] = batch_conditions[:, 0]

                    output_conditions[
                        destination_slice,
                        1,
                    ] = batch_conditions[:, 1]

                    output_conditions[
                        destination_slice,
                        2,
                    ] = batch_conditions[:, 3]

                    output_source_indices[
                        destination_slice
                    ] = batch_indices

                    write_index += batch_size
                    progress.update(batch_size)

            if write_index != EXPECTED_SAMPLES:
                raise RuntimeError(
                    f"Written samples: {write_index}, "
                    f"expected: {EXPECTED_SAMPLES}."
                )

    print("")
    print("=" * 78)
    print("EXTRACTION COMPLETE")
    print("=" * 78)
    print("Saved:", OUTPUT_PATH)
    print("Samples:", EXPECTED_SAMPLES)
    print(
        "File size:",
        f"{OUTPUT_PATH.stat().st_size / (1024 ** 3):.3f} GB",
    )


if __name__ == "__main__":
    main()