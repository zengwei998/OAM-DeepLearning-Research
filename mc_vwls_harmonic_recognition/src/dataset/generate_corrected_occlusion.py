"""
Generate corrected random circular local-occlusion observations.

Input:
    data/generated/turbulence_base_v1.h5

Output:
    data/generated/occlusion_clean_v2.h5

Physical definition:
    - The circular region is blocked.
    - Pixels outside the circle remain visible.
    - Occlusion severity is controlled by blocked optical-energy ratio,
      not by geometric disk area.
    - Five nested occlusion levels share the same random center for
      each underlying turbulence realization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = (
    ROOT
    / "data"
    / "generated"
    / "turbulence_base_v1.h5"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "generated"
    / "occlusion_clean_v2.h5"
)

GRID_SIZE = 256

TARGET_OCCLUSION_RATIOS = np.asarray(
    [0.0, 0.1, 0.2, 0.3, 0.4],
    dtype=np.float64,
)

EXPECTED_BASE_SAMPLES = 8_960
EXPECTED_OUTPUT_SAMPLES = 44_800

WRITE_BATCH_SIZE = 32

GLOBAL_MASK_SEED = 20260804

# Random obstacle-center offset relative to the intensity RMS radius.
MIN_OFFSET_FACTOR = 0.15
MAX_OFFSET_FACTOR = 0.65

POWER_EPSILON = 1.0e-20


def intensity_centroid_and_rms_radius(
    intensity: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Calculate intensity-weighted centroid and RMS beam radius.

    Returns:
        center_x, center_y, rms_radius
    """

    height, width = intensity.shape

    yy, xx = np.indices(
        (height, width),
        dtype=np.float64,
    )

    total_power = float(np.sum(intensity, dtype=np.float64))

    if total_power <= POWER_EPSILON:
        raise ValueError("Input intensity has zero total power.")

    center_x = float(
        np.sum(xx * intensity, dtype=np.float64)
        / total_power
    )

    center_y = float(
        np.sum(yy * intensity, dtype=np.float64)
        / total_power
    )

    radius_squared = (
        (xx - center_x) ** 2
        + (yy - center_y) ** 2
    )

    rms_radius = float(
        np.sqrt(
            np.sum(
                radius_squared * intensity,
                dtype=np.float64,
            )
            / total_power
        )
    )

    return center_x, center_y, rms_radius


def geometry_seed_from_condition(
    cn2: float,
    distance: float,
    propagation_seed: int,
) -> int:
    """
    Generate a deterministic mask seed from channel variables.

    The OAM label is intentionally excluded. Therefore, all 32 states
    under the same turbulence condition use the same obstacle center.
    """

    cn2_code = int(
        round(
            -np.log10(float(cn2))
            * 1_000_000
        )
    )

    distance_code = int(round(float(distance) * 10))

    seed_sequence = np.random.SeedSequence(
        [
            GLOBAL_MASK_SEED,
            cn2_code,
            distance_code,
            int(propagation_seed),
        ]
    )

    return int(
        seed_sequence.generate_state(
            1,
            dtype=np.uint32,
        )[0]
    )


def generate_random_obstacle_center(
    intensity: np.ndarray,
    mask_seed: int,
) -> Tuple[float, float]:
    """
    Generate one reproducible off-axis obstacle center.

    The center is sampled around the beam intensity centroid. Its
    displacement is scaled by the beam RMS radius.
    """

    beam_x, beam_y, rms_radius = (
        intensity_centroid_and_rms_radius(intensity)
    )

    rng = np.random.default_rng(mask_seed)

    angle = float(
        rng.uniform(
            0.0,
            2.0 * np.pi,
        )
    )

    offset_factor = float(
        rng.uniform(
            MIN_OFFSET_FACTOR,
            MAX_OFFSET_FACTOR,
        )
    )

    offset = offset_factor * rms_radius

    obstacle_x = beam_x + offset * np.cos(angle)
    obstacle_y = beam_y + offset * np.sin(angle)

    obstacle_x = float(
        np.clip(
            obstacle_x,
            0.0,
            GRID_SIZE - 1.0,
        )
    )

    obstacle_y = float(
        np.clip(
            obstacle_y,
            0.0,
            GRID_SIZE - 1.0,
        )
    )

    return obstacle_x, obstacle_y


def generate_energy_controlled_mask(
    intensity: np.ndarray,
    obstacle_x: float,
    obstacle_y: float,
    target_ratio: float,
) -> Tuple[np.ndarray, float, float]:
    """
    Generate a binary circular obstacle with a prescribed blocked-energy
    fraction.

    Returns:
        visible_mask:
            uint8 array; 1 means visible, 0 means blocked.

        achieved_ratio:
            Actual blocked-energy ratio after pixel discretization.

        radius_pixels:
            Selected obstacle radius in pixels.
    """

    if target_ratio <= 0.0:
        visible_mask = np.ones(
            intensity.shape,
            dtype=np.uint8,
        )

        return visible_mask, 0.0, 0.0

    if not 0.0 < target_ratio < 1.0:
        raise ValueError(
            f"Invalid target occlusion ratio: {target_ratio}"
        )

    yy, xx = np.indices(
        intensity.shape,
        dtype=np.float64,
    )

    distance_squared = (
        (xx - obstacle_x) ** 2
        + (yy - obstacle_y) ** 2
    )

    flat_distance_squared = distance_squared.ravel()
    flat_intensity = np.asarray(
        intensity,
        dtype=np.float64,
    ).ravel()

    total_power = float(
        np.sum(
            flat_intensity,
            dtype=np.float64,
        )
    )

    if total_power <= POWER_EPSILON:
        raise ValueError("Input intensity has zero total power.")

    order = np.argsort(
        flat_distance_squared,
        kind="stable",
    )

    sorted_energy = flat_intensity[order]

    cumulative_energy = np.cumsum(
        sorted_energy,
        dtype=np.float64,
    )

    target_energy = target_ratio * total_power

    cutoff_position = int(
        np.searchsorted(
            cumulative_energy,
            target_energy,
            side="left",
        )
    )

    cutoff_position = min(
        cutoff_position,
        len(order) - 1,
    )

    radius_squared = float(
        flat_distance_squared[
            order[cutoff_position]
        ]
    )

    blocked_mask = (
        distance_squared <= radius_squared
    )

    blocked_energy = float(
        np.sum(
            intensity[blocked_mask],
            dtype=np.float64,
        )
    )

    achieved_ratio = blocked_energy / total_power

    visible_mask = (
        ~blocked_mask
    ).astype(np.uint8)

    radius_pixels = float(
        np.sqrt(radius_squared)
    )

    return (
        visible_mask,
        achieved_ratio,
        radius_pixels,
    )


def check_source_file(
    source: h5py.File,
) -> None:
    """
    Validate the turbulence-base input file.
    """

    required_keys = {
        "fields",
        "labels",
        "conditions",
        "source_indices",
    }

    missing_keys = required_keys.difference(
        source.keys()
    )

    if missing_keys:
        raise KeyError(
            f"Missing source datasets: {sorted(missing_keys)}"
        )

    fields = source["fields"]
    labels = source["labels"]
    conditions = source["conditions"]

    if fields.shape != (
        EXPECTED_BASE_SAMPLES,
        GRID_SIZE,
        GRID_SIZE,
    ):
        raise ValueError(
            f"Unexpected fields shape: {fields.shape}"
        )

    if labels.shape != (
        EXPECTED_BASE_SAMPLES,
    ):
        raise ValueError(
            f"Unexpected labels shape: {labels.shape}"
        )

    if conditions.shape != (
        EXPECTED_BASE_SAMPLES,
        3,
    ):
        raise ValueError(
            f"Unexpected conditions shape: {conditions.shape}"
        )


def create_output_datasets(
    output: h5py.File,
) -> dict:
    """
    Create all HDF5 output datasets.
    """

    datasets = {}

    datasets["intensity"] = output.create_dataset(
        "intensity",
        shape=(
            EXPECTED_OUTPUT_SAMPLES,
            GRID_SIZE,
            GRID_SIZE,
        ),
        dtype=np.float32,
        chunks=(
            WRITE_BATCH_SIZE,
            GRID_SIZE,
            GRID_SIZE,
        ),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )

    datasets["visible_mask"] = output.create_dataset(
        "visible_mask",
        shape=(
            EXPECTED_OUTPUT_SAMPLES,
            GRID_SIZE,
            GRID_SIZE,
        ),
        dtype=np.uint8,
        chunks=(
            WRITE_BATCH_SIZE,
            GRID_SIZE,
            GRID_SIZE,
        ),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )

    datasets["labels"] = output.create_dataset(
        "labels",
        shape=(EXPECTED_OUTPUT_SAMPLES,),
        dtype=np.int32,
    )

    # Columns:
    # 0 Cn2
    # 1 propagation distance
    # 2 propagation seed
    # 3 target blocked-energy ratio
    # 4 achieved blocked-energy ratio
    # 5 obstacle center x
    # 6 obstacle center y
    # 7 obstacle radius in pixels
    # 8 mask seed
    datasets["conditions"] = output.create_dataset(
        "conditions",
        shape=(
            EXPECTED_OUTPUT_SAMPLES,
            9,
        ),
        dtype=np.float64,
    )

    datasets["base_indices"] = output.create_dataset(
        "base_indices",
        shape=(EXPECTED_OUTPUT_SAMPLES,),
        dtype=np.int64,
    )

    datasets["legacy_source_indices"] = output.create_dataset(
        "legacy_source_indices",
        shape=(EXPECTED_OUTPUT_SAMPLES,),
        dtype=np.int64,
    )

    output.attrs["dataset_name"] = (
        "corrected_random_local_occlusion_v2"
    )

    output.attrs["source_file"] = (
        SOURCE_PATH.name
    )

    output.attrs["sample_count"] = (
        EXPECTED_OUTPUT_SAMPLES
    )

    output.attrs["condition_columns"] = (
        "Cn2,distance,propagation_seed,"
        "target_energy_occlusion,"
        "achieved_energy_occlusion,"
        "obstacle_center_x,"
        "obstacle_center_y,"
        "obstacle_radius_pixels,"
        "mask_seed"
    )

    output.attrs["mask_definition"] = (
        "1=visible,0=blocked"
    )

    output.attrs["occlusion_definition"] = (
        "blocked optical energy divided by total "
        "pre-occlusion optical energy"
    )

    output.attrs["target_occlusion_ratios"] = (
        TARGET_OCCLUSION_RATIOS
    )

    output.attrs["global_mask_seed"] = (
        GLOBAL_MASK_SEED
    )

    output.attrs["intensity_definition"] = (
        "clean post-turbulence intensity after "
        "corrected binary circular local occlusion"
    )

    return datasets


def main() -> None:
    print("=" * 78)
    print("GENERATE CORRECTED RANDOM LOCAL OCCLUSION DATASET")
    print("=" * 78)

    print("Source:", SOURCE_PATH)
    print("Output:", OUTPUT_PATH)
    print(
        "Target energy-occlusion ratios:",
        TARGET_OCCLUSION_RATIOS.tolist(),
    )

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Source file does not exist: {SOURCE_PATH}"
        )

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Output file already exists: {OUTPUT_PATH}\n"
            "Do not overwrite it automatically."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    achieved_errors = []
    output_index = 0

    with h5py.File(
        SOURCE_PATH,
        "r",
    ) as source:
        check_source_file(source)

        source_fields = source["fields"]
        source_labels = source["labels"]
        source_conditions = source["conditions"]
        source_legacy_indices = source["source_indices"]

        with h5py.File(
            OUTPUT_PATH,
            "w",
        ) as output:
            datasets = create_output_datasets(output)

            intensity_buffer = []
            mask_buffer = []
            label_buffer = []
            condition_buffer = []
            base_index_buffer = []
            legacy_index_buffer = []

            def flush_buffers() -> None:
                nonlocal output_index

                number_in_buffer = len(
                    intensity_buffer
                )

                if number_in_buffer == 0:
                    return

                destination = slice(
                    output_index,
                    output_index + number_in_buffer,
                )

                datasets["intensity"][
                    destination
                ] = np.asarray(
                    intensity_buffer,
                    dtype=np.float32,
                )

                datasets["visible_mask"][
                    destination
                ] = np.asarray(
                    mask_buffer,
                    dtype=np.uint8,
                )

                datasets["labels"][
                    destination
                ] = np.asarray(
                    label_buffer,
                    dtype=np.int32,
                )

                datasets["conditions"][
                    destination
                ] = np.asarray(
                    condition_buffer,
                    dtype=np.float64,
                )

                datasets["base_indices"][
                    destination
                ] = np.asarray(
                    base_index_buffer,
                    dtype=np.int64,
                )

                datasets["legacy_source_indices"][
                    destination
                ] = np.asarray(
                    legacy_index_buffer,
                    dtype=np.int64,
                )

                output_index += number_in_buffer

                intensity_buffer.clear()
                mask_buffer.clear()
                label_buffer.clear()
                condition_buffer.clear()
                base_index_buffer.clear()
                legacy_index_buffer.clear()

            with tqdm(
                total=EXPECTED_OUTPUT_SAMPLES,
                desc="Generating",
                unit="image",
            ) as progress:
                for base_index in range(
                    EXPECTED_BASE_SAMPLES
                ):
                    field = source_fields[base_index]

                    label = int(
                        source_labels[base_index]
                    )

                    base_condition = (
                        source_conditions[base_index]
                    )

                    cn2 = float(
                        base_condition[0]
                    )

                    distance = float(
                        base_condition[1]
                    )

                    propagation_seed = int(
                        round(
                            float(base_condition[2])
                        )
                    )

                    legacy_source_index = int(
                        source_legacy_indices[base_index]
                    )

                    intensity_before = np.asarray(
                        np.abs(field) ** 2,
                        dtype=np.float64,
                    )

                    total_power = float(
                        np.sum(
                            intensity_before,
                            dtype=np.float64,
                        )
                    )

                    if (
                        not np.isfinite(total_power)
                        or total_power <= POWER_EPSILON
                    ):
                        raise ValueError(
                            f"Invalid field power at "
                            f"base index {base_index}: "
                            f"{total_power}"
                        )

                    mask_seed = (
                        geometry_seed_from_condition(
                            cn2=cn2,
                            distance=distance,
                            propagation_seed=propagation_seed,
                        )
                    )

                    obstacle_x, obstacle_y = (
                        generate_random_obstacle_center(
                            intensity=intensity_before,
                            mask_seed=mask_seed,
                        )
                    )

                    previous_radius = -1.0

                    for target_ratio in (
                        TARGET_OCCLUSION_RATIOS
                    ):
                        (
                            visible_mask,
                            achieved_ratio,
                            radius_pixels,
                        ) = generate_energy_controlled_mask(
                            intensity=intensity_before,
                            obstacle_x=obstacle_x,
                            obstacle_y=obstacle_y,
                            target_ratio=float(target_ratio),
                        )

                        if (
                            radius_pixels
                            + 1.0e-12
                            < previous_radius
                        ):
                            raise RuntimeError(
                                "Nested-mask radius decreased at "
                                f"base index {base_index}."
                            )

                        previous_radius = radius_pixels

                        clean_intensity = (
                            intensity_before
                            * visible_mask
                        )

                        achieved_check = (
                            1.0
                            - float(
                                np.sum(
                                    clean_intensity,
                                    dtype=np.float64,
                                )
                            )
                            / total_power
                        )

                        ratio_error = abs(
                            achieved_check
                            - float(target_ratio)
                        )

                        achieved_errors.append(
                            ratio_error
                        )

                        intensity_buffer.append(
                            clean_intensity.astype(
                                np.float32
                            )
                        )

                        mask_buffer.append(
                            visible_mask
                        )

                        label_buffer.append(
                            label
                        )

                        condition_buffer.append(
                            [
                                cn2,
                                distance,
                                propagation_seed,
                                float(target_ratio),
                                float(achieved_check),
                                obstacle_x,
                                obstacle_y,
                                radius_pixels,
                                mask_seed,
                            ]
                        )

                        base_index_buffer.append(
                            base_index
                        )

                        legacy_index_buffer.append(
                            legacy_source_index
                        )

                        if (
                            len(intensity_buffer)
                            >= WRITE_BATCH_SIZE
                        ):
                            flush_buffers()

                        progress.update(1)

            flush_buffers()

            output.attrs[
                "maximum_absolute_occlusion_error"
            ] = float(
                np.max(achieved_errors)
            )

            output.attrs[
                "mean_absolute_occlusion_error"
            ] = float(
                np.mean(achieved_errors)
            )

    if output_index != EXPECTED_OUTPUT_SAMPLES:
        raise RuntimeError(
            f"Written samples: {output_index}; "
            f"expected: {EXPECTED_OUTPUT_SAMPLES}."
        )

    file_size_gb = (
        OUTPUT_PATH.stat().st_size
        / (1024 ** 3)
    )

    print("")
    print("=" * 78)
    print("CORRECTED OCCLUSION DATASET COMPLETE")
    print("=" * 78)
    print("Saved:", OUTPUT_PATH)
    print("Samples:", output_index)
    print(f"File size: {file_size_gb:.3f} GB")
    print(
        "Maximum absolute energy-ratio error:",
        f"{np.max(achieved_errors):.10f}",
    )
    print(
        "Mean absolute energy-ratio error:",
        f"{np.mean(achieved_errors):.10f}",
    )


if __name__ == "__main__":
    main()