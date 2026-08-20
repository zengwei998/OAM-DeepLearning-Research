"""
Generate MC-VWLS representative cases without Matplotlib.

This script uses Pillow for all rendering to avoid the native
Matplotlib/BLAS crash in the current Windows environment.

Outputs:
    results/figures/fig_mc_vwls_failure_cases.png
    results/figures/fig_mc_vwls_failure_cases.pdf
    results/csv/mc_vwls_selected_failure_cases.csv
    results/validation/mc_vwls_failure_cases_report.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.algorithms.harmonic_fit import HarmonicRecognition
from src.algorithms.polar_sampling import PolarProfile
from src.evaluation.plot_failure_cases import (
    CANDIDATE_ORDERS,
    DATASET_PATH,
    PDF_PATH,
    PNG_PATH,
    PREDICTION_PATH,
    REPORT_PATH,
    SELECTED_CASES_PATH,
    attach_conditions,
    build_report,
    load_prediction_rows,
    reconstruct_case,
    save_selected_cases,
    select_cases,
)


CELL_WIDTH = 330
CELL_HEIGHT = 300

LEFT_MARGIN = 20
TOP_MARGIN = 50
BOTTOM_MARGIN = 30

HEADER_HEIGHT = 42
ROW_LABEL_WIDTH = 320

COLUMN_COUNT = 5

CANVAS_WIDTH = (
    ROW_LABEL_WIDTH
    + COLUMN_COUNT * CELL_WIDTH
)

BACKGROUND = (
    255,
    255,
    255,
)

TEXT_COLOR = (
    20,
    20,
    20,
)

GRID_COLOR = (
    170,
    170,
    170,
)

BAR_COLOR = (
    85,
    125,
    175,
)

PREDICTED_BAR_COLOR = (
    210,
    105,
    45,
)

TRUE_ORDER_COLOR = (
    20,
    120,
    60,
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

    for path in candidate_paths:
        if path.exists():
            return ImageFont.truetype(
                str(path),
                size=size,
            )

    return ImageFont.load_default()


TITLE_FONT = get_font(
    20
)

HEADER_FONT = get_font(
    17
)

NORMAL_FONT = get_font(
    14
)

SMALL_FONT = get_font(
    12
)

TINY_FONT = get_font(
    10
)


def robust_limits(
    array: np.ndarray,
) -> tuple[float, float]:
    values = np.asarray(
        array,
        dtype=np.float64,
    )

    finite_values = values[
        np.isfinite(
            values
        )
    ]

    if finite_values.size == 0:
        return (
            0.0,
            1.0,
        )

    lower = float(
        np.percentile(
            finite_values,
            1.0,
        )
    )

    upper = float(
        np.percentile(
            finite_values,
            99.5,
        )
    )

    if not np.isfinite(
        lower
    ):
        lower = 0.0

    if not np.isfinite(
        upper
    ):
        upper = lower + 1.0

    if upper <= lower:
        upper = lower + 1.0

    return (
        lower,
        upper,
    )


def normalize_to_uint8(
    array: np.ndarray,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> np.ndarray:
    values = np.asarray(
        array,
        dtype=np.float64,
    )

    if lower is None or upper is None:
        calculated_lower, calculated_upper = robust_limits(
            values
        )

        if lower is None:
            lower = calculated_lower

        if upper is None:
            upper = calculated_upper

    denominator = max(
        float(
            upper
            - lower
        ),
        1.0e-15,
    )

    normalized = (
        values
        - float(
            lower
        )
    ) / denominator

    normalized = np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    return np.asarray(
        np.round(
            normalized
            * 255.0
        ),
        dtype=np.uint8,
    )


def pseudo_color(
    grayscale: np.ndarray,
) -> np.ndarray:
    value = np.asarray(
        grayscale,
        dtype=np.float64,
    ) / 255.0

    red = np.clip(
        1.5 * value,
        0.0,
        1.0,
    )

    green = np.clip(
        1.5
        - 2.0
        * np.abs(
            value
            - 0.5
        ),
        0.0,
        1.0,
    )

    blue = np.clip(
        1.5
        * (
            1.0
            - value
        ),
        0.0,
        1.0,
    )

    rgb = np.stack(
        [
            red,
            green,
            blue,
        ],
        axis=-1,
    )

    return np.asarray(
        np.round(
            rgb
            * 255.0
        ),
        dtype=np.uint8,
    )


def create_array_image(
    array: np.ndarray,
    *,
    width: int,
    height: int,
    use_color: bool,
    lower: float | None = None,
    upper: float | None = None,
) -> Image.Image:
    grayscale = normalize_to_uint8(
        array,
        lower=lower,
        upper=upper,
    )

    if use_color:
        rgb = pseudo_color(
            grayscale
        )

        image = Image.fromarray(
            rgb,
            mode="RGB",
        )
    else:
        image = Image.fromarray(
            grayscale,
            mode="L",
        ).convert(
            "RGB"
        )

    return image.resize(
        (
            width,
            height,
        ),
        resample=Image.Resampling.BILINEAR,
    )


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
        spacing=3,
        align="center",
    )

    text_width = (
        text_box[2]
        - text_box[0]
    )

    text_height = (
        text_box[3]
        - text_box[1]
    )

    x = (
        left
        + (
            right
            - left
            - text_width
        )
        // 2
    )

    y = (
        top
        + (
            bottom
            - top
            - text_height
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
        spacing=3,
        align="center",
    )


def paste_image_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    image: Image.Image,
    left: int,
    top: int,
    title: str,
) -> None:
    title_height = 28

    draw.rectangle(
        (
            left,
            top,
            left + CELL_WIDTH - 1,
            top + CELL_HEIGHT - 1,
        ),
        outline=GRID_COLOR,
        width=1,
    )

    draw_centered_text(
        draw,
        (
            left + 2,
            top + 2,
            left + CELL_WIDTH - 2,
            top + title_height,
        ),
        title,
        NORMAL_FONT,
        TEXT_COLOR,
    )

    available_width = (
        CELL_WIDTH
        - 18
    )

    available_height = (
        CELL_HEIGHT
        - title_height
        - 18
    )

    image_copy = image.copy()

    image_copy.thumbnail(
        (
            available_width,
            available_height,
        ),
        resample=Image.Resampling.LANCZOS,
    )

    paste_x = (
        left
        + (
            CELL_WIDTH
            - image_copy.width
        )
        // 2
    )

    paste_y = (
        top
        + title_height
        + (
            available_height
            - image_copy.height
        )
        // 2
        + 5
    )

    canvas.paste(
        image_copy,
        (
            paste_x,
            paste_y,
        ),
    )


def draw_score_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    recognition: HarmonicRecognition,
    true_order: int,
    left: int,
    top: int,
) -> None:
    draw.rectangle(
        (
            left,
            top,
            left + CELL_WIDTH - 1,
            top + CELL_HEIGHT - 1,
        ),
        outline=GRID_COLOR,
        width=1,
    )

    title = (
        "Harmonic scores\n"
        f"confidence={recognition.confidence:.4f}, "
        f"margin={recognition.harmonic_margin:.4f}"
    )

    draw_centered_text(
        draw,
        (
            left + 3,
            top + 3,
            left + CELL_WIDTH - 3,
            top + 52,
        ),
        title,
        SMALL_FONT,
        TEXT_COLOR,
    )

    chart_left = (
        left
        + 45
    )

    chart_right = (
        left
        + CELL_WIDTH
        - 18
    )

    chart_top = (
        top
        + 66
    )

    chart_bottom = (
        top
        + CELL_HEIGHT
        - 43
    )

    draw.line(
        (
            chart_left,
            chart_top,
            chart_left,
            chart_bottom,
        ),
        fill=TEXT_COLOR,
        width=1,
    )

    draw.line(
        (
            chart_left,
            chart_bottom,
            chart_right,
            chart_bottom,
        ),
        fill=TEXT_COLOR,
        width=1,
    )

    scores = [
        float(
            recognition.candidates[
                order
            ].score
        )
        for order in CANDIDATE_ORDERS
    ]

    maximum_score = max(
        max(
            scores
        ),
        1.0e-12,
    )

    available_width = (
        chart_right
        - chart_left
    )

    slot_width = (
        available_width
        / len(
            CANDIDATE_ORDERS
        )
    )

    maximum_bar_height = (
        chart_bottom
        - chart_top
        - 15
    )

    for index, (
        order,
        score,
    ) in enumerate(
        zip(
            CANDIDATE_ORDERS,
            scores,
        )
    ):
        bar_height = int(
            maximum_bar_height
            * score
            / maximum_score
        )

        bar_left = int(
            chart_left
            + index
            * slot_width
            + 12
        )

        bar_right = int(
            chart_left
            + (
                index
                + 1
            )
            * slot_width
            - 12
        )

        bar_top = (
            chart_bottom
            - bar_height
        )

        fill = (
            PREDICTED_BAR_COLOR
            if order
            == recognition.predicted_order
            else BAR_COLOR
        )

        draw.rectangle(
            (
                bar_left,
                bar_top,
                bar_right,
                chart_bottom,
            ),
            fill=fill,
            outline=TEXT_COLOR,
            width=1,
        )

        if order == true_order:
            draw.rectangle(
                (
                    bar_left - 3,
                    bar_top - 3,
                    bar_right + 3,
                    chart_bottom + 3,
                ),
                outline=TRUE_ORDER_COLOR,
                width=3,
            )

        score_text = (
            f"{score:.3f}"
        )

        text_box = draw.textbbox(
            (
                0,
                0,
            ),
            score_text,
            font=TINY_FONT,
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
                    bar_top - 16,
                ),
            ),
            score_text,
            font=TINY_FONT,
            fill=TEXT_COLOR,
        )

        order_text = (
            f"l={order}"
        )

        text_box = draw.textbbox(
            (
                0,
                0,
            ),
            order_text,
            font=SMALL_FONT,
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
                chart_bottom + 8,
            ),
            order_text,
            font=SMALL_FONT,
            fill=TEXT_COLOR,
        )

    legend_text = (
        "Orange: predicted    "
        "Green frame: true"
    )

    draw_centered_text(
        draw,
        (
            left + 4,
            top + CELL_HEIGHT - 29,
            left + CELL_WIDTH - 4,
            top + CELL_HEIGHT - 4,
        ),
        legend_text,
        TINY_FONT,
        TEXT_COLOR,
    )


def format_row_label(
    row: Dict[str, object],
    recognition: HarmonicRecognition,
) -> str:
    return (
        f"{row['case_name']}\n\n"
        f"Sample: {row['sample_index_int']}\n"
        f"True: l={row['true_order_int']}, "
        f"phase={row['true_phase_bin_int']}\n"
        f"Predicted: l={recognition.predicted_order}, "
        f"phase={recognition.predicted_phase_bin}\n"
        f"SNR: {float(row['target_snr_db_float']):.0f} dB\n"
        f"Occlusion: "
        f"{float(row['target_occlusion_float']):.1f}\n"
        f"Cn2: {float(row['cn2']):.2e}\n"
        f"Distance: {float(row['distance']):.0f} m"
    )


def generate_figure(
    reconstructed_cases: Sequence[Dict[str, object]],
) -> None:
    row_count = len(
        reconstructed_cases
    )

    canvas_height = (
        HEADER_HEIGHT
        + row_count
        * CELL_HEIGHT
        + BOTTOM_MARGIN
    )

    canvas = Image.new(
        "RGB",
        (
            CANVAS_WIDTH,
            canvas_height,
        ),
        BACKGROUND,
    )

    draw = ImageDraw.Draw(
        canvas
    )

    draw_centered_text(
        draw,
        (
            0,
            0,
            CANVAS_WIDTH,
            HEADER_HEIGHT,
        ),
        "Representative MC-VWLS success and failure cases",
        TITLE_FONT,
        TEXT_COLOR,
    )

    for row_index, case in enumerate(
        reconstructed_cases
    ):
        print(
            f"    [PIL {row_index + 1}] Render case",
            flush=True,
        )

        row = case[
            "row"
        ]

        recognition = case[
            "recognition"
        ]

        polar = case[
            "polar"
        ]

        if not isinstance(
            recognition,
            HarmonicRecognition,
        ):
            raise TypeError(
                "Unexpected recognition object."
            )

        if not isinstance(
            polar,
            PolarProfile,
        ):
            raise TypeError(
                "Unexpected polar-profile object."
            )

        row_top = (
            HEADER_HEIGHT
            + row_index
            * CELL_HEIGHT
        )

        draw.rectangle(
            (
                0,
                row_top,
                ROW_LABEL_WIDTH - 1,
                row_top + CELL_HEIGHT - 1,
            ),
            outline=GRID_COLOR,
            width=1,
        )

        row_label = format_row_label(
            row,
            recognition,
        )

        draw_centered_text(
            draw,
            (
                10,
                row_top + 10,
                ROW_LABEL_WIDTH - 10,
                row_top + CELL_HEIGHT - 10,
            ),
            row_label,
            NORMAL_FONT,
            TEXT_COLOR,
        )

        clean_image = create_array_image(
            np.asarray(
                case[
                    "clean_intensity"
                ]
            ),
            width=290,
            height=245,
            use_color=True,
        )

        noisy_image = create_array_image(
            np.asarray(
                case[
                    "noisy_intensity"
                ]
            ),
            width=290,
            height=245,
            use_color=True,
        )

        mask_image = create_array_image(
            np.asarray(
                case[
                    "visible_mask"
                ]
            ),
            width=290,
            height=245,
            use_color=False,
            lower=0.0,
            upper=1.0,
        )

        polar_image = create_array_image(
            np.asarray(
                polar.polar_intensity
            ),
            width=300,
            height=220,
            use_color=True,
        )

        first_column_left = (
            ROW_LABEL_WIDTH
        )

        paste_image_panel(
            canvas,
            draw,
            image=clean_image,
            left=first_column_left,
            top=row_top,
            title="Occluded clean intensity",
        )

        paste_image_panel(
            canvas,
            draw,
            image=noisy_image,
            left=first_column_left
            + CELL_WIDTH,
            top=row_top,
            title="Noisy receiver intensity",
        )

        paste_image_panel(
            canvas,
            draw,
            image=mask_image,
            left=first_column_left
            + 2
            * CELL_WIDTH,
            top=row_top,
            title="Visible mask",
        )

        paste_image_panel(
            canvas,
            draw,
            image=polar_image,
            left=first_column_left
            + 3
            * CELL_WIDTH,
            top=row_top,
            title="Polar intensity",
        )

        draw_score_panel(
            canvas,
            draw,
            recognition=recognition,
            true_order=int(
                row[
                    "true_order_int"
                ]
            ),
            left=first_column_left
            + 4
            * CELL_WIDTH,
            top=row_top,
        )

    PNG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "    [SAVE] PNG",
        flush=True,
    )

    canvas.save(
        PNG_PATH,
        format="PNG",
        optimize=True,
    )

    print(
        "    [SAVE] PDF",
        flush=True,
    )

    canvas.save(
        PDF_PATH,
        format="PDF",
        resolution=300.0,
    )


def main() -> None:
    print(
        "=" * 78,
        flush=True,
    )

    print(
        "GENERATE FAILURE CASES WITH PILLOW",
        flush=True,
    )

    print(
        "=" * 78,
        flush=True,
    )

    print(
        "[1] Load prediction CSV",
        flush=True,
    )

    prediction_rows = load_prediction_rows(
        PREDICTION_PATH
    )

    print(
        f"[1] PASS: rows={len(prediction_rows)}",
        flush=True,
    )

    reconstructed_cases: List[
        Dict[str, object]
    ] = []

    print(
        "[2] Open HDF5",
        flush=True,
    )

    with h5py.File(
        DATASET_PATH,
        "r",
    ) as h5_file:
        conditions = np.asarray(
            h5_file[
                "conditions"
            ][
                :
            ],
            dtype=np.float64,
        )

        enriched_rows = attach_conditions(
            prediction_rows,
            conditions,
        )

        selected_cases = select_cases(
            enriched_rows
        )

        print(
            f"[2] PASS: selected={len(selected_cases)}",
            flush=True,
        )

        for case_index, row in enumerate(
            selected_cases,
            start=1,
        ):
            print(
                (
                    f"[3.{case_index}] Reconstruct "
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
                f"[3.{case_index}] PASS",
                flush=True,
            )

    print(
        "[4] Save selected-case CSV",
        flush=True,
    )

    save_selected_cases(
        reconstructed_cases
    )

    print(
        "[4] PASS",
        flush=True,
    )

    print(
        "[5] Render Pillow figure",
        flush=True,
    )

    generate_figure(
        reconstructed_cases
    )

    print(
        "[5] PASS",
        flush=True,
    )

    print(
        "[6] Save report",
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
        "[6] PASS",
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
        "PILLOW FAILURE CASE FIGURE COMPLETE",
        flush=True,
    )

    print(
        "=" * 78,
        flush=True,
    )

    print(
        "CSV:",
        SELECTED_CASES_PATH,
        flush=True,
    )

    print(
        "PNG:",
        PNG_PATH,
        flush=True,
    )

    print(
        "PDF:",
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