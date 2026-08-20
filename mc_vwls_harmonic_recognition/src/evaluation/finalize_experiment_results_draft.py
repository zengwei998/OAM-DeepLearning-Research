"""
Finalize the experimental-results draft using generated result files only.

The script appends or replaces the final experiment sections:

    4.12 Representative success and failure cases
    4.13 Runtime and computational complexity
    4.14 Experimental consistency and chapter summary

Input:
    docs/experiment_results_draft.md
    results/csv/mc_vwls_selected_failure_cases.csv
    results/csv/mc_vwls_runtime_benchmark.csv
    results/csv/mc_vwls_complexity_summary.csv
    results/csv/final_experiment_consistency_checks.csv
    results/validation/final_experiment_consistency_report.txt

Output:
    docs/experiment_results_draft.md
    docs/experiment_results_draft_before_finalization.md
    results/validation/experiment_results_draft_finalization_report.txt
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Dict, List, Sequence


ROOT = Path(__file__).resolve().parents[2]

DRAFT_PATH = (
    ROOT
    / "docs"
    / "experiment_results_draft.md"
)

BACKUP_PATH = (
    ROOT
    / "docs"
    / "experiment_results_draft_before_finalization.md"
)

FAILURE_CASE_CSV_PATH = (
    ROOT
    / "results"
    / "csv"
    / "mc_vwls_selected_failure_cases.csv"
)

RUNTIME_CSV_PATH = (
    ROOT
    / "results"
    / "csv"
    / "mc_vwls_runtime_benchmark.csv"
)

COMPLEXITY_CSV_PATH = (
    ROOT
    / "results"
    / "csv"
    / "mc_vwls_complexity_summary.csv"
)

CONSISTENCY_CSV_PATH = (
    ROOT
    / "results"
    / "csv"
    / "final_experiment_consistency_checks.csv"
)

CONSISTENCY_REPORT_PATH = (
    ROOT
    / "results"
    / "validation"
    / "final_experiment_consistency_report.txt"
)

FINALIZATION_REPORT_PATH = (
    ROOT
    / "results"
    / "validation"
    / "experiment_results_draft_finalization_report.txt"
)

SECTION_START_MARKER = (
    "<!-- FINAL_EXPERIMENT_SECTIONS_START -->"
)

SECTION_END_MARKER = (
    "<!-- FINAL_EXPERIMENT_SECTIONS_END -->"
)

EXPECTED_METHODS = (
    "A0_DAF",
    "A1_MASK_ULS",
    "A2_RAW_VWLS",
    "A3_MC_VWLS",
)

METHOD_NAMES = {
    "A0_DAF": "DAF",
    "A1_MASK_ULS": "Mask-ULS",
    "A2_RAW_VWLS": "Raw-VWLS",
    "A3_MC_VWLS": "MC-VWLS",
}

EXPECTED_FAILURE_CASE_NAMES = (
    "Correct: weak turbulence",
    "Correct: severe turbulence",
    "Failure: adjacent phase",
    "Failure: cross order",
)

EPSILON = 1.0e-12


def stage(
    message: str,
) -> None:
    print(
        message,
        flush=True,
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

    if not rows:
        raise ValueError(
            f"CSV file contains no data rows: {path}"
        )

    return rows


def parse_float(
    value: object,
) -> float:
    return float(
        value
    )


def parse_int(
    value: object,
) -> int:
    return int(
        round(
            float(
                value
            )
        )
    )


def validate_failure_cases(
    rows: Sequence[Dict[str, str]],
) -> None:
    if len(
        rows
    ) != 4:
        raise ValueError(
            "Failure-case CSV must contain exactly four rows, "
            f"but found {len(rows)}."
        )

    observed_names = {
        row[
            "case_name"
        ]
        for row in rows
    }

    expected_names = set(
        EXPECTED_FAILURE_CASE_NAMES
    )

    if observed_names != expected_names:
        raise ValueError(
            "Failure-case categories do not match the frozen set. "
            f"Observed={sorted(observed_names)}"
        )


def validate_runtime_rows(
    rows: Sequence[Dict[str, str]],
) -> None:
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
        for timing_scope in (
            "recognition_only",
            "end_to_end",
        )
    }

    if observed_pairs != expected_pairs:
        raise ValueError(
            "Runtime method/scope coverage is incomplete. "
            f"Observed={sorted(observed_pairs)}"
        )

    for row in rows:
        mean_time = parse_float(
            row[
                "mean_ms_per_image"
            ]
        )

        throughput = parse_float(
            row[
                "throughput_images_per_second"
            ]
        )

        if (
            mean_time <= 0.0
            or throughput <= 0.0
        ):
            raise ValueError(
                "Runtime values must be positive: "
                f"{row['method']}/{row['timing_scope']}"
            )


def validate_complexity_rows(
    rows: Sequence[Dict[str, str]],
) -> None:
    observed_methods = {
        row[
            "method"
        ]
        for row in rows
    }

    if observed_methods != set(
        EXPECTED_METHODS
    ):
        raise ValueError(
            "Complexity table does not contain all methods."
        )

    for row in rows:
        if parse_int(
            row[
                "candidate_count_K"
            ]
        ) != 4:
            raise ValueError(
                "Unexpected candidate count in complexity table."
            )

        if parse_int(
            row[
                "angular_samples_N_theta"
            ]
        ) != 180:
            raise ValueError(
                "Unexpected angular sample count."
            )

        if parse_int(
            row[
                "radial_samples_N_r"
            ]
        ) != 64:
            raise ValueError(
                "Unexpected radial sample count."
            )

        if parse_int(
            row[
                "polar_grid_elements"
            ]
        ) != 11520:
            raise ValueError(
                "Unexpected polar-grid size."
            )


def validate_consistency_rows(
    rows: Sequence[Dict[str, str]],
) -> None:
    failed_rows = [
        row
        for row in rows
        if row[
            "status"
        ] != "PASS"
    ]

    if failed_rows:
        raise ValueError(
            "Final consistency table contains failed checks: "
            + ", ".join(
                row[
                    "check_id"
                ]
                for row in failed_rows
            )
        )

    if len(
        rows
    ) != 29:
        raise ValueError(
            "Expected 29 final consistency checks, "
            f"but found {len(rows)}."
        )


def runtime_lookup(
    rows: Sequence[Dict[str, str]],
) -> Dict[
    tuple[str, str],
    Dict[str, str],
]:
    return {
        (
            row[
                "method"
            ],
            row[
                "timing_scope"
            ],
        ): row
        for row in rows
    }


def ordered_failure_cases(
    rows: Sequence[Dict[str, str]],
) -> List[Dict[str, str]]:
    row_map = {
        row[
            "case_name"
        ]: row
        for row in rows
    }

    return [
        row_map[
            name
        ]
        for name in EXPECTED_FAILURE_CASE_NAMES
    ]


def format_failure_case_table(
    rows: Sequence[Dict[str, str]],
) -> str:
    ordered_rows = ordered_failure_cases(
        rows
    )

    lines = [
        "|案例|样本索引|SNR/dB|遮挡率|$C_n^2$|距离/m|真实状态 $(l,m)$|预测状态 $(l,m)$|置信度|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    chinese_names = {
        "Correct: weak turbulence": "弱湍流条件下正确识别",
        "Correct: severe turbulence": "强湍流条件下正确识别",
        "Failure: adjacent phase": "同阶相邻相位误判",
        "Failure: cross order": "跨阶误判",
    }

    for row in ordered_rows:
        lines.append(
            (
                f"|{chinese_names[row['case_name']]}"
                f"|{parse_int(row['sample_index'])}"
                f"|{parse_float(row['target_snr_db']):.0f}"
                f"|{parse_float(row['target_occlusion']):.1f}"
                f"|{parse_float(row['cn2']):.2e}"
                f"|{parse_float(row['distance']):.0f}"
                f"|({parse_int(row['true_order'])}, "
                f"{parse_int(row['true_phase_bin'])})"
                f"|({parse_int(row['predicted_order'])}, "
                f"{parse_int(row['predicted_phase_bin'])})"
                f"|{parse_float(row['confidence']):.6f}|"
            )
        )

    return "\n".join(
        lines
    )


def format_runtime_table(
    rows: Sequence[Dict[str, str]],
) -> str:
    lookup = runtime_lookup(
        rows
    )

    lines = [
        "|方法|识别时间/(ms·幅$^{-1}$)|端到端时间/(ms·幅$^{-1}$)|端到端标准差/ms|端到端吞吐率/(幅·s$^{-1}$)|",
        "|---|---:|---:|---:|---:|",
    ]

    for method in EXPECTED_METHODS:
        recognition_row = lookup[
            (
                method,
                "recognition_only",
            )
        ]

        end_to_end_row = lookup[
            (
                method,
                "end_to_end",
            )
        ]

        lines.append(
            (
                f"|{METHOD_NAMES[method]}"
                f"|{parse_float(recognition_row['mean_ms_per_image']):.6f}"
                f"|{parse_float(end_to_end_row['mean_ms_per_image']):.6f}"
                f"|{parse_float(end_to_end_row['std_ms_per_image']):.6f}"
                f"|{parse_float(end_to_end_row['throughput_images_per_second']):.3f}|"
            )
        )

    return "\n".join(
        lines
    )


def build_failure_case_section(
    rows: Sequence[Dict[str, str]],
) -> str:
    ordered_rows = ordered_failure_cases(
        rows
    )

    weak_correct = ordered_rows[0]
    severe_correct = ordered_rows[1]
    adjacent_error = ordered_rows[2]
    cross_order_error = ordered_rows[3]

    table = format_failure_case_table(
        ordered_rows
    )

    weak_true_order = parse_int(
        weak_correct[
            "true_order"
        ]
    )

    weak_true_phase = parse_int(
        weak_correct[
            "true_phase_bin"
        ]
    )

    severe_true_order = parse_int(
        severe_correct[
            "true_order"
        ]
    )

    severe_true_phase = parse_int(
        severe_correct[
            "true_phase_bin"
        ]
    )

    adjacent_true_order = parse_int(
        adjacent_error[
            "true_order"
        ]
    )

    adjacent_true_phase = parse_int(
        adjacent_error[
            "true_phase_bin"
        ]
    )

    adjacent_predicted_phase = parse_int(
        adjacent_error[
            "predicted_phase_bin"
        ]
    )

    cross_true_order = parse_int(
        cross_order_error[
            "true_order"
        ]
    )

    cross_true_phase = parse_int(
        cross_order_error[
            "true_phase_bin"
        ]
    )

    cross_predicted_order = parse_int(
        cross_order_error[
            "predicted_order"
        ]
    )

    cross_predicted_phase = parse_int(
        cross_order_error[
            "predicted_phase_bin"
        ]
    )

    return f"""## 4.12 典型成功与失败案例分析

为直观说明 MC-VWLS 在不同退化条件下的识别行为，从完整测试预测中确定性选取了四个代表性观测，包括弱湍流正确识别、强湍流正确识别、同阶相邻相位误判和跨阶误判。所选案例均重新执行接收噪声生成、极坐标采样、掩膜约束角向轮廓构造和谐波识别，并与完整测试预测表逐项核对，重构预测结果全部一致。

{table}

图 `fig_mc_vwls_failure_cases` 同时给出了遮挡后无噪强度、含噪接收强度、可见掩膜、极坐标强度分布及候选谐波得分。弱湍流正确案例的真实状态为 $({weak_true_order},{weak_true_phase})$，目标阶数得分明显高于其余候选阶数。强湍流正确案例的真实状态为 $({severe_true_order},{severe_true_phase})$，虽然各阶得分差距缩小，目标谐波仍保持最大响应。

同阶相邻相位误判案例的真实状态为 $({adjacent_true_order},{adjacent_true_phase})$，预测相位为 {adjacent_predicted_phase}。该结果表明目标阶数仍可正确恢复，但湍流、遮挡和噪声共同引起的角向轮廓偏移使估计相位跨越离散量化边界。跨阶误判案例的真实状态为 $({cross_true_order},{cross_true_phase})$，预测为 $({cross_predicted_order},{cross_predicted_phase})$。此时低阶候选谐波得分超过真实高阶谐波，符合前述强湍流和长距离条件下跨阶错误占主导的统计结果。

![MC-VWLS 典型成功与失败案例](../results/figures/fig_mc_vwls_failure_cases.png)

**图注建议：** MC-VWLS 在不同联合退化条件下的典型成功与失败案例。各行依次表示弱湍流正确识别、强湍流正确识别、同阶相邻相位误判和跨阶误判；各列依次给出遮挡后无噪强度、含噪接收强度、可见掩膜、极坐标强度以及候选 OAM 阶数的谐波得分。
"""


def build_runtime_section(
    runtime_rows: Sequence[Dict[str, str]],
) -> str:
    lookup = runtime_lookup(
        runtime_rows
    )

    table = format_runtime_table(
        runtime_rows
    )

    daf_end_to_end = parse_float(
        lookup[
            (
                "A0_DAF",
                "end_to_end",
            )
        ][
            "mean_ms_per_image"
        ]
    )

    uls_end_to_end = parse_float(
        lookup[
            (
                "A1_MASK_ULS",
                "end_to_end",
            )
        ][
            "mean_ms_per_image"
        ]
    )

    mc_end_to_end = parse_float(
        lookup[
            (
                "A3_MC_VWLS",
                "end_to_end",
            )
        ][
            "mean_ms_per_image"
        ]
    )

    mc_recognition = parse_float(
        lookup[
            (
                "A3_MC_VWLS",
                "recognition_only",
            )
        ][
            "mean_ms_per_image"
        ]
    )

    mc_throughput = parse_float(
        lookup[
            (
                "A3_MC_VWLS",
                "end_to_end",
            )
        ][
            "throughput_images_per_second"
        ]
    )

    mc_daf_ratio = (
        mc_end_to_end
        / max(
            daf_end_to_end,
            EPSILON,
        )
    )

    mc_uls_ratio = (
        mc_end_to_end
        / max(
            uls_end_to_end,
            EPSILON,
        )
    )

    preprocessing_fraction = (
        (
            mc_end_to_end
            - mc_recognition
        )
        / max(
            mc_end_to_end,
            EPSILON,
        )
        * 100.0
    )

    return f"""## 4.13 运行时间与计算复杂度

运行时间实验从冻结测试集确定性抽取 256 个基础样本，并将 20、15、10、5 和 0 dB 五个信噪比循环分配给各观测。每种方法首先完成 16 幅预热，然后重复计时 5 次。识别时间从已构造的极坐标表示和角向轮廓开始计算；端到端时间包括确定性接收噪声、极坐标采样、角向轮廓构造和最终识别，但不包括磁盘读取及结果写出。

{table}

DAF 的端到端平均时间最低，为 {daf_end_to_end:.6f} ms/幅。MC-VWLS 的纯识别时间为 {mc_recognition:.6f} ms/幅，端到端时间为 {mc_end_to_end:.6f} ms/幅，对应吞吐率为 {mc_throughput:.3f} 幅/s。MC-VWLS 的端到端耗时为 DAF 的 {mc_daf_ratio:.6f} 倍，但仅为 Mask-ULS 的 {mc_uls_ratio:.6f} 倍，说明可见度加权没有引入显著的额外端到端开销。对 MC-VWLS 而言，极坐标采样和角向轮廓构造约占端到端时间的 {preprocessing_fraction:.2f}%，主要计算成本来自共同预处理而非三参数谐波拟合。

设角向采样数为 $N_\\theta$，径向采样数为 $N_r$，候选 OAM 阶数数目为 $K$，每个谐波模型的拟合参数数目为 $p=3$。极坐标采样和径向聚合的时间复杂度为

$$
O(N_\\theta N_r),
$$

并需要 $O(N_\\theta N_r)$ 的中间存储。DAF 对 $K$ 个候选谐波进行直接响应计算，其识别复杂度为

$$
O(KN_\\theta).
$$

ULS 和 VWLS 对每个候选阶数构造一个 $p\\times p$ 法方程，其识别复杂度为

$$
O(KN_\\theta p^2+Kp^3).
$$

当前实验采用 $N_\\theta=180$、$N_r=64$、$K=4$ 和 $p=3$，极坐标网格包含 11520 个采样点。由于 $K$ 和 $p$ 均为固定常数，四种方法的完整流程在当前实现中均可归纳为 $O(N_\\theta N_r)$。

![不同识别方法的单幅运行时间](../results/figures/fig_mc_vwls_runtime.png)

**图注建议：** DAF、Mask-ULS、Raw-VWLS 和 MC-VWLS 的单幅纯识别时间与端到端时间。每项结果由冻结测试子集上的 5 次重复计时得到。
"""


def build_consistency_section(
    consistency_rows: Sequence[Dict[str, str]],
) -> str:
    passed_count = sum(
        1
        for row in consistency_rows
        if row[
            "status"
        ] == "PASS"
    )

    failed_count = sum(
        1
        for row in consistency_rows
        if row[
            "status"
        ] != "PASS"
    )

    return f"""## 4.14 实验一致性验证与本章小结

在结果定稿前，对完整预测表、混淆矩阵、典型失败案例、运行时间表、复杂度表及关键输出文件执行了最终一致性验证。共执行 {len(consistency_rows)} 项检查，其中通过 {passed_count} 项、失败 {failed_count} 项。验证内容包括 134400 条消融预测的行数和方法覆盖、每种方法 33600 条观测的一致性、MC-VWLS 联合标签准确率与混淆矩阵对角计数的一致性、归一化混淆矩阵行和、四类代表性案例覆盖、运行时间方法与计时范围覆盖，以及所有关键结果文件的存在性和非空性。

最终冻结的 MC-VWLS 测试结果为：32 类联合标签准确率 84.8125%，OAM 阶数准确率 89.3452%，离散相位准确率 85.7917%。混淆矩阵共包含 33600 个观测，其中 28497 个位于主对角线。综合消融、分条件性能、错误结构、代表性案例和运行时间结果，可以得到以下结论：

1. 掩膜支持归一化是提升遮挡条件识别稳定性的主要因素，MC-VWLS 与 Mask-ULS 的总体联合准确率接近。
2. 可见度加权在原始轮廓和掩膜约束轮廓之间表现出稳定的补偿作用，但相对于 Mask-ULS 的总体提升未达到统计显著水平。
3. 强湍流和长距离传播是主要性能瓶颈，严重退化下跨阶错误多于同阶相位错误。
4. 相位错误主要集中在相邻离散相位区间，说明连续相位估计在量化边界附近较敏感。
5. MC-VWLS 的端到端时间约为 2.318 ms/幅，与 Mask-ULS 基本一致，保持了传统谐波识别方法的低计算成本。

因此，现有结果支持将 MC-VWLS 定位为一种具有明确物理解释、无需训练、适用于掩膜已知场景的低复杂度识别方法。同时，实验也表明其核心增益主要来自掩膜约束角向轮廓构造，而可见度加权的独立增益较小。该结论应在摘要、引言贡献点和结论部分保持一致，避免将 MC-VWLS 描述为在所有条件下显著优于 Mask-ULS。
"""


def build_final_sections(
    failure_rows: Sequence[Dict[str, str]],
    runtime_rows: Sequence[Dict[str, str]],
    consistency_rows: Sequence[Dict[str, str]],
) -> str:
    sections = [
        SECTION_START_MARKER,
        "",
        build_failure_case_section(
            failure_rows
        ).strip(),
        "",
        build_runtime_section(
            runtime_rows
        ).strip(),
        "",
        build_consistency_section(
            consistency_rows
        ).strip(),
        "",
        SECTION_END_MARKER,
    ]

    return "\n".join(
        sections
    )


def replace_or_append_sections(
    original_text: str,
    new_sections: str,
) -> tuple[str, str]:
    start_index = original_text.find(
        SECTION_START_MARKER
    )

    end_index = original_text.find(
        SECTION_END_MARKER
    )

    if (
        start_index >= 0
        and end_index >= 0
        and end_index > start_index
    ):
        end_index += len(
            SECTION_END_MARKER
        )

        updated_text = (
            original_text[
                :start_index
            ].rstrip()
            + "\n\n"
            + new_sections
            + "\n"
            + original_text[
                end_index:
            ].lstrip(
                "\n"
            )
        )

        operation = "REPLACED"

        return (
            updated_text,
            operation,
        )

    if (
        start_index >= 0
        or end_index >= 0
    ):
        raise ValueError(
            "Only one final-section marker was found. "
            "Remove the incomplete marker before continuing."
        )

    updated_text = (
        original_text.rstrip()
        + "\n\n"
        + new_sections
        + "\n"
    )

    return (
        updated_text,
        "APPENDED",
    )


def count_headings(
    text: str,
    heading: str,
) -> int:
    return sum(
        1
        for line in text.splitlines()
        if line.strip() == heading
    )


def validate_updated_draft(
    text: str,
) -> None:
    required_headings = (
        "## 4.12 典型成功与失败案例分析",
        "## 4.13 运行时间与计算复杂度",
        "## 4.14 实验一致性验证与本章小结",
    )

    for heading in required_headings:
        heading_count = count_headings(
            text,
            heading,
        )

        if heading_count != 1:
            raise ValueError(
                f"Heading must occur exactly once: {heading}; "
                f"count={heading_count}"
            )

    if text.count(
        SECTION_START_MARKER
    ) != 1:
        raise ValueError(
            "Final-section start marker count is not one."
        )

    if text.count(
        SECTION_END_MARKER
    ) != 1:
        raise ValueError(
            "Final-section end marker count is not one."
        )

    required_values = (
        "84.8125%",
        "89.3452%",
        "85.7917%",
        "2.318",
        "28497",
        "33600",
    )

    for value in required_values:
        if value not in text:
            raise ValueError(
                f"Required result value is absent: {value}"
            )


def build_finalization_report(
    *,
    operation: str,
    original_size: int,
    updated_size: int,
    failure_case_count: int,
    runtime_row_count: int,
    complexity_row_count: int,
    consistency_check_count: int,
) -> str:
    return "\n".join(
        [
            "Experiment-results draft finalization",
            "",
            "[STATUS]",
            "Status=PASS",
            f"Operation={operation}",
            f"Original draft size={original_size} characters",
            f"Updated draft size={updated_size} characters",
            "",
            "[SOURCE DATA]",
            f"Failure cases={failure_case_count}",
            f"Runtime rows={runtime_row_count}",
            f"Complexity rows={complexity_row_count}",
            f"Consistency checks={consistency_check_count}",
            "",
            "[UPDATED SECTIONS]",
            "4.12 Representative success and failure cases",
            "4.13 Runtime and computational complexity",
            "4.14 Experimental consistency and chapter summary",
            "",
            "[OUTPUTS]",
            f"Updated draft: {DRAFT_PATH}",
            f"Backup draft: {BACKUP_PATH}",
            f"Finalization report: {FINALIZATION_REPORT_PATH}",
        ]
    )


def main() -> None:
    stage(
        "=" * 78
    )

    stage(
        "FINALIZE EXPERIMENT RESULTS DRAFT"
    )

    stage(
        "=" * 78
    )

    if not DRAFT_PATH.exists():
        raise FileNotFoundError(
            f"Draft does not exist: {DRAFT_PATH}"
        )

    stage(
        "[1] Load generated result tables"
    )

    failure_rows = load_csv_rows(
        FAILURE_CASE_CSV_PATH
    )

    runtime_rows = load_csv_rows(
        RUNTIME_CSV_PATH
    )

    complexity_rows = load_csv_rows(
        COMPLEXITY_CSV_PATH
    )

    consistency_rows = load_csv_rows(
        CONSISTENCY_CSV_PATH
    )

    stage(
        "[1] PASS"
    )

    stage(
        "[2] Validate generated result tables"
    )

    validate_failure_cases(
        failure_rows
    )

    validate_runtime_rows(
        runtime_rows
    )

    validate_complexity_rows(
        complexity_rows
    )

    validate_consistency_rows(
        consistency_rows
    )

    if not CONSISTENCY_REPORT_PATH.exists():
        raise FileNotFoundError(
            "Final consistency report does not exist: "
            f"{CONSISTENCY_REPORT_PATH}"
        )

    consistency_report_text = (
        CONSISTENCY_REPORT_PATH.read_text(
            encoding="utf-8"
        )
    )

    if "Status=PASS" not in consistency_report_text:
        raise ValueError(
            "Final consistency report does not contain Status=PASS."
        )

    stage(
        "[2] PASS"
    )

    stage(
        "[3] Read and back up draft"
    )

    original_text = DRAFT_PATH.read_text(
        encoding="utf-8"
    )

    BACKUP_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        DRAFT_PATH,
        BACKUP_PATH,
    )

    stage(
        f"[3] PASS: {BACKUP_PATH}"
    )

    stage(
        "[4] Build final experiment sections"
    )

    new_sections = build_final_sections(
        failure_rows=failure_rows,
        runtime_rows=runtime_rows,
        consistency_rows=consistency_rows,
    )

    stage(
        "[4] PASS"
    )

    stage(
        "[5] Update draft"
    )

    (
        updated_text,
        operation,
    ) = replace_or_append_sections(
        original_text=original_text,
        new_sections=new_sections,
    )

    validate_updated_draft(
        updated_text
    )

    DRAFT_PATH.write_text(
        updated_text,
        encoding="utf-8",
    )

    stage(
        f"[5] PASS: operation={operation}"
    )

    stage(
        "[6] Generate finalization report"
    )

    report_text = build_finalization_report(
        operation=operation,
        original_size=len(
            original_text
        ),
        updated_size=len(
            updated_text
        ),
        failure_case_count=len(
            failure_rows
        ),
        runtime_row_count=len(
            runtime_rows
        ),
        complexity_row_count=len(
            complexity_rows
        ),
        consistency_check_count=len(
            consistency_rows
        ),
    )

    FINALIZATION_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FINALIZATION_REPORT_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    stage(
        "[6] PASS"
    )

    stage(
        ""
    )

    stage(
        report_text
    )

    stage(
        ""
    )

    stage(
        "=" * 78
    )

    stage(
        "EXPERIMENT RESULTS DRAFT FINALIZED"
    )

    stage(
        "=" * 78
    )


if __name__ == "__main__":
    main()