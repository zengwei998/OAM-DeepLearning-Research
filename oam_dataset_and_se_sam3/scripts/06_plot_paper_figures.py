import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

in_dir = Path("outputs/group_analysis")
out_dir = Path("outputs/paper_figures")
out_dir.mkdir(parents=True, exist_ok=True)


models = [
    ("ResNet-50", "ResNet_50"),
    ("Swin-T", "Swin_T"),
    ("SAM3 baseline", "SAM3_baseline"),
    ("SE-SAM3 (Ours)", "SE_SAM3_Ours"),
]


def plot_compare(file_suffix, x_col, title, save_name):
    plt.figure(figsize=(8, 5))

    for label, file_prefix in models:
        csv_path = in_dir / f"{file_prefix}_{file_suffix}.csv"
        df = pd.read_csv(csv_path)

        plt.plot(
            df[x_col].astype(str),
            df["accuracy_percent"],
            marker="o",
            label=label
        )

    plt.xlabel(x_col)
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = out_dir / save_name
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved:", save_path)


def main():
    plot_compare(
        file_suffix="snr_accuracy",
        x_col="snr_db",
        title="Accuracy under Different SNR Conditions",
        save_name="fig_snr_comparison.png"
    )

    plot_compare(
        file_suffix="cn2_accuracy",
        x_col="cn2",
        title="Accuracy under Different Turbulence Strengths",
        save_name="fig_cn2_comparison.png"
    )

    plot_compare(
        file_suffix="charge_accuracy",
        x_col="topological_charge_abs",
        title="Accuracy under Different OAM Orders",
        save_name="fig_oam_order_comparison.png"
    )

    print("\n全部论文图已生成到：", out_dir)


if __name__ == "__main__":
    main()