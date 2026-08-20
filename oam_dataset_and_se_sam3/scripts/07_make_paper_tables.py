import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

out_dir = Path("outputs/paper_tables")
out_dir.mkdir(parents=True, exist_ok=True)

eval_dir = Path("outputs/evaluation")
group_dir = Path("outputs/group_analysis")


def save_table(df, name):
    csv_path = out_dir / f"{name}.csv"
    txt_path = out_dir / f"{name}.txt"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(df.to_string(index=False))

    print("Saved:", csv_path)
    print("Saved:", txt_path)


# 表4-1：四模型整体性能
table4_1 = pd.read_csv(eval_dir / "table4_1_model_comparison.csv")
table4_1 = table4_1[[
    "model",
    "accuracy_percent",
    "precision_macro",
    "recall_macro",
    "f1_macro"
]]
table4_1.columns = [
    "模型",
    "准确率(%)",
    "宏平均精确率",
    "宏平均召回率",
    "宏平均F1"
]
save_table(table4_1, "table4_1_model_comparison")


# 表4-2：不同SNR下四模型准确率
models = {
    "ResNet-50": "ResNet_50_snr_accuracy.csv",
    "Swin-T": "Swin_T_snr_accuracy.csv",
    "SAM3 baseline": "SAM3_baseline_snr_accuracy.csv",
    "SE-SAM3 (Ours)": "SE_SAM3_Ours_snr_accuracy.csv",
}

snr_table = None
for model_name, file_name in models.items():
    df = pd.read_csv(group_dir / file_name)
    df = df[["snr_db", "accuracy_percent"]]
    df.columns = ["SNR(dB)", model_name]

    if snr_table is None:
        snr_table = df
    else:
        snr_table = pd.merge(snr_table, df, on="SNR(dB)")

save_table(snr_table, "table4_2_snr_accuracy")


# 表4-3：不同Cn2下四模型准确率
models = {
    "ResNet-50": "ResNet_50_cn2_accuracy.csv",
    "Swin-T": "Swin_T_cn2_accuracy.csv",
    "SAM3 baseline": "SAM3_baseline_cn2_accuracy.csv",
    "SE-SAM3 (Ours)": "SE_SAM3_Ours_cn2_accuracy.csv",
}

cn2_table = None
for model_name, file_name in models.items():
    df = pd.read_csv(group_dir / file_name)
    df = df[["cn2", "accuracy_percent"]]
    df.columns = ["Cn2", model_name]

    if cn2_table is None:
        cn2_table = df
    else:
        cn2_table = pd.merge(cn2_table, df, on="Cn2")

save_table(cn2_table, "table4_3_cn2_accuracy")


# 表4-4：不同OAM阶数下四模型准确率
models = {
    "ResNet-50": "ResNet_50_charge_accuracy.csv",
    "Swin-T": "Swin_T_charge_accuracy.csv",
    "SAM3 baseline": "SAM3_baseline_charge_accuracy.csv",
    "SE-SAM3 (Ours)": "SE_SAM3_Ours_charge_accuracy.csv",
}

charge_table = None
for model_name, file_name in models.items():
    df = pd.read_csv(group_dir / file_name)
    df = df[["topological_charge_abs", "accuracy_percent"]]
    df.columns = ["|l|", model_name]

    if charge_table is None:
        charge_table = df
    else:
        charge_table = pd.merge(charge_table, df, on="|l|")

save_table(charge_table, "table4_4_oam_order_accuracy")

print("\n全部论文表格已生成到：outputs/paper_tables")