import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import csv
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tv_models
import timm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.oam_dataset import OAMDataset
from src.models_se_sam3 import SEBlock, PhysicsPrior


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 11
batch_size = 64


class SESAM3PPI_Eval(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()

        backbone = tv_models.resnet50(weights=None)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.se = SEBlock(2048)
        self.ppi = PhysicsPrior(2048)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.se(x)
        x = self.ppi(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def load_state(model, path):
    obj = torch.load(path, map_location=device)

    if isinstance(obj, dict) and "model_state_dict" in obj:
        state = obj["model_state_dict"]
    else:
        state = obj

    model.load_state_dict(state)
    return model


def predict(model, loader):
    model.to(device)
    model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc="Predicting", ncols=100):
            x = x.to(device)
            out = model(x)
            pred = out.argmax(dim=1).cpu().numpy()

            preds.extend(pred.tolist())
            labels.extend(y.numpy().tolist())

    return np.array(labels), np.array(preds)


def group_accuracy(df, group_col):
    rows = []

    for value, sub in df.groupby(group_col):
        acc = (sub["label"].values == sub["pred"].values).mean()
        rows.append({
            group_col: value,
            "accuracy": acc,
            "accuracy_percent": acc * 100,
            "samples": len(sub),
        })

    return pd.DataFrame(rows).sort_values(group_col)


def plot_group(df, x_col, y_col, title, save_path):
    plt.figure(figsize=(7, 5))
    plt.plot(df[x_col].astype(str), df[y_col], marker="o")
    plt.xlabel(x_col)
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    out_dir = Path("outputs/group_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    test_csv = "data/manifests/test.csv"
    test_df_base = pd.read_csv(test_csv)

    test_set = OAMDataset(test_csv, image_size=224, train=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    models = {}

    print("Loading ResNet-50...")
    resnet = tv_models.resnet50(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    models["ResNet-50"] = load_state(resnet, "checkpoints/resnet50/best.pt")

    print("Loading Swin-T...")
    swin = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        num_classes=num_classes,
    )
    models["Swin-T"] = load_state(swin, "best_swin.pth")

    print("Loading SAM3 baseline...")
    sam_backbone = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=0,
    )
    sam_classifier = nn.Linear(sam_backbone.embed_dim, num_classes)
    sam3 = nn.Sequential(sam_backbone, sam_classifier)
    models["SAM3 baseline"] = load_state(sam3, "best_sam3_baseline.pth")

    print("Loading SE-SAM3...")
    se_sam3 = SESAM3PPI_Eval(num_classes=num_classes)
    models["SE-SAM3 (Ours)"] = load_state(se_sam3, "checkpoints/sesam3/best.pt")

    summary_rows = []

    for model_name, model in models.items():
        print(f"\nEvaluating groups: {model_name}")

        labels, preds = predict(model, test_loader)

        df = test_df_base.copy()
        df["label"] = labels
        df["pred"] = preds
        df["correct"] = df["label"] == df["pred"]

        safe_name = (
            model_name.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
        )

        snr_df = group_accuracy(df, "snr_db")
        cn2_df = group_accuracy(df, "cn2")
        l_df = group_accuracy(df, "topological_charge_abs")

        snr_df.to_csv(out_dir / f"{safe_name}_snr_accuracy.csv", index=False, encoding="utf-8-sig")
        cn2_df.to_csv(out_dir / f"{safe_name}_cn2_accuracy.csv", index=False, encoding="utf-8-sig")
        l_df.to_csv(out_dir / f"{safe_name}_charge_accuracy.csv", index=False, encoding="utf-8-sig")

        plot_group(
            snr_df,
            "snr_db",
            "accuracy_percent",
            f"{model_name} Accuracy under Different SNR",
            out_dir / f"{safe_name}_snr_curve.png",
        )

        plot_group(
            cn2_df,
            "cn2",
            "accuracy_percent",
            f"{model_name} Accuracy under Different Cn2",
            out_dir / f"{safe_name}_cn2_curve.png",
        )

        plot_group(
            l_df,
            "topological_charge_abs",
            "accuracy_percent",
            f"{model_name} Accuracy under Different OAM Orders",
            out_dir / f"{safe_name}_charge_curve.png",
        )

        summary_rows.append({
            "model": model_name,
            "overall_accuracy_percent": df["correct"].mean() * 100,
            "snr_-5_accuracy_percent": float(snr_df[snr_df["snr_db"] == -5]["accuracy_percent"].iloc[0]),
            "strong_cn2_accuracy_percent": float(cn2_df.iloc[-1]["accuracy_percent"]),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "group_summary.csv", index=False, encoding="utf-8-sig")

    print("\n==============================")
    print("GROUP ANALYSIS FINISHED")
    print("==============================")
    print(summary_df)
    print(f"\n结果已保存到: {out_dir}")


if __name__ == "__main__":
    main()