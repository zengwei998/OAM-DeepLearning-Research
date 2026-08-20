import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import csv
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tv_models
import timm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.oam_dataset import OAMDataset
from src.models_se_sam3 import SEBlock, PhysicsPrior


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

print("Current Working Dir:", os.getcwd())

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
        x = self.fc(x)
        return x


def load_state(model, path):
    obj = torch.load(path, map_location=device)

    if isinstance(obj, dict) and "model_state_dict" in obj:
        state = obj["model_state_dict"]
    else:
        state = obj

    model.load_state_dict(state)
    return model


def evaluate_model(model, loader):
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc="Evaluating", ncols=100):
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            pred = out.argmax(dim=1)

            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = (y_true == y_pred).mean()

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    precision_list = []
    recall_list = []
    f1_list = []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {
        "accuracy": acc,
        "precision_macro": float(np.mean(precision_list)),
        "recall_macro": float(np.mean(recall_list)),
        "f1_macro": float(np.mean(f1_list)),
        "confusion_matrix": cm,
    }


def save_confusion_matrix(cm, save_path, title):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar()

    ticks = np.arange(num_classes)
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_set = OAMDataset(
        "data/manifests/test.csv",
        image_size=224,
        train=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    models = {}

    print("Loading ResNet50...")
    resnet = tv_models.resnet50(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    resnet = load_state(resnet, "checkpoints/resnet50/best.pt")
    models["ResNet-50"] = resnet

    print("Loading Swin-T...")
    swin = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        num_classes=num_classes,
    )
    swin = load_state(swin, "best_swin.pth")
    models["Swin-T"] = swin

    print("Loading SAM3 baseline...")
    sam_backbone = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=0,
    )
    sam_classifier = nn.Linear(sam_backbone.embed_dim, num_classes)
    sam3 = nn.Sequential(sam_backbone, sam_classifier)
    sam3 = load_state(sam3, "best_sam3_baseline.pth")
    models["SAM3 baseline"] = sam3

    print("Loading SE-SAM3...")
    se_sam3 = SESAM3PPI_Eval(num_classes=num_classes)
    se_sam3 = load_state(se_sam3, "checkpoints/sesam3/best.pt")
    models["SE-SAM3 (Ours)"] = se_sam3

    results = []

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        metric = evaluate_model(model, test_loader)

        results.append({
            "model": name,
            "accuracy": metric["accuracy"],
            "accuracy_percent": metric["accuracy"] * 100,
            "precision_macro": metric["precision_macro"],
            "recall_macro": metric["recall_macro"],
            "f1_macro": metric["f1_macro"],
        })

        cm_path = output_dir / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}_confusion_matrix.png"
        save_confusion_matrix(
            metric["confusion_matrix"],
            cm_path,
            f"{name} Confusion Matrix",
        )

    csv_path = output_dir / "table4_1_model_comparison.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "accuracy_percent",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n==============================")
    print("TABLE 4-1 FINAL RESULTS")
    print("==============================")

    for r in results:
        print(f"{r['model']:<18}: {r['accuracy_percent']:.2f}%")

    print("==============================")
    print(f"结果表已保存: {csv_path}")
    print(f"混淆矩阵已保存到: {output_dir}")


if __name__ == "__main__":
    main()