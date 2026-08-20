import os
from pathlib import Path

import torch
import torch.nn as nn
import timm
from tqdm import tqdm

from torch.utils.data import DataLoader
from src.oam_dataset import OAMDataset


# =========================
# 固定路径
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

print("Current Dir:", os.getcwd())


# =========================
# device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# dataset（统一CSV）
# =========================
train_csv = "data/manifests/train.csv"
val_csv   = "data/manifests/val.csv"

train_set = OAMDataset(train_csv, image_size=224, train=True)
val_set   = OAMDataset(val_csv, image_size=224, train=False)

train_loader = DataLoader(train_set, batch_size=50, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_set, batch_size=50, shuffle=False, num_workers=0)


# =========================
# SAM3 baseline（核心）
# =========================
# 用 ViT / ConvNeXt 作为 SAM-like backbone
backbone = timm.create_model(
    "vit_base_patch16_224",
    pretrained=False,   # 保证不下载
    num_classes=0       # 去掉分类头
)

in_features = backbone.embed_dim

classifier = nn.Linear(in_features, 11)

model = nn.Sequential(
    backbone,
    classifier
).to(device)


# =========================
# loss / optimizer
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


# =========================
# train loop
# =========================
best_acc = 0

for epoch in range(50):

    # -------- train --------
    model.train()
    correct, total = 0, 0

    loop = tqdm(train_loader, desc=f"Train {epoch+1}")

    for x, y, _ in loop:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        loop.set_postfix(acc=correct/total)

    train_acc = correct / total


    # -------- val --------
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y, _ in val_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            pred = out.argmax(1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    val_acc = correct / total


    print("\n========================")
    print(f"Epoch {epoch+1}")
    print(f"Train Acc: {train_acc:.4f}")
    print(f"Val   Acc: {val_acc:.4f}")


    # -------- save best --------
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_sam3_baseline.pth")
        print("🔥 Saved Best SAM3 baseline")

print("\nTraining Finished")
print("Best Acc:", best_acc)