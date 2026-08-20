import os
from pathlib import Path

import torch
import torch.nn as nn
import timm
from tqdm import tqdm

from torch.utils.data import DataLoader
from src.oam_dataset import OAMDataset


# =========================
# 固定工程路径
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

print("当前工作目录：", os.getcwd())


# =========================
# device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# dataset（CSV模式）
# =========================
train_csv = "data/manifests/train.csv"
val_csv   = "data/manifests/val.csv"

train_set = OAMDataset(train_csv, image_size=224, train=True)
val_set   = OAMDataset(val_csv, image_size=224, train=False)

train_loader = DataLoader(train_set, batch_size=50, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_set, batch_size=50, shuffle=False, num_workers=0)


# =========================
# Swin-T（关键修复：关闭预训练，避免网络错误）
# =========================
model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=False,   # ❗关键：避免huggingface下载失败
    num_classes=11
)

model = model.to(device)


# =========================
# loss & optimizer
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


# =========================
# training loop
# =========================
best_acc = 0

for epoch in range(50):

    # ---------------- TRAIN ----------------
    model.train()
    correct, total = 0, 0

    loop = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")

    for x, y, _ in loop:
        x = x.to(device)
        y = y.to(device)

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


    # ---------------- VAL ----------------
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y, _ in val_loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            pred = out.argmax(1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    val_acc = correct / total


    print("\n==========================")
    print(f"Epoch {epoch+1}")
    print(f"Train Acc: {train_acc:.4f}")
    print(f"Val   Acc: {val_acc:.4f}")


    # ---------------- SAVE BEST ----------------
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_swin.pth")
        print("🔥 Saved Best Swin Model")

print("\nTraining Finished")
print("Best Acc:", best_acc)