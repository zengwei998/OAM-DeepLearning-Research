import os
import json
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.oam_dataset import OAMDataset
from src.models_se_sam3 import SESAM3PPI


# =========================
# 路径统一
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =========================
# Train
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    scaler = torch.cuda.amp.GradScaler()

    pbar = tqdm(loader, desc="Train", ncols=120)

    for x, y, _ in pbar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * y.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        pbar.set_postfix(
            loss=total_loss / total,
            acc=correct / total
        )

    return total_loss / total, correct / total


# =========================
# Val
# =========================
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Val", ncols=120)

    for x, y, _ in pbar:
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        loss = criterion(out, y)

        total_loss += loss.item() * y.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        pbar.set_postfix(
            loss=total_loss / total,
            acc=correct / total
        )

    return total_loss / total, correct / total


# =========================
# Main
# =========================
def main():

    cfg = load_yaml("configs/training.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Training SE-SAM3-PPI")
    print("=" * 80)
    print("Device:", device)
    print("=" * 80)

    # =========================
    # Dataset
    # =========================
    train_set = OAMDataset(
        cfg["paths"]["train_csv"],
        image_size=cfg["model"]["image_size"],
        train=True
    )

    val_set = OAMDataset(
        cfg["paths"]["val_csv"],
        image_size=cfg["model"]["image_size"],
        train=False
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True
    )

    # =========================
    # Model（核心）
    # =========================
    model = SESAM3PPI(num_classes=cfg["model"]["num_classes"]).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"]
    )

    # =========================
    # Save dir
    # =========================
    ckpt_dir = Path("checkpoints/sesam3")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    metric_dir = Path("outputs/metrics/sesam3")
    metric_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0
    history = []

    # =========================
    # Train loop
    # =========================
    for epoch in range(cfg["train"]["epochs"]):

        print(f"\nEpoch [{epoch+1}/{cfg['train']['epochs']}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print("-" * 80)
        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }

        history.append(record)

        # =========================
        # Save last
        # =========================
        torch.save(model.state_dict(), ckpt_dir / "last.pt")

        # =========================
        # Save best
        # =========================
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt_dir / "best.pt")

            print(f"🔥 New Best Acc: {best_acc:.4f}")

        # =========================
        # Save logs
        # =========================
        with open(metric_dir / "history.json", "w") as f:
            json.dump(history, f, indent=4)

    print("\nTraining Finished")
    print("Best Acc:", best_acc)


if __name__ == "__main__":
    main()