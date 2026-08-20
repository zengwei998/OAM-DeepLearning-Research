import os
import json
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import models

from src.oam_dataset import OAMDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="Train", ncols=120)
    for x, y, _ in pbar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += batch_size

        pbar.set_postfix(
            loss=f"{total_loss / total:.4f}",
            acc=f"{correct / total * 100:.2f}%"
        )

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="Val", ncols=120)
    for x, y, _ in pbar:
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        loss = criterion(out, y)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += batch_size

        pbar.set_postfix(
            loss=f"{total_loss / total:.4f}",
            acc=f"{correct / total * 100:.2f}%"
        )

    return total_loss / total, correct / total


def main():
    cfg = load_yaml("configs/training.yaml")

    torch.manual_seed(int(cfg["seed"]))
    torch.cuda.manual_seed_all(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"]["use_cuda"] else "cpu")

    print("=" * 80)
    print("ResNet-50 Training Start")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Batch size: {cfg['train']['batch_size']}")
    print(f"Epochs: {cfg['train']['epochs']}")
    print("=" * 80)

    train_dataset = OAMDataset(
        cfg["paths"]["train_csv"],
        image_size=int(cfg["model"]["image_size"]),
        train=True,
    )

    val_dataset = OAMDataset(
        cfg["paths"]["val_csv"],
        image_size=int(cfg["model"]["image_size"]),
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=bool(cfg["train"]["pin_memory"]),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=bool(cfg["train"]["pin_memory"]),
    )

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, int(cfg["model"]["num_classes"]))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    checkpoint_dir = Path(cfg["paths"]["checkpoints_dir"]) / "resnet50"
    metrics_dir = Path(cfg["paths"]["metrics_dir"]) / "resnet50"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        print(f"\nEpoch [{epoch}/{cfg['train']['epochs']}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print("-" * 80)
        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_acc_percent": train_acc * 100,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_acc_percent": val_acc * 100,
        }
        history.append(record)

        torch.save(
            {
                "model": "resnet50",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "history": history,
            },
            checkpoint_dir / "last.pt",
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            torch.save(
                {
                    "model": "resnet50",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "history": history,
                },
                checkpoint_dir / "best.pt",
            )

            print(f"New Best Val Acc: {best_val_acc * 100:.2f}%")

        with open(metrics_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)

        with open(metrics_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": "resnet50",
                    "best_epoch": best_epoch,
                    "best_val_acc": best_val_acc,
                    "best_val_acc_percent": best_val_acc * 100,
                    "epochs_finished": epoch,
                },
                f,
                indent=4,
                ensure_ascii=False,
            )

        print(f"Best Val Acc So Far: {best_val_acc * 100:.2f}%")
        print("-" * 80)

    print("\nTraining Finished")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Acc: {best_val_acc * 100:.2f}%")


if __name__ == "__main__":
    main()