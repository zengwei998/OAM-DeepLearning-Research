import os
import yaml
import random
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_yaml("configs/simulation.yaml")

    seed = int(cfg["seed"])
    random.seed(seed)

    all_csv = Path(cfg["output"]["manifest_dir"]) / "all.csv"

    if not all_csv.exists():
        raise FileNotFoundError("找不到 all.csv，请先运行 01_generate_dataset.py")

    df = pd.read_csv(all_csv)

    train_ratio = float(cfg["data"]["train_ratio"])
    val_ratio = float(cfg["data"]["val_ratio"])

    # 关键：保证论文一致性（按物理条件分层随机）
    grouped = df.groupby([
        "label",
        "cn2",
        "propagation_distance_m",
        "snr_db"
    ])

    train_list = []
    val_list = []
    test_list = []

    for _, group in grouped:
        group = group.sample(frac=1, random_state=seed).reset_index(drop=True)

        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_list.append(group[:n_train])
        val_list.append(group[n_train:n_train + n_val])
        test_list.append(group[n_train + n_val:])

    train_df = pd.concat(train_list).sample(frac=1, random_state=seed)
    val_df = pd.concat(val_list).sample(frac=1, random_state=seed)
    test_df = pd.concat(test_list).sample(frac=1, random_state=seed)

    out_dir = Path(cfg["output"]["manifest_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(out_dir / "val.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(out_dir / "test.csv", index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("数据集划分完成")
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")
    print("=" * 60)


if __name__ == "__main__":
    main()