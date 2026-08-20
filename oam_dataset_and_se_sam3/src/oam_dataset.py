import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class OAMDataset(Dataset):
    def __init__(self, csv_path, image_size=224, train=False):
        self.df = pd.read_csv(csv_path)

        if train:
            self.tf = transforms.Compose([
                transforms.Grayscale(3),
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Grayscale(3),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("L")
        img = self.tf(img)
        label = int(row["label"])
        return img, torch.tensor(label, dtype=torch.long), {}