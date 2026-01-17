# src/dataset.py
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import torch

from src.config import cfg


class ImageFolderDataset(Dataset):
    def __init__(self, folder: Path):
        self.images = sorted(
            [p for p in folder.glob("*.png")]
        )
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.transform(img)
        return x, img_path.name