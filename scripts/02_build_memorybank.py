# scripts/02_build_memorybank.py
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from src.config import cfg
from src.dataset import ImageFolderDataset
from src.feature_extractor import ResNetFeatureExtractor


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.ensure_dirs()

    # Dataset: train/good
    train_dir = cfg.dataset_dir() / "train" / "good"
    dataset = ImageFolderDataset(train_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"Train images: {len(dataset)}")

    # Feature extractor
    model = ResNetFeatureExtractor(
        backbone=cfg.backbone,
        layers=cfg.feature_layers
    ).to(device)
    model.eval()

    memory = []

    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Extracting features"):
            x = x.to(device)
            feats = model(x)

            # รวม patch จากทุก layer
            for f in feats.values():
                # f: [1, C, H, W] → [H*W, C]
                patches = f.squeeze(0).permute(1, 2, 0).reshape(-1, f.shape[1])
                memory.append(patches.cpu())

    memory = torch.cat(memory, dim=0)
    print("Total patches:", memory.shape[0])

    # Subsample memory bank
    if memory.shape[0] > cfg.memory_bank_size:
        idx = torch.randperm(memory.shape[0])[: cfg.memory_bank_size]
        memory = memory[idx]

    print("Memory bank size:", memory.shape)

    out_path = cfg.output_root / "models" / "patchcore_memory.pt"
    torch.save(memory, out_path)
    print("Saved memory bank to:", out_path)


if __name__ == "__main__":
    torch.manual_seed(cfg.seed)
    main()