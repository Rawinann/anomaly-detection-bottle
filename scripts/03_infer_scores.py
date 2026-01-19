# scripts/03_infer_scores.py
from pathlib import Path
import csv
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import cfg
from src.dataset import ImageFolderDataset
from src.feature_extractor import ResNetFeatureExtractor


def image_score_from_patch_dist(patch_dists: torch.Tensor, topk: int = 5) -> float:
    """Aggregate patch distances -> 1 image anomaly score (top-k mean)."""
    k = min(topk, patch_dists.numel())
    return patch_dists.topk(k).values.mean().item()


def compute_patch_dists(
    x: torch.Tensor,
    model: torch.nn.Module,
    memory: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Returns:
      dists: [P] patch anomaly distances (cosine distance = 1 - max_sim)
      hw: (H, W) feature map spatial size for reshaping to anomaly map
    """
    feats = model(x)
    # NOTE: in your current infer, you use the first layer only.
    # This keeps behavior consistent with your current pipeline.
    f = list(feats.values())[0]  # [1, C, H, W]
    _, c, h, w = f.shape

    patches = f.squeeze(0).permute(1, 2, 0).reshape(-1, c)  # [P, C]
    patches = F.normalize(patches, dim=1)

    # cosine similarity -> distance
    sim = patches @ memory.T         # [P, M]
    max_sim, _ = sim.max(dim=1)      # [P]
    dists = 1.0 - max_sim            # [P]
    return dists, (h, w)


def make_heatmap_overlay(
    img_path: Path,
    anomaly_map_small: torch.Tensor,  # [Hf, Wf] on device or cpu
    out_heatmap_path: Path,
    out_overlay_path: Path,
    image_size: int,
    alpha: float = 0.45,
):
    """
    Create:
      - heatmap image
      - overlay heatmap on original image (resized to image_size)
    """
    # Load original (BGR), convert to RGB for blend in RGB then save back BGR
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return

    img_bgr = cv2.resize(img_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)

    # Upsample anomaly map to image size
    amap = anomaly_map_small
    if isinstance(amap, torch.Tensor):
        amap = amap.detach().float()
        if amap.ndim != 2:
            amap = amap.squeeze()
        # to cpu numpy
        amap_t = amap.unsqueeze(0).unsqueeze(0)  # [1,1,Hf,Wf]
        amap_up = F.interpolate(amap_t, size=(image_size, image_size), mode="bilinear", align_corners=False)
        amap_up = amap_up.squeeze().cpu().numpy()
    else:
        amap_up = amap

    # Normalize 0..1
    amap_up = amap_up - float(amap_up.min())
    denom = float(amap_up.max() - amap_up.min()) + 1e-8
    amap_up = amap_up / denom

    # Heatmap color
    heat = (amap_up * 255).astype(np.uint8)
    heat_color_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    # Overlay
    overlay_bgr = cv2.addWeighted(img_bgr, 1.0 - alpha, heat_color_bgr, alpha, 0)

    out_heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    out_overlay_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_heatmap_path), heat_color_bgr)
    cv2.imwrite(str(out_overlay_path), overlay_bgr)


def collect_scores_in_folder(
    folder: Path,
    label: int,
    class_name: str,
    device: str,
    model: torch.nn.Module,
    memory: torch.Tensor,
    topk: int,
    save_heatmaps: bool,
    heatmaps_dir: Path,
    figures_dir: Path,
    image_size: int,
) -> list[list]:
    """
    Iterate images in a folder and return rows for CSV.
    Each row: [image, class, label, anomaly_score, threshold, pred]
    """
    ds = ImageFolderDataset(folder)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    rows = []
    for x, fname in tqdm(loader, desc=f"Infer {class_name}", total=len(ds)):
        x = x.to(device)
        dists, (h, w) = compute_patch_dists(x, model, memory)
        score = image_score_from_patch_dist(dists, topk=topk)

        # Create heatmap/overlay if needed
        if save_heatmaps:
            anomaly_map_small = dists.view(h, w)
            img_path = folder / fname[0]
            out_hm = heatmaps_dir / f"{class_name}__{fname[0].replace('.png','')}_heatmap.png"
            out_ov = figures_dir / f"{class_name}__{fname[0].replace('.png','')}_overlay.png"
            make_heatmap_overlay(
                img_path=img_path,
                anomaly_map_small=anomaly_map_small,
                out_heatmap_path=out_hm,
                out_overlay_path=out_ov,
                image_size=image_size,
            )

        rows.append([fname[0], class_name, label, float(score)])
    return rows


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.ensure_dirs()

    topk = int(getattr(cfg, "topk", 5))
    thr_percentile = float(getattr(cfg, "threshold_percentile", 99.5))
    save_heatmaps = bool(getattr(cfg, "save_heatmaps", True))

    # Output dirs
    out_csv = cfg.output_root / "predictions" / "scores.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    heatmaps_dir = cfg.output_root / "heatmaps"
    figures_dir = cfg.output_root / "figures"
    heatmaps_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load memory bank
    mem_path = cfg.output_root / "models" / "patchcore_memory.pt"
    memory = torch.load(mem_path, map_location="cpu")  # [M, C]
    memory = memory.to(device)
    memory = F.normalize(memory, dim=1)  # stabilize cosine distance

    # Feature extractor
    model = ResNetFeatureExtractor(
        backbone=cfg.backbone,
        layers=cfg.feature_layers
    ).to(device)
    model.eval()

    # 1) Build threshold from train/good scores
    train_good_dir = cfg.dataset_dir() / "train" / "good"
    train_rows = collect_scores_in_folder(
        folder=train_good_dir,
        label=0,
        class_name="train_good",
        device=device,
        model=model,
        memory=memory,
        topk=topk,
        save_heatmaps=False,          # ไม่จำเป็นสำหรับ train/good
        heatmaps_dir=heatmaps_dir,
        figures_dir=figures_dir,
        image_size=cfg.image_size,
    )
    train_scores = np.array([r[3] for r in train_rows], dtype=np.float32)
    threshold = float(np.percentile(train_scores, thr_percentile))

    print(f"\nThreshold (percentile {thr_percentile} of train/good): {threshold:.6f}\n")

    # 2) Infer test set + PASS/FAIL + heatmaps
    test_dir = cfg.dataset_dir() / "test"
    classes = [p for p in test_dir.iterdir() if p.is_dir()]
    classes = sorted(classes, key=lambda x: x.name)

    all_rows = []
    with torch.no_grad():
        for cls in classes:
            label = 0 if cls.name == "good" else 1
            rows = collect_scores_in_folder(
                folder=cls,
                label=label,
                class_name=cls.name,
                device=device,
                model=model,
                memory=memory,
                topk=topk,
                save_heatmaps=save_heatmaps,
                heatmaps_dir=heatmaps_dir,
                figures_dir=figures_dir,
                image_size=cfg.image_size,
            )

            # add threshold + pred
            for r in rows:
                score = r[3]
                pred = "FAIL" if score > threshold else "PASS"
                all_rows.append([r[0], r[1], r[2], score, threshold, pred])

    # save csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "class", "label", "anomaly_score", "threshold", "pred"])
        w.writerows(all_rows)

    print("Saved:", out_csv)
    if save_heatmaps:
        print("Heatmaps:", heatmaps_dir)
        print("Overlays:", figures_dir)
    print("Done ✅")


if __name__ == "__main__":
    torch.manual_seed(cfg.seed)
    main()