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


# -------------------------
# Scoring helpers
# -------------------------
def image_score_from_patch_dist(
    patch_dists: torch.Tensor,
    topk: int = 5,
    topk_ratio: float | None = None,
    min_k: int = 5,
) -> float:
    """
    Aggregate patch distances -> 1 image anomaly score.

    If topk_ratio is provided: use adaptive top-k = max(min_k, int(P * topk_ratio)).
    Else: use fixed topk.
    """
    P = patch_dists.numel()
    if P == 0:
        return 0.0

    if topk_ratio is not None:
        k = max(min_k, int(P * float(topk_ratio)))
    else:
        k = int(topk)

    k = min(max(k, 1), P)
    return patch_dists.topk(k).values.mean().item()


def extract_patches_from_feat(f: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
    """
    f: [1, C, H, W] -> patches: [H*W, C]
    Returns patches, C, H, W
    """
    _, c, h, w = f.shape
    patches = f.squeeze(0).permute(1, 2, 0).reshape(-1, c)  # [P, C]
    return patches, c, h, w


def compute_patch_dists(
    x: torch.Tensor,
    model: torch.nn.Module,
    memory: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, int, int]]:
    """
    Cosine distance = 1 - max_cos_sim

    Returns:
      dists_all:  [P] distances for scoring
      heat_info: (dists_heat, H, W) for heatmap reshape/upsample
    """
    feats = model(x)
    if len(feats) == 0:
        raise RuntimeError("No features extracted. Check cfg.feature_layers / feature extractor.")

    # NOTE:
    # Your memory bank is saved as [M, C]. That implies single-layer feature with channel C.
    # If multiple layers are enabled and C differs -> cannot match the same memory.
    # So we pick the FIRST feature map and enforce C match with memory.
    first_key = list(feats.keys())[0]
    f = feats[first_key]  # [1, C, H, W]

    patches, c, h, w = extract_patches_from_feat(f)
    if memory.shape[1] != c:
        raise ValueError(
            f"Memory bank feature dim mismatch: memory C={memory.shape[1]} but extracted C={c}. "
            f"Likely cfg.feature_layers differs between build_memorybank and inference. "
            f"Fix by using the same single layer (recommended: ('layer3',))."
        )

    patches = F.normalize(patches, dim=1)
    sim = patches @ memory.T            # [P, M]
    max_sim, _ = sim.max(dim=1)         # [P]
    dists = 1.0 - max_sim               # [P]

    # For heatmap, we use the same layer (reshapeable)
    return dists, (dists, h, w)


# -------------------------
# Visualization
# -------------------------
def make_heatmap_overlay(
    img_path: Path,
    anomaly_map_small: torch.Tensor,   # [Hf, Wf]
    out_heatmap_path: Path,
    out_overlay_path: Path,
    image_size: int,
    alpha: float = 0.45,
    blur_ksize: int = 7,
):
    """
    Create:
      - heatmap colored image
      - overlay heatmap on original image (resized to image_size)
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return

    img_bgr = cv2.resize(img_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)

    amap = anomaly_map_small.detach().float()
    if amap.ndim != 2:
        amap = amap.squeeze()

    # Upsample to image size
    amap_t = amap.unsqueeze(0).unsqueeze(0)  # [1,1,Hf,Wf]
    amap_up = F.interpolate(
        amap_t, size=(image_size, image_size), mode="bilinear", align_corners=False
    ).squeeze().cpu().numpy()

    # Normalize 0..1
    amap_up = amap_up - float(amap_up.min())
    denom = float(amap_up.max() - amap_up.min()) + 1e-8
    amap_up = amap_up / denom

    # Optional blur for smoother heatmap
    if blur_ksize and blur_ksize >= 3:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        amap_up = cv2.GaussianBlur(amap_up, (blur_ksize, blur_ksize), 0)

    heat = (amap_up * 255).astype(np.uint8)
    heat_color_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay_bgr = cv2.addWeighted(img_bgr, 1.0 - alpha, heat_color_bgr, alpha, 0)

    out_heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    out_overlay_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_heatmap_path), heat_color_bgr)
    cv2.imwrite(str(out_overlay_path), overlay_bgr)


# -------------------------
# Folder inference
# -------------------------
def infer_folder(
    folder: Path,
    class_name: str,
    label: int,
    device: str,
    model: torch.nn.Module,
    memory: torch.Tensor,
    image_size: int,
    threshold: float | None,
    out_heatmaps_dir: Path,
    out_figures_dir: Path,
    save_heatmaps: bool,
    topk: int,
    topk_ratio: float | None,
    min_k: int,
) -> list[list]:
    """
    Returns rows:
      [image, class, label, anomaly_score, threshold, pred]
    """
    ds = ImageFolderDataset(folder)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    rows = []
    for x, fname in tqdm(loader, desc=f"Infer {class_name}", total=len(ds)):
        x = x.to(device)

        dists, (dists_h, h, w) = compute_patch_dists(x, model, memory)

        score = image_score_from_patch_dist(
            dists, topk=topk, topk_ratio=topk_ratio, min_k=min_k
        )

        # ตัดสิน PASS / FAIL ก่อน
        pred = ""
        thr_val = ""
        if threshold is not None:
            pred = "FAIL" if score > threshold else "PASS"
            thr_val = threshold

        # save heatmap เฉพาะ FAIL เท่านั้น
        if save_heatmaps and pred == "FAIL":
            anomaly_map_small = dists_h.view(h, w)
            img_path = folder / fname[0]
            stem = fname[0].replace(".png", "")
            out_hm = out_heatmaps_dir / f"{class_name}__{stem}_heatmap.png"
            out_ov = out_figures_dir / f"{class_name}__{stem}_overlay.png"
            make_heatmap_overlay(
                img_path=img_path,
                anomaly_map_small=anomaly_map_small,
                out_heatmap_path=out_hm,
                out_overlay_path=out_ov,
                image_size=image_size,
                alpha=float(getattr(cfg, "heatmap_alpha", 0.45)),
                blur_ksize=int(getattr(cfg, "heatmap_blur_ksize", 7)),
            )
        rows.append([fname[0], class_name, label, float(score), thr_val, pred])

    return rows


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.ensure_dirs()

    # Config fallbacks
    image_size = int(getattr(cfg, "image_size", 224))

    # Top-k settings
    topk = int(getattr(cfg, "topk", 5))
    topk_ratio = getattr(cfg, "topk_ratio", None)
    if topk_ratio is not None:
        topk_ratio = float(topk_ratio)
    min_k = int(getattr(cfg, "min_k", 5))

    # Threshold settings
    thr_percentile = float(getattr(cfg, "threshold_percentile", 99.5))

    # Heatmap settings
    save_heatmaps = bool(getattr(cfg, "save_heatmaps", True))

    # Output paths
    out_csv = cfg.output_root / "predictions" / "scores.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_heatmaps_dir = cfg.output_root / "heatmaps"
    out_figures_dir = cfg.output_root / "figures"
    out_heatmaps_dir.mkdir(parents=True, exist_ok=True)
    out_figures_dir.mkdir(parents=True, exist_ok=True)

    # Load memory bank
    mem_path = cfg.output_root / "models" / "patchcore_memory.pt"
    memory = torch.load(mem_path, map_location="cpu")  # [M, C]
    memory = memory.to(device)
    memory = F.normalize(memory, dim=1)

    # Feature extractor
    model = ResNetFeatureExtractor(
        backbone=cfg.backbone,
        layers=cfg.feature_layers
    ).to(device)
    model.eval()

    with torch.no_grad():
        # 1) Build threshold from train/good
        train_good_dir = cfg.dataset_dir() / "train" / "good"
        train_rows = infer_folder(
            folder=train_good_dir,
            class_name="train_good",
            label=0,
            device=device,
            model=model,
            memory=memory,
            image_size=image_size,
            threshold=None,  # no pred for train
            out_heatmaps_dir=out_heatmaps_dir,
            out_figures_dir=out_figures_dir,
            save_heatmaps=False,  # ไม่จำเป็นสำหรับ train/good
            topk=topk,
            topk_ratio=topk_ratio,
            min_k=min_k,
        )

        train_scores = np.array([r[3] for r in train_rows], dtype=np.float32)
        threshold = float(np.percentile(train_scores, thr_percentile))

        print(f"\nThreshold = percentile({thr_percentile}) of train/good scores => {threshold:.6f}\n")

        # 2) Infer test/*
        test_dir = cfg.dataset_dir() / "test"
        classes = [p for p in test_dir.iterdir() if p.is_dir()]
        classes = sorted(classes, key=lambda x: x.name)

        all_rows = []
        for cls in classes:
            label = 0 if cls.name == "good" else 1
            rows = infer_folder(
                folder=cls,
                class_name=cls.name,
                label=label,
                device=device,
                model=model,
                memory=memory,
                image_size=image_size,
                threshold=threshold,
                out_heatmaps_dir=out_heatmaps_dir,
                out_figures_dir=out_figures_dir,
                save_heatmaps=save_heatmaps,
                topk=topk,
                topk_ratio=topk_ratio,
                min_k=min_k,
            )
            all_rows.extend(rows)

    # Save CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "class", "label", "anomaly_score", "threshold", "pred"])
        w.writerows(all_rows)

    print("Saved CSV:", out_csv)
    if save_heatmaps:
        print("Saved heatmaps:", out_heatmaps_dir)
        print("Saved overlays:", out_figures_dir)
    print("Done ✅")


if __name__ == "__main__":
    torch.manual_seed(getattr(cfg, "seed", 0))
    main()