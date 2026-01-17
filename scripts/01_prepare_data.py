# scripts/01_prepare_data.py
from pathlib import Path
import random

import cv2

from src.config import cfg


def count_images(folder: Path) -> int:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in exts)


def sample_image(folder: Path) -> Path | None:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    imgs = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    return random.choice(imgs) if imgs else None


def save_preview(img_path: Path, out_path: Path) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def main():
    ds = cfg.dataset_dir()
    print("Dataset dir:", ds)
    print("Exists?:", ds.exists())

    # List what is inside bottle/
    if ds.exists():
        print("\n[Contents of category dir]")
        for p in sorted(ds.iterdir()):
            print(" -", p.name)
    else:
        raise FileNotFoundError(f"Category directory not found: {ds}")

    test_dir = ds / "test"
    train_dir = ds / "train"

    print("\ntrain dir:", train_dir, "exists?", train_dir.exists())
    print("test  dir:", test_dir,  "exists?", test_dir.exists())

    if not test_dir.exists():
        raise FileNotFoundError(
            f"Missing test directory: {test_dir}\n"
            f"Check that you extracted the dataset to: {cfg.data_root}"
        )

    train_good = train_dir / "good"
    test_good = test_dir / "good"
    test_defects = [p for p in test_dir.iterdir() if p.is_dir() and p.name != "good"]

    print("\n[Counts]")
    print("train/good:", count_images(train_good))
    print("test/good :", count_images(test_good))
    for d in sorted(test_defects, key=lambda x: x.name):
        print(f"test/{d.name:16}:", count_images(d))

    # Save previews
    out_dir = cfg.output_root / "figures" / "data_preview"
    cfg.ensure_dirs()
    print("\n[Saving previews to]", out_dir)

    p = sample_image(train_good)
    if p:
        save_preview(p, out_dir / "train_good.png")
        print("Saved:", out_dir / "train_good.png")

    p = sample_image(test_good)
    if p:
        save_preview(p, out_dir / "test_good.png")
        print("Saved:", out_dir / "test_good.png")

    for d in sorted(test_defects, key=lambda x: x.name):
        p = sample_image(d)
        if p:
            save_preview(p, out_dir / f"test_{d.name}.png")
            print("Saved:", out_dir / f"test_{d.name}.png")

    print("\nDone ✅")

if __name__ == "__main__":
    random.seed(cfg.seed)
    main()