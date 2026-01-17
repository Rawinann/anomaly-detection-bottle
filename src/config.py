# src/config.py
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # ---- Paths ----
    project_root: Path = Path(__file__).resolve().parents[1]
    data_root: Path = project_root / "data" / "mvtec_ad"
    category: str = "bottle"
    output_root: Path = project_root / "outputs"

    # ---- Image preprocessing ----
    image_size: int = 256  # MVTec ใช้ภาพขนาดคงที่ได้ดี
    center_crop: bool = False  # ถ้าอยาก crop ให้ True

    # ---- PatchCore backbone / feature layers ----
    backbone: str = "resnet18"  # เริ่มจาก resnet18 เบาและพอ
    # โดยทั่วไป PatchCore จะดึง feature จาก layer กลางๆ 1-2 จุด
    # เดี๋ยวเราจะ map ชื่อ layer จริงใน feature_extractor.py อีกที
    feature_layers: tuple = ("layer3",)

    # ---- Memory bank / sampling ----
    # จำนวน patch feature มักเยอะมาก ต้อง subsample
    # ยิ่งตัวเลขน้อย ยิ่งเร็ว แต่ความละเอียดอาจลดลง
    memory_bank_size: int = 10000

    # ---- Inference / scoring ----
    # threshold สำหรับ PASS/FAIL (เราจะหา threshold จาก val หรือจาก test-good ภายหลัง)
    threshold: float | None = None

    # ---- Reproducibility ----
    seed: int = 42

    def dataset_dir(self) -> Path:
        """Return path to category folder, e.g. data/mvtec_ad/bottle"""
        return self.data_root / self.category

    def ensure_dirs(self) -> None:
        """Create output folders if missing."""
        (self.output_root / "models").mkdir(parents=True, exist_ok=True)
        (self.output_root / "predictions").mkdir(parents=True, exist_ok=True)
        (self.output_root / "heatmaps").mkdir(parents=True, exist_ok=True)
        (self.output_root / "figures").mkdir(parents=True, exist_ok=True)


cfg = Config()