from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Sample:
    image: Path
    label: int
    mask: Optional[Path]
    defect_type: str


def validate_mvtec(root: str | Path, category: str = "metal_nut") -> dict:
    root = Path(root)
    cat = root / category
    required = [cat / "train" / "good", cat / "test", cat / "ground_truth"]
    missing = [str(p) for p in required if not p.exists()]
    train_good = sorted((cat / "train" / "good").glob("*.png")) if (cat / "train" / "good").exists() else []
    test_images = list((cat / "test").glob("*/*.png")) if (cat / "test").exists() else []
    return {"root": str(root), "category": category, "valid": not missing and bool(train_good) and bool(test_images),
            "missing": missing, "train_good": len(train_good), "test_images": len(test_images)}


def collect_samples(root: str | Path, category: str = "metal_nut") -> tuple[list[Sample], list[Sample]]:
    cat = Path(root) / category
    train = [Sample(p, 0, None, "good") for p in sorted((cat / "train" / "good").glob("*.png"))]
    test = []
    for defect_dir in sorted((cat / "test").iterdir()):
        if not defect_dir.is_dir():
            continue
        for image in sorted(defect_dir.glob("*.png")):
            mask = None
            if defect_dir.name != "good":
                mask = cat / "ground_truth" / defect_dir.name / f"{image.stem}_mask.png"
            test.append(Sample(image, int(defect_dir.name != "good"), mask, defect_dir.name))
    return train, test


def load_image(path: str | Path, size: int = 224) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR))
