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
    missing_masks = []
    for image in test_images:
        if image.parent.name != "good":
            mask = cat / "ground_truth" / image.parent.name / f"{image.stem}_mask.png"
            if not mask.exists():
                missing_masks.append(str(mask))
    return {"root": str(root), "category": category,
            "valid": not missing and bool(train_good) and bool(test_images) and not missing_masks,
            "missing": missing, "missing_masks": missing_masks,
            "train_good": len(train_good), "test_images": len(test_images)}


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


def split_train_validation(samples: list[Sample], validation_ratio: float = 0.2) -> tuple[list[Sample], list[Sample]]:
    """Deterministic split of defect-free train images; validation has no test labels."""
    if not 0 < validation_ratio < 1 or len(samples) < 2:
        raise ValueError("validation_ratio must be between 0 and 1 and at least two samples are required")
    n_val = max(1, int(round(len(samples) * validation_ratio)))
    return samples[:-n_val], samples[-n_val:]


def load_image(path: str | Path, size: int = 224) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR))
