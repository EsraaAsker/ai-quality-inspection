from pathlib import Path
import json
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, f1_score
import matplotlib.pyplot as plt
from .dataset import Sample, load_image
from .model import PatchCore


def select_normal_threshold(model: PatchCore, validation_samples: list[Sample], quantile: float = 0.995) -> float:
    """Select a threshold from defect-free validation images only."""
    if not validation_samples or any(s.label != 0 for s in validation_samples):
        raise ValueError("Threshold validation set must contain at least one normal image and no anomalies")
    scores = [model.predict(load_image(s.image, model.image_size)).score for s in validation_samples]
    return float(np.quantile(scores, quantile))


def evaluate(model: PatchCore, samples: list[Sample], output_dir: str | Path,
             threshold: float | None = None, threshold_source: str = "provided_validation") -> dict:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    if threshold is None:
        raise ValueError("Pass a threshold selected from validation data; test scores must not select it")
    y_true, y_score, pixel_true, pixel_score = [], [], [], []
    for i, sample in enumerate(samples):
        image = load_image(sample.image, model.image_size)
        pred = model.predict(image)
        y_true.append(sample.label); y_score.append(pred.score)
        if sample.label:
            if sample.mask is None or not sample.mask.exists():
                raise FileNotFoundError(f"Missing ground-truth mask for {sample.image}: {sample.mask}")
            mask = np.asarray(Image.open(sample.mask).convert("L").resize((model.image_size, model.image_size), Image.Resampling.NEAREST)) > 0
        else:
            mask = np.zeros((model.image_size, model.image_size), dtype=bool)
        heat = np.asarray(Image.fromarray((pred.heatmap * 255).astype(np.uint8)).resize((model.image_size, model.image_size), Image.Resampling.BILINEAR)) / 255.0
        pixel_true.extend(mask.ravel()); pixel_score.extend(heat.ravel())
        if i < 12:
            fig, ax = plt.subplots(1, 3, figsize=(9, 3)); ax[0].imshow(image); ax[0].set_title(sample.defect_type)
            ax[1].imshow(heat, cmap="magma"); ax[1].set_title(f"score={pred.score:.3f}")
            ax[2].imshow(image); ax[2].imshow(heat, cmap="jet", alpha=.45); ax[2].set_title("overlay")
            for a in ax: a.axis("off")
            fig.tight_layout(); fig.savefig(output_dir / f"sample_{i:03d}.png", dpi=120); plt.close(fig)
    y_pred = (np.array(y_score) >= threshold).astype(int)
    result = {"n_images": len(samples), "n_anomalies": int(sum(y_true)),
              "image_auroc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else None,
              "image_average_precision": float(average_precision_score(y_true, y_score)) if len(set(y_true)) > 1 else None,
              "image_f1_at_validation_threshold": float(f1_score(y_true, y_pred, zero_division=0)),
              "image_threshold": float(threshold), "threshold_source": threshold_source,
              "pixel_auroc": float(roc_auc_score(pixel_true, pixel_score)) if len(set(pixel_true)) > 1 else None,
              "pixel_average_precision": float(average_precision_score(pixel_true, pixel_score)) if len(set(pixel_true)) > 1 else None,
              "confusion_matrix_at_validation_threshold": confusion_matrix(y_true, y_pred).tolist()}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result
