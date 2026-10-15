from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


@dataclass
class Prediction:
    score: float
    heatmap: np.ndarray


class PatchCore:
    """Small, CPU-friendly PatchCore implementation using ResNet18 layer2/layer3 features."""
    def __init__(self, device: str | None = None, image_size: int = 224, coreset_ratio: float = 0.1):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.image_size, self.coreset_ratio = image_size, coreset_ratio
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT).to(self.device).eval()
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-2]).to(self.device).eval()
        self.memory_bank: torch.Tensor | None = None
        self.threshold: float | None = None
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _features(self, images: list[np.ndarray]) -> tuple[torch.Tensor, tuple[int, int]]:
        x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().to(self.device) / 255.0
        x = (x - self.mean) / self.std
        with torch.inference_mode():
            # layer2 and layer3 spatial activations are concatenated after matching resolution.
            x1 = self.feature_extractor[:6](x)  # through layer2
            x2 = self.feature_extractor[6](x1)   # continue through layer3
            x1 = torch.nn.functional.interpolate(x1, size=x2.shape[-2:], mode="bilinear", align_corners=False)
            feat = torch.cat([x1, x2], dim=1).permute(0, 2, 3, 1)
            feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat.reshape(-1, feat.shape[-1]), (feat.shape[1], feat.shape[2])

    def fit(self, images: list[np.ndarray]) -> None:
        feats, _ = self._features(images)
        n = max(1, int(len(feats) * self.coreset_ratio))
        if n < len(feats):
            idx = torch.linspace(0, len(feats) - 1, n, device=feats.device).long()
            feats = feats[idx]
        self.memory_bank = feats.detach()

    def predict(self, image: np.ndarray) -> Prediction:
        if self.memory_bank is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        feats, shape = self._features([image])
        distances = torch.cdist(feats, self.memory_bank).min(dim=1).values
        heat = distances.reshape(shape).detach().cpu().numpy()
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        # image score is the max patch distance, robustly scaled by the mean of top patches.
        k = max(1, int(0.1 * distances.numel()))
        score = float(torch.topk(distances, k).values.mean().cpu())
        return Prediction(score, heat)

    def save(self, path: str | Path) -> None:
        if self.memory_bank is None: raise RuntimeError("Cannot save an unfitted model")
        torch.save({"memory_bank": self.memory_bank.cpu(), "image_size": self.image_size, "coreset_ratio": self.coreset_ratio}, path)

    def load_memory_bank(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.memory_bank = state["memory_bank"].to(self.device)
