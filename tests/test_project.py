from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from aqi.dataset import validate_mvtec
from aqi.model import PatchCore


def test_invalid_dataset_is_reported(tmp_path):
    report = validate_mvtec(tmp_path)
    assert report["valid"] is False
    assert report["missing"]


def test_model_requires_fit():
    model = PatchCore(device="cpu")
    with pytest.raises(RuntimeError): model.predict(np.zeros((224, 224, 3), dtype=np.uint8))


def test_model_fit_and_predict():
    model = PatchCore(device="cpu", coreset_ratio=0.2)
    model.fit([np.zeros((224, 224, 3), dtype=np.uint8), np.ones((224, 224, 3), dtype=np.uint8) * 255])
    result = model.predict(np.zeros((224, 224, 3), dtype=np.uint8))
    assert result.heatmap.shape == (14, 14)
    assert np.isfinite(result.score)
