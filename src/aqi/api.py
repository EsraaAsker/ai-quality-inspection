import os
from pathlib import Path
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from .model import PatchCore

app = FastAPI(title="AI Quality Inspection", version="0.1.0")
_model = None


def get_model():
    global _model
    if _model is None:
        artifact = os.getenv("AQI_MEMORY_BANK", "artifacts/metal_nut/memory_bank.pt")
        if not Path(artifact).exists(): raise HTTPException(503, "Model artifact not found; run the evaluation CLI first.")
        _model = PatchCore(); _model.load_memory_bank(artifact)
    return _model


@app.get("/health")
def health(): return {"status": "ok", "model_artifact": os.getenv("AQI_MEMORY_BANK", "artifacts/metal_nut/memory_bank.pt")}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}: raise HTTPException(415, "Upload a PNG or JPEG image")
    try: image = np.asarray(Image.open(file.file).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))
    except Exception as exc: raise HTTPException(400, f"Invalid image: {exc}")
    pred = get_model().predict(image)
    return {"anomaly_score": pred.score, "heatmap": pred.heatmap.tolist(), "model": "patchcore-resnet18"}
