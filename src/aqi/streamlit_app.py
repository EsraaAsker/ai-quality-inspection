import os
import sys
from pathlib import Path

# Streamlit Cloud runs this file directly, so the repository's src/ directory
# is not guaranteed to be on Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import streamlit as st
from PIL import Image

from aqi.model import PatchCore

st.set_page_config(page_title="AI Quality Inspection", layout="wide")
st.title("AI Quality Inspection — metal_nut")
st.caption(
    "PatchCore-style nearest-neighbour anomaly detection with a pretrained ResNet18 backbone."
)

_default_memory_bank = PROJECT_ROOT / "artifacts" / "metal_nut" / "memory_bank.pt"
memory_bank_path = Path(os.getenv("AQI_MEMORY_BANK", str(_default_memory_bank)))
if not memory_bank_path.is_absolute():
    memory_bank_path = PROJECT_ROOT / memory_bank_path

if not memory_bank_path.is_file():
    st.error(f"Model artifact not found: {memory_bank_path}. Run the CLI first.")
    st.stop()


@st.cache_resource
def load_model(model_path: str) -> PatchCore:
    model = PatchCore()
    model.load_memory_bank(model_path)
    return model


model = load_model(str(memory_bank_path))
uploaded = st.file_uploader("Upload a metal nut image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB").resize(
        (224, 224), Image.Resampling.BILINEAR
    )
    image_array = np.asarray(image)
    prediction = model.predict(image_array)

    input_col, heatmap_col, overlay_col = st.columns(3)
    input_col.image(image_array, caption="Input")
    heatmap_col.image(
        prediction.heatmap,
        caption=f"Heatmap — score {prediction.score:.3f}",
        clamp=True,
    )

    heat = np.asarray(
        Image.fromarray((prediction.heatmap * 255).astype("uint8")).resize(
            (224, 224)
        )
    )
    colors = np.stack([heat, np.zeros_like(heat), 255 - heat], axis=-1)
    overlay = (0.65 * image_array + 0.35 * colors).astype("uint8")
    overlay_col.image(overlay, caption="Overlay")
    st.metric("Anomaly score", f"{prediction.score:.4f}")
