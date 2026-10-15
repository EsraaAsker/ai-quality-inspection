import os
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
from .model import PatchCore

st.set_page_config(page_title="AI Quality Inspection", layout="wide")
st.title("AI Quality Inspection — metal_nut")
st.caption("PatchCore-style nearest-neighbour anomaly detection with a pretrained ResNet18 backbone.")
path = os.getenv("AQI_MEMORY_BANK", "artifacts/metal_nut/memory_bank.pt")
if not Path(path).exists(): st.error(f"Model artifact not found: {path}. Run the CLI first."); st.stop()
@st.cache_resource
def load():
    m = PatchCore(); m.load_memory_bank(path); return m
model = load(); uploaded = st.file_uploader("Upload a metal nut image", type=["png", "jpg", "jpeg"])
if uploaded:
    image = np.asarray(Image.open(uploaded).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)); pred = model.predict(image)
    c1, c2, c3 = st.columns(3); c1.image(image, caption="Input"); c2.image(pred.heatmap, caption=f"Heatmap — score {pred.score:.3f}", clamp=True)
    overlay = image.copy(); heat = np.asarray(Image.fromarray((pred.heatmap * 255).astype("uint8")).resize((224,224)))
    overlay = (0.65 * overlay + 0.35 * np.stack([heat, np.zeros_like(heat), 255-heat], axis=-1)).astype("uint8")
    c3.image(overlay, caption="Overlay"); st.metric("Anomaly score", f"{pred.score:.4f}")
