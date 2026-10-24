from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

st.set_page_config(page_title="DigitVision AI | MNIST", page_icon="🧠", layout="wide")

ROOT = Path(__file__).resolve().parent
MODELS = {
    "Fine-tuned MobileNetV2": "mnist_fine_tuned_model.keras",
    "Transfer-learning baseline": "mnist_transfer_model.keras",
}

st.markdown('''
<style>
.stApp {background:#0b1020;color:#f8fafc;}
.block-container {max-width:1150px;padding-top:2rem;}
[data-testid="stSidebar"] {background:#18253c;}
[data-testid="stSidebar"] * {color:#f8fafc !important;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {color:#cbd5e1 !important;}
.mnist-hero {background:linear-gradient(120deg,#1e293b,#3730a3);border:1px solid #6366f1;border-radius:22px;padding:36px;margin-bottom:24px;}
.mnist-hero h1 {color:white !important;font-size:2.7rem;margin:0 0 10px;}
.mnist-hero p {color:#e0e7ff !important;font-size:1.05rem;margin:0;}
[data-testid="stFileUploader"] {border:1px dashed #818cf8;border-radius:16px;padding:12px;}
[data-testid="stMetric"] {background:#192841;border:1px solid #334155;border-radius:12px;padding:16px;}
.stButton>button {border-radius:10px;background:#6366f1;color:white;border:0;font-weight:600;}
.stButton>button:hover {background:#4f46e5;color:white;}
</style>
''', unsafe_allow_html=True)

st.markdown('''<div class="mnist-hero"><h1>🧠 DigitVision AI</h1>
<p>Handwritten Digit Recognition · MobileNetV2 Transfer Learning · MNIST</p></div>''', unsafe_allow_html=True)

with st.sidebar:
    st.header("🔬 Model laboratory")
    selected = st.radio("Choose trained checkpoint", list(MODELS), index=0)
    st.divider()
    st.markdown("**Architecture:** MobileNetV2 + 10-class digit head")
    st.markdown("**Task:** Handwritten digits 0–9")
    st.markdown("**Input:** 32 × 32 RGB; normalised to −1 to 1")
    st.markdown("**Output:** 10 class scores")

@st.cache_resource(show_spinner="Loading saved neural network…")
def load_saved_model(name: str):
    filename = MODELS[name]
    path = ROOT / filename
    if not path.is_file():
        available = ", ".join(sorted(p.name for p in ROOT.iterdir()))
        raise FileNotFoundError(f"Missing {filename} in {ROOT}. Available files: {available}")
    return tf.keras.models.load_model(str(path), compile=False)

try:
    model = load_saved_model(selected)
except Exception as exc:
    st.error("The selected model could not be loaded. Check the saved .keras file and the deployment details below.")
    with st.expander("Technical details"):
        st.exception(exc)
    st.stop()

st.subheader("📤 Upload a handwritten digit")
st.caption("Choose a PNG or JPG containing one handwritten number from 0 to 9.")
uploaded = st.file_uploader("Digit image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded is None:
    st.info("Upload an image to see the model's prediction.")
else:
    try:
        image = ImageOps.exif_transpose(Image.open(uploaded)).convert("L")
        col_image, col_result = st.columns([1, 1], gap="large")
        with col_image:
            st.subheader("🖼️ Uploaded digit")
            st.image(image, caption="Your image", width=300)
        with col_result:
            st.subheader("🤖 Prediction")
            if st.button("Recognise digit", type="primary", use_container_width=True):
                resized = image.resize((32, 32), Image.Resampling.BILINEAR)
                gray = np.asarray(resized, dtype=np.float32)
                rgb = np.stack([gray] * 3, axis=-1)
                batch = np.expand_dims(rgb / 127.5 - 1.0, axis=0)
                raw = np.asarray(model.predict(batch, verbose=0)).reshape(-1)
                if raw.size != 10 or not np.all(np.isfinite(raw)):
                    raise ValueError("Expected 10 finite scores, one for each digit (0–9).")
                # Saved networks may output softmax probabilities or raw logits.
                if np.all(raw >= 0) and np.all(raw <= 1) and np.isclose(raw.sum(), 1.0, atol=0.01):
                    probabilities = raw
                else:
                    exp = np.exp(raw - np.max(raw))
                    probabilities = exp / exp.sum()
                digit = int(np.argmax(probabilities))
                confidence = float(probabilities[digit])
                st.metric("Recognised digit", str(digit))
                st.metric("Model confidence", f"{confidence:.1%}")
                st.progress(confidence)
                st.caption("Confidence is a model output, not a guarantee that the prediction is correct.")
                st.subheader("Confidence across all digits")
                st.bar_chart({"Probability": probabilities}, x=None)
                with st.expander("All digit probabilities"):
                    st.dataframe({"Digit": list(range(10)), "Probability": probabilities}, hide_index=True)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"Could not process this image: {exc}")
    except Exception as exc:
        st.error("Prediction failed.")
        with st.expander("Technical details"):
            st.exception(exc)

st.divider()
with st.expander("📊 Original evaluation figures"):
    for filename, title in [
        ("confusion_matrix.png", "Confusion matrix"),
        ("sample_predictions.png", "Sample predictions"),
    ]:
        path = ROOT / filename
        if path.is_file():
            st.markdown(f"**{title}**")
            st.image(str(path), use_container_width=True)
        else:
            st.caption(f"{filename} is not included in this deployment.")
