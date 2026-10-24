import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

st.set_page_config(page_title="DigitVision AI | MNIST", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.stApp {background: #0b1224; color: #f8fafc;}
.block-container {max-width: 1150px; padding-top: 2rem; padding-bottom: 3rem;}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stSidebar"] {background: #17243b !important;}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] li {color: #f1f5f9 !important;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color: #cbd5e1 !important;}
[data-testid="stSidebar"] div[role="radiogroup"] label {color: #f1f5f9 !important;}
.hero {background: linear-gradient(115deg, #1e293b, #3730a3); border: 1px solid #4f46e5;
  border-radius: 22px; padding: 30px 36px; margin-bottom: 24px;}
.hero h1 {color: #fff !important; font-size: clamp(2rem, 4vw, 3rem); margin: 0 0 10px;}
.hero p {color: #e0e7ff !important; font-size: 1.05rem; margin: 0;}
[data-testid="stFileUploader"] {background: #192641; border: 1px dashed #818cf8; border-radius: 16px; padding: 12px;}
[data-testid="stMetric"] {background: #1e293b; border: 1px solid #475569; border-radius: 14px; padding: 12px;}
[data-testid="stImage"] img {border-radius: 12px;}
.stButton > button[kind="primary"] {background: #6366f1; color: #fff; border-radius: 10px; border: none;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
 <h1>🧠 DigitVision AI</h1>
 <p>Handwritten Digit Recognition · MobileNetV2 Transfer Learning · MNIST</p>
</div>
""", unsafe_allow_html=True)

CHECKPOINTS = {
    "Fine-tuned MobileNetV2": "mnist_fine_tuned_model.keras",
    "Transfer-learning baseline": "mnist_transfer_model.keras",
}

with st.sidebar:
    st.header("🔬 Model laboratory")
    st.markdown("**Choose trained checkpoint**")
    selected_name = st.radio("Model", list(CHECKPOINTS), label_visibility="collapsed")
    st.divider()
    st.markdown("**Architecture:** MobileNetV2 + digit classification head")
    st.markdown("**Task:** MNIST handwritten digit classification")
    st.markdown("**Input:** 32 × 32 RGB, values −1 to 1")
    st.markdown("**Output:** 10 digit probabilities")

@st.cache_resource(show_spinner="Loading saved neural network…")
from pathlib import Path

@st.cache_resource
def load_model(path):
    model_path = Path(__file__).resolve().parent / path

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            f"Available files: "
            f"{[p.name for p in model_path.parent.iterdir()]}"
        )

    return tf.keras.models.load_model(
        str(model_path),
        compile=False
    )

model_path = CHECKPOINTS[selected_name]
if not os.path.isfile(model_path):
    st.error(f"Missing model file: {model_path}. Keep both original .keras files in the repository root.")
    st.stop()

try:
    model = load_model(model_path)
except Exception as exc:
    st.error("The saved model could not be loaded. Check TensorFlow version compatibility in requirements.txt.")
    with st.expander("Technical details"):
        st.exception(exc)
    st.stop()

st.subheader("Recognise a handwritten digit")
st.write("Upload one image containing a digit from 0 to 9 to see the prediction and all ten scores.")
uploaded_file = st.file_uploader("Upload a PNG or JPG image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        original = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("L")
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("Please upload a valid PNG or JPG image.")
        st.stop()

    col_image, col_results = st.columns([1, 1.4], gap="large")
    with col_image:
        st.markdown("#### Uploaded image")
        st.image(original, caption="Input image", use_container_width=True)

    # Preserve the preprocessing used in the original project.
    resized = original.resize((32, 32))
    array = np.asarray(resized, dtype=np.float32)
    array = np.stack([array] * 3, axis=-1)
    array = (array / 127.5) - 1.0
    batch = np.expand_dims(array, axis=0)

    with col_results:
        if st.button("✨ Recognise digit", type="primary", use_container_width=True):
            try:
                raw = np.asarray(model.predict(batch, verbose=0))[0].reshape(-1)
                if raw.size != 10:
                    raise ValueError(f"Expected 10 model outputs, received {raw.size}.")
                # If the saved head returns logits, convert to probabilities.
                if np.any(raw < 0) or not np.isclose(raw.sum(), 1, atol=0.02):
                    probabilities = tf.nn.softmax(raw).numpy()
                else:
                    probabilities = raw / raw.sum()
                digit = int(np.argmax(probabilities))
                confidence = float(probabilities[digit])
                st.markdown("#### Prediction")
                c1, c2 = st.columns(2)
                c1.metric("Recognised digit", str(digit))
                c2.metric("Model confidence", f"{confidence:.1%}")
                st.markdown("#### Confidence by digit")
                st.bar_chart({str(i): float(probabilities[i]) for i in range(10)}, horizontal=True)
                st.caption("Confidence is a model score, not a guarantee that the prediction is correct.")
            except Exception as exc:
                st.error("Prediction could not be completed.")
                with st.expander("Technical details"):
                    st.exception(exc)
    
with st.expander("📊 Existing evaluation results"):
    st.write("These figures were generated by your original project. They are not recalculated for each upload.")
    for filename, title in [
        ("confusion_matrix.png", "Confusion matrix"),
        ("sample_predictions.png", "Sample predictions"),
    ]:
        if os.path.isfile(filename):
            st.markdown(f"**{title}**")
            st.image(filename, use_container_width=True)
        else:
            st.caption(f"{filename} is not present in the deployed repository.")
