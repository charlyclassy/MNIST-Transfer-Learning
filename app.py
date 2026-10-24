"""DigitVision AI Studio — use existing trained MobileNetV2 MNIST checkpoints."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

st.set_page_config(page_title="DigitVision AI Studio", page_icon="🔢", layout="wide")
ROOT = Path(__file__).resolve().parent
CHECKPOINTS = {
    "Fine-tuned CNN": "mnist_fine_tuned_model.keras",
    "Transfer-learning baseline": "mnist_transfer_model.keras",
}

st.markdown("""
<style>
.stApp {background:linear-gradient(135deg,#0b1020,#111c35);color:#f8fafc;}
.block-container {max-width:1200px;padding-top:2rem;padding-bottom:3rem;}
[data-testid="stSidebar"] {background:#182640;}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {color:#f1f5ff!important;opacity:1!important;}
[data-testid="stMetric"] {background:#1e2b46;border:1px solid #43577a;border-radius:16px;padding:18px;}
[data-testid="stMetricLabel"] *,[data-testid="stMetricValue"] * {color:#f8fafc!important;opacity:1!important;}
.hero {background:linear-gradient(115deg,#25345c,#3e32a0);border:1px solid #696bf0;border-radius:22px;padding:32px;margin-bottom:25px;}
.hero h1 {color:#fff!important;font-size:2.6rem;margin:0 0 10px;}
.hero p {color:#e0e7ff!important;font-size:1.04rem;margin:0;}
.stButton>button[kind="primary"] {background:#6366f1;color:#fff;border:0;border-radius:12px;font-weight:700;}
[data-testid="stFileUploader"] {background:#1c2d4a;border:1px dashed #8d96ff;border-radius:16px;padding:12px;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🔢 DigitVision AI Studio</h1><p>Handwritten number recognition · MobileNetV2 CNN · Multi-digit analysis</p></div>', unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading saved neural network…")
def load_checkpoint(filename):
    file = ROOT / filename
    if not file.is_file():
        raise FileNotFoundError(f"Model not found: {file}. Repository files: {[x.name for x in ROOT.iterdir()]}")
    return tf.keras.models.load_model(str(file), compile=False)


def otsu(gray):
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    p = hist / max(hist.sum(), 1)
    omega = p.cumsum()
    mu = (p * np.arange(256)).cumsum()
    denom = omega * (1 - omega)
    between = np.divide((mu[-1] * omega - mu) ** 2, denom, out=np.zeros(256), where=denom > 0)
    return int(np.argmax(between))


def make_mask(image, ink, adjustment):
    arr = np.asarray(image.convert("L"), dtype=np.uint8)
    threshold = int(np.clip(otsu(arr) + adjustment, 5, 250))
    border = np.r_[arr[0], arr[-1], arr[:, 0], arr[:, -1]]
    dark = bool(np.median(border) > 127) if ink == "Automatic" else ink == "Dark ink on light background"
    mask = arr < threshold if dark else arr > threshold
    if mask.mean() > 0.75:
        mask = ~mask
    return mask


def regions(mask, cuts=None, merge_gap=2):
    h, w = mask.shape
    if cuts is not None:
        coords = [0] + sorted({int(c) for c in cuts if 0 < c < w}) + [w]
        bands = list(zip(coords[:-1], coords[1:]))
    else:
        active = mask.any(axis=0)
        starts = np.where(active & ~np.r_[False, active[:-1]])[0]
        ends = np.where(active & ~np.r_[active[1:], False])[0] + 1
        bands = []
        for a, b in zip(starts, ends):
            if bands and a - bands[-1][1] <= merge_gap:
                bands[-1] = (bands[-1][0], int(b))
            else:
                bands.append((int(a), int(b)))
    boxes = []
    for a, b in bands:
        if a >= b:
            continue
        y, x = np.where(mask[:, a:b])
        if len(x) < max(3, int(mask.size * .0001)):
            continue
        boxes.append((a + int(x.min()), int(y.min()), a + int(x.max()) + 1, int(y.max()) + 1))
    return boxes


def digit_image(crop, strategy):
    """Return 32x32 uint8 image: foreground=white, background=black."""
    y, x = np.where(crop)
    if not len(x):
        raise ValueError("Empty digit region. Adjust the threshold or cuts.")
    region = (crop[y.min():y.max()+1, x.min():x.max()+1] * 255).astype(np.uint8)
    pil = Image.fromarray(region, "L")
    if strategy == "Match original training resize":
        # Original training resized full MNIST images directly to 32x32.
        # Each digit crop is resized directly here, without extra padding.
        return np.asarray(pil.resize((32, 32), Image.Resampling.BILINEAR))
    # Alternative: preserve shape and centre within a 32x32 canvas.
    w, h = pil.size
    ratio = 24 / max(w, h)
    scaled = pil.resize((max(1, round(w * ratio)), max(1, round(h * ratio))), Image.Resampling.BILINEAR)
    a = np.asarray(scaled)
    result = np.zeros((32, 32), dtype=np.uint8)
    yy = (32-a.shape[0])//2
    xx = (32-a.shape[1])//2
    result[yy:yy+a.shape[0], xx:xx+a.shape[1]] = a
    return result


def predict(model, digit):
    # Identical channel replication and [-1,1] mapping used during training.
    rgb = np.repeat(digit[...,None], 3, axis=2).astype(np.float32)
    batch = (rgb / 127.5 - 1.0)[None]
    raw = np.asarray(model.predict(batch, verbose=0)).reshape(-1)
    if len(raw) != 10 or not np.isfinite(raw).all():
        raise ValueError("Saved model did not produce ten valid digit scores.")
    if np.all((raw >= 0) & (raw <= 1)) and abs(raw.sum() - 1) < .01:
        probs = raw / max(float(raw.sum()), 1e-9)
    else:
        e = np.exp(raw - raw.max())
        probs = e/e.sum()
    return int(np.argmax(probs)), float(np.max(probs)), probs

with st.sidebar:
    st.header("⚙️ Recognition settings")
    choice = st.radio("CNN checkpoint", list(CHECKPOINTS))
    polarity = st.selectbox("Handwriting colours", ["Automatic", "Dark ink on light background", "Light ink on dark background"])
    threshold = st.slider("Contrast adjustment", -50, 50, 0)
    separation = st.radio("Digit separation", ["Automatic", "Manual cut positions"])
    processing = st.selectbox("Digit preparation", ["Match original training resize", "Preserve shape and centre"])
    st.caption("Try both preparation methods when a digit is misclassified; neither guarantees correctness.")

try:
    model = load_checkpoint(CHECKPOINTS[choice])
except Exception as exc:
    st.error("Unable to load the selected CNN. Check that its .keras file is in the GitHub repository.")
    with st.expander("Technical details"):
        st.exception(exc)
    st.stop()

c1, c2 = st.columns([1,1], gap="large")
with c1:
    st.subheader("01 · Upload handwriting")
    uploaded = st.file_uploader("JPG or PNG containing one or more handwritten digits", type=["jpg", "jpeg", "png"])
with c2:
    st.subheader("02 · Recognise number")
    st.write("Check the detected regions and prepared digit previews before classifying.")

if uploaded:
    try:
        image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
        mask = make_mask(image, polarity, threshold)
        cut_points = None
        if separation == "Manual cut positions":
            text = st.text_input("Vertical cut positions, comma-separated pixels", placeholder="65, 132")
            if text.strip():
                try:
                    cut_points = [int(p.strip()) for p in text.split(",") if p.strip()]
                except ValueError:
                    st.error("Use whole numbers separated by commas.")
                    st.stop()
        boxes = regions(mask, cut_points)
        overlay = image.copy()
        brush = ImageDraw.Draw(overlay)
        for i, (x0,y0,x1,y1) in enumerate(boxes, 1):
            brush.rectangle((x0,y0,x1,y1), outline="#a78bfa", width=max(2, image.width//180))
            brush.text((x0+3, max(0,y0-14)), str(i), fill="#e879f9")
        st.subheader("Detected handwriting")
        preview, stats = st.columns([3, 1])
        with preview:
            st.image(overlay, caption="Purple boxes indicate detected digit regions", use_container_width=True)
        with stats:
            st.metric("Segments", len(boxes))
            st.metric("Image width", f"{image.width}px")
        if not boxes:
            st.warning("No digit regions detected. Try changing the handwriting colour setting or threshold.")
        else:
            if len(boxes) > 15:
                st.warning("Many digit regions were detected. Check for noise and try manual cuts.")
            prepared = [digit_image(mask[y0:y1,x0:x1], processing) for x0,y0,x1,y1 in boxes]
            with st.expander("Inspect the images sent to the CNN", expanded=True):
                cols = st.columns(min(5, len(prepared)))
                for i, d in enumerate(prepared):
                    with cols[i % len(cols)]:
                        st.image(d, caption=f"Position {i+1}", width=110, clamp=True)
                st.caption("These previews show the exact black-background digit inputs, before RGB replication and normalisation.")
            if st.button("✨ Recognise handwriting", type="primary", use_container_width=True):
                results = []
                with st.spinner("Running the saved CNN…"):
                    for d in prepared:
                        results.append((d, *predict(model, d)))
                number = "".join(str(digit) for _, digit, _, _ in results)
                st.subheader("Recognised number")
                st.title(number)
                st.metric("Lowest per-digit confidence", f"{min(r[2] for r in results):.1%}")
                if min(r[2] for r in results) < .65:
                    st.warning("At least one digit has low confidence. Inspect its preview and try the alternative preparation method.")
                st.subheader("Individual digit results")
                cols = st.columns(min(5, len(results)))
                for i, (d, digit, confidence, _) in enumerate(results):
                    with cols[i % len(cols)]:
                        st.image(d, width=110)
                        st.write(f"**Position {i+1}: {digit}** · {confidence:.1%}")
                with st.expander("Confidence for all digits"):
                    for i, (_, digit, _, probs) in enumerate(results, 1):
                        st.write(f"**Position {i}: {digit}**")
                        st.bar_chart(pd.DataFrame({"Digit":range(10), "Probability":probs}).set_index("Digit"))
                st.caption("Confidence is not the same as measured accuracy. Multi-digit results depend on successful separation of individual digits.")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"Could not process this image: {exc}")
    except Exception as exc:
        st.error("Recognition failed.")
        with st.expander("Technical details"):
            st.exception(exc)
else:
    st.info("Upload an image of handwritten digits to begin.")

st.divider()
with st.expander("📊 Original evaluation figures"):
    for fname, title in [("confusion_matrix.png", "Confusion matrix"), ("sample_predictions.png", "Sample predictions")]:
        path = ROOT / fname
        if path.is_file():
            st.write(f"**{title}**")
            st.image(str(path), use_container_width=True)
