"""DigitVision AI: multi-digit handwriting recognition using existing MNIST CNN.

The saved CNN still classifies single digits 0-9. A segmentation pipeline splits
separate handwritten characters and combines predictions into a number.
"""
from pathlib import Path
import io
import re

import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
import streamlit as st
import tensorflow as tf

BASE = Path(__file__).resolve().parent
MODELS = {
    "Fine-tuned CNN": "mnist_fine_tuned_model.keras",
    "Transfer-learning baseline": "mnist_transfer_model.keras",
}

st.set_page_config(page_title="DigitVision AI | Multi-digit", page_icon="🔢", layout="wide", initial_sidebar_state="expanded")

# Professional dark blue theme
st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        135deg, #0B1020, #151D35
    ) !important;
    color: #F8FAFC !important;
}

[data-testid="stSidebar"] {
    background-color: #16243E !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}

[data-testid="stMetric"] {
    background-color: #1E293B !important;
    border: 1px solid #475569 !important;
    border-radius: 16px !important;
    padding: 20px !important;
}

[data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"] * {
    color: #FFFFFF !important;
}

.stButton > button {
    background: linear-gradient(
        90deg, #6366F1, #7C3AED
    ) !important;
    color: white !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# Fix metric card text visibility
st.markdown("""
<style>
/* Metric card background */
[data-testid="stMetric"] {
    background-color: #1E293B !important;
    border: 1px solid #475569 !important;
    border-radius: 16px !important;
    padding: 20px !important;
}

/* Metric labels */
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #CBD5E1 !important;
    opacity: 1 !important;
}

/* Large metric values */
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

/* Metric descriptions */
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #A5B4FC !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading your trained CNN checkpoint…")
def load_checkpoint(filename: str):
    path = BASE / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing '{filename}' beside app.py. Keep both original .keras files in your GitHub repository. "
            f"Files currently visible: {[p.name for p in BASE.iterdir() if p.is_file()]}"
        )
    return tf.keras.models.load_model(str(path), compile=False)


def otsu_threshold(gray: np.ndarray) -> int:
    counts = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = gray.size
    if total == 0:
        return 127
    probability = counts / total
    omega = np.cumsum(probability)
    mu = np.cumsum(probability * np.arange(256))
    numerator = (mu[-1] * omega - mu) ** 2
    denominator = omega * (1 - omega)
    variance = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return int(np.argmax(variance))


def prepare_ink(pil_image: Image.Image, threshold_adjust: int, mode: str):
    grayscale = np.asarray(ImageOps.exif_transpose(pil_image).convert("L"), dtype=np.uint8)
    threshold = int(np.clip(otsu_threshold(grayscale) + threshold_adjust, 1, 254))
    if mode == "Light ink on dark background":
        mask = grayscale > threshold
    elif mode == "Dark ink on light background":
        mask = grayscale < threshold
    else:
        # Border is more representative of page/background than the central digit.
        border = np.concatenate((grayscale[0, :], grayscale[-1, :], grayscale[:, 0], grayscale[:, -1]))
        mask = grayscale < threshold if np.median(border) > 127 else grayscale > threshold
    return mask, grayscale, threshold


def ink_extent(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("No handwriting detected. Try another image, background setting or threshold.")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def automatic_boxes(mask: np.ndarray, min_width: int = 2):
    # Segment along empty vertical columns, then tightly bound ink in each segment.
    x0, y0, x1, y1 = ink_extent(mask)
    col_active = np.any(mask[y0:y1, x0:x1], axis=0)
    changes = np.diff(np.r_[False, col_active, False].astype(np.int8))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    boxes = []
    for start, end in zip(starts, ends):
        if end - start < min_width:
            continue
        sub = mask[y0:y1, x0 + start:x0 + end]
        if sub.sum() < 8:
            continue
        local_y, local_x = np.where(sub)
        boxes.append((x0 + start + int(local_x.min()), y0 + int(local_y.min()),
                      x0 + start + int(local_x.max()) + 1, y0 + int(local_y.max()) + 1))
    return boxes


def manual_boxes(mask: np.ndarray, cuts: str):
    x0, y0, x1, y1 = ink_extent(mask)
    values = [int(x.strip()) for x in cuts.split(",") if x.strip()]
    if not values or sorted(values) != values or len(set(values)) != len(values):
        raise ValueError("Enter increasing comma-separated cut positions, e.g. 35,70.")
    if any(v <= 0 or v >= 100 for v in values):
        raise ValueError("Cut positions must be percentages between 1 and 99.")
    separators = [x0] + [x0 + round((x1 - x0) * v / 100) for v in values] + [x1]
    boxes = []
    for left, right in zip(separators[:-1], separators[1:]):
        sub = mask[y0:y1, left:right]
        if sub.sum() < 8:
            raise ValueError("A manual segment contains no detectable ink; adjust your cut positions.")
        ys, xs = np.where(sub)
        boxes.append((left + int(xs.min()), y0 + int(ys.min()), left + int(xs.max()) + 1, y0 + int(ys.max()) + 1))
    return boxes


def make_digit_input(mask: np.ndarray, box):
    """Centre each segmented digit on a 32x32 RGB, white-ink / black canvas."""
    left, top, right, bottom = box
    crop = mask[top:bottom, left:right]
    # Put digit inside a square canvas with ~15% margin before model resize.
    height, width = crop.shape
    side = max(height, width)
    margin = max(2, round(side * 0.22))
    canvas = np.zeros((side + 2 * margin, side + 2 * margin), dtype=np.uint8)
    oy = (canvas.shape[0] - height) // 2
    ox = (canvas.shape[1] - width) // 2
    canvas[oy:oy + height, ox:ox + width] = crop.astype(np.uint8) * 255
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    im = Image.fromarray(canvas).resize((32, 32), resampling)
    rgb = np.repeat(np.asarray(im, dtype=np.float32)[..., None], 3, axis=-1)
    return rgb / 127.5 - 1.0, im


def get_probabilities(output: np.ndarray):
    scores = np.asarray(output).reshape(-1).astype(np.float64)
    if scores.size != 10:
        raise ValueError(f"Expected the original 10-digit model output, received {scores.size} values.")
    if np.all(scores >= 0) and np.all(scores <= 1) and abs(scores.sum() - 1) < 0.02:
        probs = scores / scores.sum()
    else:
        exp = np.exp(scores - np.max(scores))
        probs = exp / exp.sum()
    return probs


st.markdown("""
<div class="dv-hero"><div class="dv-eyebrow">DEEP LEARNING · HANDWRITING RECOGNITION</div>
<h1>🔢 DigitVision AI Studio</h1>
<p>Recognise handwritten numbers — from a single digit to multi-digit sequences — using your trained CNN and smart digit separation.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Recognition settings")
    chosen_model = st.radio("Choose CNN model", list(MODELS), index=0)
    st.divider()
    background = st.selectbox("Handwriting colours", ["Auto detect", "Dark ink on light background", "Light ink on dark background"])
    threshold_adjust = st.slider("Contrast threshold adjustment", -75, 75, 0, 5)
    method = st.radio("Separate digits", ["Automatic", "Manual cut positions"])
    cuts = st.text_input("Cut positions (% of writing width)", "50", help="For 20, enter 50. For 125, try 33,66. Adjust to match spaces between digits.") if method == "Manual cut positions" else ""
    st.divider()
    st.caption("**Network:** MobileNetV2 transfer learning\n\n**Recognition:** CNN + digit segmentation\n\n**Source:** existing trained model files")

left, right = st.columns([1.1, 1], gap="large")
with left:
    st.subheader("01 · Upload handwriting")
    uploaded_file = st.file_uploader("Image containing one or more handwritten digits", type=["png", "jpg", "jpeg"])
    st.caption("Works best with separated digits on a clear background. Joined or overlapping handwriting may need manual cuts.")
with right:
    st.subheader("02 · Recognise number")
    st.write("Choose an image, review the detected digit regions, then run the CNN.")

if uploaded_file is None:
    st.info("Upload a handwriting image above to begin.")
else:
    try:
        raw_image = Image.open(io.BytesIO(uploaded_file.getvalue()))
        if raw_image.width * raw_image.height > 16_000_000:
            raise ValueError("Image is too large. Use an image smaller than 16 megapixels.")
        ink_mask, gray, threshold = prepare_ink(raw_image, threshold_adjust, background)
        boxes = manual_boxes(ink_mask, cuts) if method == "Manual cut positions" else automatic_boxes(ink_mask)
        if not boxes:
            raise ValueError("No usable digit regions were found. Check image contrast or use manual cuts.")
        if len(boxes) > 40:
            raise ValueError("More than 40 regions detected. Crop the image to a single handwritten number.")
        annotated = ImageOps.exif_transpose(raw_image).convert("RGB")
        overlay = ImageDraw.Draw(annotated)
        for idx, (x0, y0, x1, y1) in enumerate(boxes, 1):
            overlay.rectangle((x0, y0, x1, y1), outline="#a78bfa", width=max(2, raw_image.width // 250))
            overlay.text((x0 + 2, max(0, y0 - 14)), str(idx), fill="#a78bfa")
        st.subheader("Detected handwriting regions")
        col_image, col_info = st.columns([2, 1])
        with col_image:
            st.image(annotated, caption="Purple boxes show the digit regions selected for recognition", use_container_width=True)
        with col_info:
            st.metric("Segments", len(boxes))
            st.metric("Image width", f"{raw_image.width}px")
            st.caption(f"Applied intensity threshold: {threshold}")
            if len(boxes) == 1 and method == "Automatic":
                st.warning("Only one region found. If you uploaded a multi-digit number, switch to manual cut positions.")
        if st.button("✨ Recognise handwriting", type="primary", use_container_width=True):
            with st.spinner("Running the saved CNN on each detected region…"):
                model = load_checkpoint(MODELS[chosen_model])
                processed = [make_digit_input(ink_mask, box) for box in boxes]
                batch = np.stack([arr for arr, _ in processed], axis=0)
                raw_preds = model.predict(batch, verbose=0)
                probabilities = np.vstack([get_probabilities(pred) for pred in raw_preds])
                predicted = probabilities.argmax(axis=1)
                confidences = probabilities.max(axis=1)
                predicted_text = "".join(map(str, predicted))
            st.markdown("<div class='dv-card'><div class='dv-eyebrow'>RECOGNISED NUMBER</div><div class='dv-number'>" + predicted_text + "</div></div>", unsafe_allow_html=True)
            st.caption("Read the number as text to preserve leading zeroes, e.g. 007.")
            st.metric("Lowest per-digit confidence", f"{float(np.min(confidences))*100:.1f}%")
            if np.min(confidences) < 0.6:
                st.warning("At least one digit has low confidence. Check the segment boxes or try adjusting contrast/cuts.")
            st.subheader("Individual digit results")
            columns = st.columns(min(len(processed), 5))
            for i, ((_, thumb), digit, confidence) in enumerate(zip(processed, predicted, confidences)):
                with columns[i % len(columns)]:
                    st.image(thumb.resize((112,112)), caption=f"Position {i+1}", width=112)
                    st.write(f"**{digit}** · {confidence*100:.1f}%")
            st.subheader("Digit confidence breakdown")
            st.bar_chart({f"Position {i+1}": probabilities[i] for i in range(len(predicted))})
            st.download_button("Download transcription (.txt)", data=predicted_text + "\n", file_name="digitvision_result.txt", mime="text/plain")
    except (UnidentifiedImageError, OSError, ValueError, FileNotFoundError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error("The model could not complete recognition. See details below.")
        with st.expander("Technical details"):
            st.exception(exc)

with st.expander("Model evaluation and limitations"):
    st.write("The CNN is trained to classify individual digits, while this app handles larger numbers by separating them into digit regions and reading each region from left to right. It does not train a new sequence-recognition model. Connected, overlapping, slanted, or very noisy handwriting may be mis-segmented.")
    if (BASE / "confusion_matrix.png").exists():
        st.image(str(BASE / "confusion_matrix.png"), caption="Original single-digit evaluation confusion matrix", use_container_width=True)
    if (BASE / "sample_predictions.png").exists():
        st.image(str(BASE / "sample_predictions.png"), caption="Original example predictions", use_container_width=True)

st.caption("DigitVision AI · TensorFlow / Keras · MobileNetV2 · Streamlit")
