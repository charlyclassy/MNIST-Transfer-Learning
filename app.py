"""DigitVision AI Studio: single- and multi-digit inference using trained DigitVisionCNN."""
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, UnidentifiedImageError
from model_utils import (grayscale, foreground, extract_boxes, boxes_from_cuts,
                         normalise_digit, predict_digits)

st.set_page_config(page_title='DigitVision AI Studio', page_icon='🔢', layout='wide')
BASE = Path(__file__).resolve().parent
MODEL_FILE = BASE / 'digitvision_cnn.keras'

st.markdown('''<style>
.stApp {background:linear-gradient(125deg,#080f23,#111e3b 65%,#101a32);color:#f7faff;}
.block-container {max-width:1200px;padding-top:1.5rem;padding-bottom:3rem;}
[data-testid="stSidebar"] {background:#142340!important;}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] span,[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2 {color:#f7faff!important;opacity:1!important;}
.hero {background:linear-gradient(115deg,#243e83,#4932a5);padding:29px 34px;border-radius:22px;
 border:1px solid #7382ed;box-shadow:0 15px 35px #03081a77;margin-bottom:24px;}
.hero h1 {color:white!important;margin:0;font-size:2.5rem;}
.hero p {color:#e3eaff!important;margin:10px 0 0;font-size:1.08rem;}
[data-testid="stMetric"] {background:#192b4d!important;border:1px solid #50648b!important;
 border-radius:14px;padding:17px;}
[data-testid="stMetricLabel"] *,[data-testid="stMetricValue"] *,
[data-testid="stMetricValue"] {color:#fff!important;opacity:1!important;}
[data-testid="stFileUploader"] {background:#1b2f50;border:1px dashed #909cf5;
 border-radius:14px;padding:13px;}
.stButton>button[kind="primary"] {background:linear-gradient(90deg,#5c64e8,#7c3aed)!important;
 color:white!important;border:0!important;border-radius:12px!important;font-weight:700!important;}
[data-testid="stAlert"] {border-radius:12px;}
</style>''', unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🔢 DigitVision AI Studio</h1><p>Deep learning handwritten number recognition · Dedicated CNN · Multi-digit analysis</p></div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner='Loading trained DigitVision CNN...')
def get_model():
    import tensorflow as tf
    return tf.keras.models.load_model(str(MODEL_FILE), compile=False)


with st.sidebar:
    st.header('⚙️ Recognition settings')
    polarity = st.selectbox('Handwriting colours', ['Auto', 'Dark ink on light background', 'Light ink on dark background'])
    offset = st.slider('Segmentation contrast', -35, 35, 0, help='Used to detect digit boundaries. Predictions retain grayscale strokes.')
    separation = st.radio('Digit separation', ['Automatic', 'Manual cut positions'])
    cuts = ''
    if separation == 'Manual cut positions':
        cuts = st.text_input('Vertical cut positions (pixels)', placeholder='e.g. 75, 145',
                             help='Enter x coordinates between digits, measured from the left edge of the uploaded image.')
    show_details = st.checkbox('Show digit previews', value=True)
    st.divider()
    st.caption('The new CNN must be trained first. Use train_model.py to produce digitvision_cnn.keras.')

left, right = st.columns([1.05, 0.95], gap='large')
with left:
    st.subheader('01 · Upload handwriting')
    uploaded = st.file_uploader('Choose a JPG or PNG with one or more handwritten digits', type=['png', 'jpg', 'jpeg'])
    st.caption('For best results, use clear, separated digits. Touching digits may require manual cuts.')

if not MODEL_FILE.is_file():
    st.warning('**Trained model not found.** Train the new CNN with `python train_model.py --epochs 15`, then upload `digitvision_cnn.keras` alongside `app.py` in GitHub.')
    st.info('This package includes the architecture and training scripts, not fabricated trained weights. You can upload and view the interface now; predictions become available after training.')

if uploaded is not None:
    try:
        source = Image.open(uploaded)
        source.load()
        if source.width * source.height > 15_000_000:
            st.error('Image too large. Please use an image smaller than 15 megapixels.')
            st.stop()
        image = source.convert('RGBA')
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f'Could not open the image: {exc}')
        st.stop()

    gray = grayscale(image)
    ink, mask = foreground(gray, polarity, offset)
    try:
        if separation == 'Manual cut positions' and cuts.strip():
            boxes = boxes_from_cuts(mask, cuts)
        else:
            boxes = extract_boxes(mask)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    # Limit excessive regions and discard tiny noise.
    boxes = [b for b in boxes if (b[2]-b[0]) * (b[3]-b[1]) >= 9]
    boxes = sorted(boxes, key=lambda b: b[0])
    preview = image.convert('RGB').copy()
    draw = ImageDraw.Draw(preview)
    for idx, (x1, y1, x2, y2) in enumerate(boxes, 1):
        draw.rectangle((x1, y1, x2, y2), outline='#9B7CFF', width=max(2, image.width // 150))
        draw.text((x1, max(0, y1-15)), str(idx), fill='#F59E0B')

    with left:
        st.image(image, caption='Uploaded handwriting', use_container_width=True)
    with right:
        st.subheader('02 · Detected digit regions')
        st.image(preview, caption='Purple boundaries = individual digit regions', use_container_width=True)
        a, b = st.columns(2)
        a.metric('Detected digits', str(len(boxes)))
        b.metric('Image width', f'{image.width}px')

    if not boxes:
        st.warning('No digits detected. Try another handwriting colour or adjust the segmentation contrast.')
    elif len(boxes) > 30:
        st.warning('More than 30 regions detected. Reduce noise or manually separate the digits.')
    else:
        digits = []
        try:
            for box in boxes:
                digits.append(normalise_digit(ink, box))
        except ValueError as exc:
            st.warning(f'Could not prepare a digit: {exc}')
            st.stop()
        if show_details:
            with st.expander('🔬 Inspect the actual 28×28 images sent to the CNN', expanded=True):
                columns = st.columns(min(6, len(digits)))
                for idx, digit in enumerate(digits):
                    columns[idx % len(columns)].image(digit, caption=f'Digit region {idx+1}', width=100, clamp=True)

        if st.button('✨ Recognise Number', type='primary', use_container_width=True, disabled=not MODEL_FILE.is_file()):
            try:
                model = get_model()
                probabilities = predict_digits(model, digits)
                choices = probabilities.argmax(axis=1).tolist()
                confidences = probabilities.max(axis=1).tolist()
                result = ''.join(str(n) for n in choices)
                st.divider()
                st.subheader('03 · Recognition results')
                st.success(f'**Predicted number: {result}**')
                if len(choices) > 1:
                    st.caption('Multi-digit result is assembled from individual predictions, left to right.')
                if min(confidences) < 0.8:
                    st.warning('At least one digit has low model confidence. Inspect the prepared images or try manual separation.')
                for i, (pred, conf, p) in enumerate(zip(choices, confidences, probabilities), 1):
                    with st.expander(f'Digit {i}: predicted {pred} · model confidence {conf:.1%}', expanded=len(choices) <= 3):
                        st.bar_chart({str(k): float(v) for k, v in enumerate(p)}, horizontal=True)
                st.caption('Confidence is not guaranteed correctness. For unfamiliar handwriting, model accuracy may be lower than on MNIST.')
            except Exception as exc:
                st.error(f'Unable to run the trained model: {exc}')
else:
    with right:
        st.subheader('02 · Recognition preview')
        st.info('Upload handwriting to see digit detection, model inputs and predictions.')

st.divider()
st.caption('DigitVision AI · Digit classification uses a dedicated CNN. Multi-digit segmentation is heuristic, so touching digits may need manual correction.')
