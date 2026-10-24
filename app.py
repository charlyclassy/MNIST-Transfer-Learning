"""DigitVision AI Studio: single and multi-digit inference using existing MNIST CNN checkpoints."""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ImageDraw, UnidentifiedImageError

st.set_page_config(page_title="DigitVision AI Studio", page_icon="🔢", layout="wide", initial_sidebar_state="expanded")
ROOT = Path(__file__).resolve().parent
MODELS = {
    "Fine-tuned CNN": "mnist_fine_tuned_model.keras",
    "Transfer-learning baseline": "mnist_transfer_model.keras",
}

st.markdown("""
<style>
.stApp {background:linear-gradient(135deg,#0b1020,#101b34);color:#f8fafc;}
.block-container {max-width:1220px;padding-top:2rem;padding-bottom:3rem;}
[data-testid="stSidebar"] {background:#182640;}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {color:#f1f5ff !important;opacity:1 !important;}
[data-testid="stMetric"] {background:#1e2b46;border:1px solid #43577a;border-radius:16px;padding:18px;}
[data-testid="stMetricLabel"] *, [data-testid="stMetricValue"] * {color:#f8fafc !important;opacity:1 !important;}
.hero {background:linear-gradient(115deg,#25345c,#3e32a0);border:1px solid #6d6df0;border-radius:22px;padding:32px;margin-bottom:25px;}
.hero h1 {color:white !important;font-size:2.7rem;margin:0 0 10px;}
.hero p {color:#e0e7ff !important;font-size:1.08rem;margin:0;}
.stButton>button[kind="primary"] {background:#6366f1;color:white;border:0;border-radius:12px;font-weight:700;}
[data-testid="stFileUploader"] {background:#1c2d4a;border:1px dashed #8d96ff;border-radius:16px;padding:12px;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🔢 DigitVision AI Studio</h1><p>Handwritten number recognition · MobileNetV2 CNN · Multi-digit analysis</p></div>', unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading trained CNN checkpoint…")
def load_saved_model(filename):
    path = ROOT / filename
    if not path.is_file():
        raise FileNotFoundError(f"Model file missing: {path}. Files available: {[p.name for p in ROOT.iterdir()]}")
    return tf.keras.models.load_model(str(path), compile=False)


def otsu_threshold(gray):
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / max(1, hist.sum())
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    numerator = (mu[-1] * omega - mu) ** 2
    denominator = omega * (1 - omega)
    variance = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return int(np.argmax(variance))


def foreground_mask(gray_image, mode="Auto detect", adjustment=0):
    arr = np.asarray(gray_image.convert("L"), dtype=np.uint8)
    threshold = int(np.clip(otsu_threshold(arr) + adjustment, 12, 245))
    # The surrounding border indicates the likely paper/background colour.
    border = np.concatenate((arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]))
    dark_ink = np.median(border) >= 128
    if mode == "Dark handwriting on light":
        dark_ink = True
    elif mode == "Light handwriting on dark":
        dark_ink = False
    mask = arr < threshold if dark_ink else arr > threshold
    # Some anti-aliased backgrounds can produce a full-image mask. Fallback gracefully.
    if mask.mean() > 0.72:
        mask = arr < threshold if not dark_ink else arr > threshold
    return mask


def split_regions(mask, cuts=None, gap=2):
    h, w = mask.shape
    active = mask.sum(axis=0) > 0
    if cuts is not None:
        xs = [0] + sorted(set(int(x) for x in cuts if 0 < int(x) < w)) + [w]
        bands = [(a,b) for a,b in zip(xs[:-1], xs[1:]) if b > a]
    else:
        segments = []
        start = None
        for i, v in enumerate(active):
            if v and start is None:
                start = i
            if start is not None and (not v or i == w-1):
                end = (i+1) if v and i == w-1 else i
                segments.append((start,end)); start = None
        # Merge separated strokes within a digit when gaps are very narrow.
        bands=[]
        for a,b in segments:
            if bands and a-bands[-1][1] <= gap:
                bands[-1]=(bands[-1][0],b)
            else:
                bands.append((a,b))
    boxes=[]
    min_pixels=max(3,int(mask.size*0.00015))
    for a,b in bands:
        ys, xs = np.where(mask[:,a:b])
        if len(ys) < min_pixels:
            continue
        left=a+int(xs.min()); right=a+int(xs.max())+1
        top=int(ys.min()); bottom=int(ys.max())+1
        boxes.append((left,top,right,bottom))
    return boxes


def mnist_centered(mask_crop, size=32, body=24):
    # MNIST convention: bright foreground on black, centered and aspect-preserved.
    ys,xs=np.where(mask_crop)
    if len(xs)==0:
        raise ValueError("A detected region contains no visible handwriting.")
    cropped=mask_crop[ys.min():ys.max()+1, xs.min():xs.max()+1]
    height,width=cropped.shape
    factor=body/max(height,width)
    new_size=(max(1,int(round(width*factor))),max(1,int(round(height*factor))))
    digit=Image.fromarray((cropped*255).astype(np.uint8)).resize(new_size, Image.Resampling.LANCZOS)
    canvas=np.zeros((size,size),dtype=np.uint8)
    a=np.asarray(digit)
    y=(size-a.shape[0])//2; x=(size-a.shape[1])//2
    canvas[y:y+a.shape[0],x:x+a.shape[1]]=a
    # Move centre of mass to the canvas midpoint, within available margins.
    yy,xx=np.indices(canvas.shape)
    total=canvas.sum()
    if total:
        shift_x=int(round((size-1)/2 - (xx*canvas).sum()/total))
        shift_y=int(round((size-1)/2 - (yy*canvas).sum()/total))
        shifted=np.zeros_like(canvas)
        sx0=max(0,-shift_x); sx1=min(size,size-shift_x)
        sy0=max(0,-shift_y); sy1=min(size,size-shift_y)
        shifted[sy0+shift_y:sy1+shift_y,sx0+shift_x:sx1+shift_x]=canvas[sy0:sy1,sx0:sx1]
        canvas=shifted
    return canvas


def predict_digit(model, canvas):
    rgb=np.repeat(canvas[:,:,None],3,axis=2).astype(np.float32)
    batch=(rgb / 127.5 - 1.0)[None,...]
    raw=np.asarray(model.predict(batch,verbose=0)).reshape(-1)
    if raw.size!=10 or not np.all(np.isfinite(raw)):
        raise ValueError("The saved CNN must return 10 finite digit scores.")
    if np.all(raw>=0) and np.all(raw<=1) and abs(float(raw.sum())-1)<0.01:
        probabilities=raw / raw.sum()
    else:
        e=np.exp(raw-np.max(raw)); probabilities=e/e.sum()
    return int(np.argmax(probabilities)), float(np.max(probabilities)), probabilities

with st.sidebar:
    st.header("⚙️ Recognition settings")
    selected=st.radio("Choose CNN model",list(MODELS),index=0)
    mode=st.selectbox("Handwriting colours",["Auto detect","Dark handwriting on light","Light handwriting on dark"])
    adjustment=st.slider("Contrast threshold adjustment",-50,50,0)
    separation=st.radio("Separate digits",["Automatic","Manual cut positions"],index=0)
    st.divider()
    st.caption("MobileNetV2 transfer learning · Digit segmentation + individual CNN predictions")

try:
    model=load_saved_model(MODELS[selected])
except Exception as error:
    st.error("The trained CNN could not be loaded. Keep both .keras files beside app.py in your GitHub repository.")
    with st.expander("Technical details"):
        st.exception(error)
    st.stop()

left,right=st.columns([1,1],gap="large")
with left:
    st.subheader("01 · Upload handwriting")
    uploaded=st.file_uploader("Choose a PNG or JPG of one or more digits",type=["png","jpg","jpeg"])
with right:
    st.subheader("02 · Recognise number")
    st.write("Review detected digit regions, then run your saved CNN.")

if uploaded is not None:
    try:
        image=ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
        grayscale=image.convert("L")
        mask=foreground_mask(grayscale,mode,adjustment)
        cuts=None
        if separation=="Manual cut positions":
            user_cuts=st.text_input("Column boundaries, separated by commas (pixels)",placeholder="e.g. 65, 132")
            if user_cuts.strip():
                try:
                    cuts=[int(s.strip()) for s in user_cuts.split(",") if s.strip()]
                except ValueError:
                    st.error("Enter whole-number column positions separated by commas.")
                    st.stop()
        boxes=split_regions(mask,cuts)
        drawing=image.copy(); draw=ImageDraw.Draw(drawing)
        for i,(x0,y0,x1,y1) in enumerate(boxes,1):
            draw.rectangle((x0,y0,x1,y1),outline="#A78BFA",width=max(2,image.width//180))
            draw.text((x0+3,max(0,y0-14)),str(i),fill="#E879F9")
        st.subheader("Detected handwriting regions")
        v1,v2=st.columns([3,1])
        with v1: st.image(drawing,caption="Purple boxes indicate the regions sent to the CNN",use_container_width=True)
        with v2:
            st.metric("Segments",len(boxes))
            st.metric("Image width",f"{image.width}px")
        if len(boxes)==0:
            st.warning("No digits detected. Adjust handwriting colours or contrast threshold.")
        else:
            if len(boxes)>12:
                st.warning("Many regions detected. Try a cleaner image or manual cuts.")
            if st.button("✨ Recognise handwriting",type="primary",use_container_width=True):
                digits=[]; confidences=[]; results=[]
                with st.spinner("Analysing handwriting…"):
                    for x0,y0,x1,y1 in boxes:
                        crop=mask[y0:y1,x0:x1]
                        canvas=mnist_centered(crop)
                        digit,confidence,probs=predict_digit(model,canvas)
                        digits.append(str(digit));confidences.append(confidence)
                        results.append((canvas,digit,confidence,probs))
                number_text="".join(digits)
                st.subheader("Recognised number")
                st.title(number_text)
                st.caption("Shown as text so leading zeros are preserved. Recognition may be incorrect for joined or ambiguous handwriting.")
                st.metric("Lowest per-digit confidence",f"{min(confidences):.1%}")
                st.subheader("Individual digit results")
                cols=st.columns(min(5,len(results)))
                for i,(canvas,digit,confidence,probs) in enumerate(results):
                    with cols[i%len(cols)]:
                        st.image(canvas,caption=f"Position {i+1}",width=115)
                        st.write(f"**{digit}** · {confidence:.1%}")
                with st.expander("Digit confidence breakdown"):
                    for i,(_,digit,confidence,probs) in enumerate(results,1):
                        st.markdown(f"**Position {i}: predicted {digit}**")
                        st.bar_chart(pd.DataFrame({"Digit":list(range(10)),"Probability":probs}).set_index("Digit"))
                if min(confidences)<0.65:
                    st.warning("At least one digit has low model confidence. Check the segmentation boxes and handwriting style.")
    except (UnidentifiedImageError,OSError,ValueError) as exc:
        st.error(f"Could not process the image: {exc}")
    except Exception as exc:
        st.error("Recognition failed. See the technical details below.")
        with st.expander("Technical details"):
            st.exception(exc)
else:
    st.info("Upload a handwritten number to begin. Clear gaps between digits work best.")

st.divider()
with st.expander("📊 Original evaluation figures"):
    for filename,title in [("confusion_matrix.png","Confusion matrix"),("sample_predictions.png","Sample predictions")]:
        path=ROOT/filename
        if path.is_file():
            st.markdown(f"**{title}**")
            st.image(str(path),use_container_width=True)
