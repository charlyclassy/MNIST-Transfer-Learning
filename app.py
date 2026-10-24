from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

st.set_page_config(page_title='DigitVision AI | MNIST', page_icon='🧠', layout='wide', initial_sidebar_state='expanded')
ROOT = Path(__file__).resolve().parent

# Fix sidebar text visibility
st.markdown("""
<style>
/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #111B30 !important;
}

/* All normal sidebar text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] li {
    color: #F8FAFC !important;
}

/* Radio button labels */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #F8FAFC !important;
}

/* Captions and secondary information */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #CBD5E1 !important;
}

/* Help text */
[data-testid="stSidebar"] small {
    color: #CBD5E1 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('''<div class="hero"><div class="eyebrow">DEEP LEARNING · TRANSFER LEARNING</div><h1>🧠 DigitVision AI</h1><p>Recognise handwritten digits from 0 to 9 with your original fine-tuned MobileNetV2 network.</p></div>''',unsafe_allow_html=True)

with st.sidebar:
    st.header('🔬 Model laboratory')
    choice=st.radio('Choose trained checkpoint',('Fine-tuned MobileNetV2','Transfer-learning baseline'))
    file_name = 'mnist_fine_tuned_model.keras' if choice.startswith('Fine') else 'mnist_transfer_model.keras'
    st.divider()
    st.markdown('**Architecture:** MobileNetV2 + digit classification head')
    st.markdown('**Task:** MNIST handwritten digit classification')
    st.markdown('**Input:** 32 × 32, RGB, values −1 to 1')
    st.markdown('**Output:** 10 digit probabilities')
    st.caption('These are your original saved models, not newly trained demo weights.')
    st.divider()
    st.info('Tip: Try a large, centered digit on a dark background with bright handwriting, similar to MNIST.')

@st.cache_resource(show_spinner='Loading original trained model…')
def load_model(filename):
    path=ROOT / filename
    if not path.exists():
        raise FileNotFoundError(f'Missing {filename}. Keep both original .keras files in your repository.')
    return tf.keras.models.load_model(str(path),compile=False)

def prepare(image, invert=False):
    image=ImageOps.exif_transpose(image).convert('L')
    if invert:
        image=ImageOps.invert(image)
    image=image.resize((32,32),Image.Resampling.BILINEAR)
    arr=np.asarray(image,dtype=np.float32)
    arr=np.repeat(arr[...,None],3,axis=-1)
    arr=arr/127.5-1.0
    return arr[None,...]

st.markdown('<div class="section-heading">01 / Try handwritten digit recognition</div>',unsafe_allow_html=True)
left,right=st.columns([1,1],gap='large')
with left:
    uploaded=st.file_uploader('Upload a digit image',type=['png','jpg','jpeg'],help='An image containing one handwritten number, 0–9')
    invert=st.checkbox('Invert black and white',value=False,help='Try this for dark digits on white paper; MNIST often has bright strokes on dark backgrounds.')
    if uploaded:
        try:
            pic=Image.open(uploaded)
            st.image(pic,caption='Uploaded image',width=300)
        except (OSError,UnidentifiedImageError):
            pic=None
            st.error('Could not read that image. Please choose another JPG or PNG.')
    else:
        pic=None
        st.info('Upload a handwritten digit to begin.')
with right:
    st.subheader('AI prediction')
    if pic is not None:
        if st.button('✨ Recognise digit',type='primary',use_container_width=True):
            try:
                with st.spinner('Running the trained neural network…'):
                    model=load_model(file_name)
                    x=prepare(pic,invert)
                    probs=np.asarray(model.predict(x,verbose=0)[0],dtype=float)
                    if probs.size!=10:
                        raise ValueError(f'Expected ten output scores, received {probs.size}.')
                    if not (np.all(probs>=0) and abs(probs.sum()-1)<0.02):
                        probs=tf.nn.softmax(probs).numpy()
                    probs=probs/probs.sum()
                    st.session_state['latest']=(file_name,uploaded.name,invert,probs)
            except Exception as error:
                st.error('Prediction could not be completed.')
                st.code(str(error))
    if 'latest' in st.session_state and pic is not None:
        model_used,filename,inverted,probs=st.session_state['latest']
        if (model_used,filename,inverted)!=(file_name,uploaded.name,invert):
            st.info('Image or model selection changed. Select Recognise digit to update the results.')
        else:
            digit=int(np.argmax(probs)); score=float(np.max(probs))
            c1,c2=st.columns(2)
            c1.metric('Recognised digit',str(digit))
            c2.metric('Model confidence',f'{score:.1%}')
            if score<.5:st.warning('Low confidence — try centering the digit or using the invert option.')
            st.markdown('**Probability for each digit**')
            for idx in range(10):
                a,b=st.columns([1,7])
                a.write(str(idx))
                b.progress(float(probs[idx]),text=f'{probs[idx]:.1%}')
            st.caption('Confidence is a model output, not a guarantee of correctness.')
    elif pic is None:
        st.caption('Prediction and confidence scores will appear here after you upload an image.')

st.divider()
st.markdown('<div class="section-heading">02 / Project evaluation</div>',unsafe_allow_html=True)
tab1,tab2=st.tabs(['Confusion matrix','Sample predictions'])
with tab1:
    path=ROOT/'confusion_matrix.png'
    if path.exists():st.image(str(path),caption='Original model evaluation figure',use_container_width=True)
    else:st.info('Keep confusion_matrix.png in the repository to display the evaluation.')
with tab2:
    path=ROOT/'sample_predictions.png'
    if path.exists():st.image(str(path),caption='Original model prediction examples',use_container_width=True)
    else:st.info('Keep sample_predictions.png in the repository to display examples.')
st.markdown('<div class="footer">DigitVision AI · Original MNIST transfer-learning project · Streamlit</div>',unsafe_allow_html=True)
