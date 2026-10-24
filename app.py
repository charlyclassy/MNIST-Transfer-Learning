
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Professional page title
st.markdown("""
<div class="mnist-hero">
    <h1>🧠 DigitVision AI</h1>
    <p>
        Handwritten Digit Recognition using
        Deep Learning and Transfer Learning
    </p>
    <p>
        Upload a handwritten digit and discover
        how AI recognises numbers from 0 to 9.
    </p>
</div>
""", unsafe_allow_html=True)

st.write(
    "This application uses MobileNetV2 transfer learning "
    "to recognise handwritten digits from 0 to 9."
)

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "mnist_fine_tuned_model.keras"
    )

model = load_model()

# Upload an image
uploaded_file = st.file_uploader(
    "Upload an image of a handwritten digit",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    st.image(image, caption="Uploaded Image", width=200)

    # Match MNIST preprocessing
    image = image.resize((32, 32))
    image_array = np.array(image)

    # Convert grayscale to RGB
    image_array = np.stack(
        [image_array] * 3, axis=-1
    )

    # Normalise for MobileNetV2
    image_array = (
        image_array.astype("float32") / 127.5
    ) - 1.0

    image_array = np.expand_dims(
        image_array, axis=0
    )

    # Predict digit
    prediction = model.predict(image_array, verbose=0)

    predicted_digit = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    st.success(f"Predicted Digit: {predicted_digit}")

    st.write(f"Confidence: {confidence:.2f}%")
