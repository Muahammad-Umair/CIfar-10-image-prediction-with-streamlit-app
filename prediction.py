import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# CIFAR-10 labels
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('cifar10_model.h5')
    return model

model = load_model()

# Streamlit UI
st.set_page_config(page_title="CIFAR-10 Classifier", layout="centered")
st.title("🚀 CIFAR-10 Image Classifier")
st.markdown("Upload a 32x32 image (CIFAR-10 style) and predict the category.")

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

# Image preprocessing and prediction
def preprocess_image(img):
    img = img.resize((32, 32,))  # Resize to CIFAR-10 size
    img_array = np.array(img).astype("float32") / 255.0
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]  # Drop alpha channel if present
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 32, 32, 3)
    return img_array

if uploaded_file:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image",  width=200)

    img_array = preprocess_image(image)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display result
    st.success(f"**Predicted Class:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}")
