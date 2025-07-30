# streamlit_app.py
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load Keras model
@st.cache_resource
def load_keras_model():
    return load_model("model.keras")

model = load_keras_model()


# Fashion MNIST class labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

st.title("Clothing Classification")
st.write("Capture a picture with your webcam and classify it using your trained model.")


# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and preprocess the image
    img = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    img = img.resize((28, 28))                    # Resize to match dataset
    img_array = np.array(img) / 255.0             # Normalize
    img_array = img_array.reshape((1, 784))       # Flatten and add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # Display results
    st.image(img.resize((140, 140)), caption="Uploaded Image", width=140)
    st.markdown(f"### 🏷️ Predicted Class: `{class_names[predicted_class]}`")
    st.markdown(f"**Confidence:** `{confidence:.2f}`")
    st.bar_chart(prediction[0])
