import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("Pneumonia Detection AI 🏥")
st.write("Upload a Chest X-ray to get a diagnosis.")

# Load your trained model
model = tf.keras.models.load_model('pneumonia_model.h5')

uploaded_file = st.file_uploader("Choose an image...", type="jpeg")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded X-ray', use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Predict
    prediction = model.predict(x)[0][0]

    if prediction > 0.5:
        st.error(f"Result: PNEUMONIA DETECTED (Confidence: {prediction*100:.2f}%)")
    else:
        st.success(f"Result: NORMAL LUNGS (Confidence: {(1-prediction)*100:.2f}%)")