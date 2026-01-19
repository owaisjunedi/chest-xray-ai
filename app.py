import streamlit as st
import tensorflow as tf
from PIL import Image
from predict import make_prediction

st.set_page_config(page_title="Pneumonia AI Detector", page_icon="🏥")

st.title("Chest X-ray Pneumonia Detector 🏥")
st.write("Upload a pediatric chest X-ray for an AI-powered diagnosis.")

# Load model once and cache it for speed
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('pneumonia_model.h5')

model = load_my_model()

uploaded_file = st.file_uploader("Upload X-ray Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Scan", use_container_width=True)
    
    with st.spinner('Analyzing scan...'):
        label, prob = make_prediction(model, image)
        
    if label == "PNEUMONIA":
        st.error(f"Prediction: **{label}**")
        st.write(f"Confidence: {prob*100:.2f}%")
    else:
        st.success(f"Prediction: **{label}**")
        st.write(f"Confidence: {(1-prob)*100:.2f}%")