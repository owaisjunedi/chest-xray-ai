import tensorflow as tf
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify

# --- CORE PREDICTION LOGIC (Used by both Streamlit and Flask) ---

def prepare_image(image):
    """Processes a PIL image for the model."""
    img = image.convert('RGB').resize((224, 224))
    x = np.array(img) / 255.0
    return np.expand_dims(x, axis=0).astype('float32')

def make_prediction(model, image):
    """
    Standard function used by app.py (Streamlit)
    """
    x = prepare_image(image)
    preds = model.predict(x)
    probability = float(preds[0][0])
    label = "PNEUMONIA" if probability > 0.5 else "NORMAL"
    return label, probability

# --- FLASK API SERVER (Used for Curl/Docker) ---

app = Flask(__name__)
# Load model for the API
MODEL = tf.keras.models.load_model('pneumonia_model.h5')

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    try:
        # Open image from path provided in JSON
        image = Image.open(url)
        label, prob = make_prediction(MODEL, image)
        
        return jsonify({
            'prediction': label,
            'probability': prob
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    print("Starting Flask API Server on port 8080...")
    app.run(host='0.0.0.0', port=8080)