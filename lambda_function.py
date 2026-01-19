import numpy as np
from PIL import Image
# This is the import for the stable 2.5.0 version
from tflite_runtime.interpreter import Interpreter
import os

# 1. Load the Model
interpreter = Interpreter(model_path='pneumonia_model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def predict(img_path):
    # 2. Pre-process the image
    img = Image.open(img_path).resize((224, 224))
    
    # Ensure image is RGB (convert if grayscale)
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    x = np.array(img, dtype='float32')
    x = np.expand_dims(x, axis=0)
    x /= 255.
    
    # 3. Run Inference
    interpreter.set_tensor(input_index, x)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    
    return float(preds[0][0])

def lambda_handler(event, context):
    url = event.get('url')
    if not url:
        return {'error': 'No URL provided'}
    
    try:
        result = predict(url)
        prediction_label = "PNEUMONIA" if result > 0.5 else "NORMAL"
        return {
            'probability': result,
            'prediction': prediction_label
        }
    except Exception as e:
        return {'error': str(e)}