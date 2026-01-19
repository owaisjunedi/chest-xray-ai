import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np
import os

# 1. Initialize Interpreter
# We do this OUTSIDE the handler so it stays "warm" in the cloud
interpreter = tflite.Interpreter(model_path='pneumonia_model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def predict(img_path):
    # 2. Preprocess
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    x = np.array(img, dtype='float32')
    x = np.expand_dims(x, axis=0)
    x /= 255.
    
    # 3. Inference
    interpreter.set_tensor(input_index, x)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    
    return float(preds[0][0])

def lambda_handler(event, context):
    url = event.get('url')
    if not url:
        return {'error': 'No URL provided in event'}
    
    try:
        # Check if file exists (since we are testing locally with volume)
        if not os.path.exists(url):
            return {'error': f'File not found at path: {url}'}
            
        result = predict(url)
        prediction_label = "PNEUMONIA" if result > 0.5 else "NORMAL"
        
        return {
            'probability': result,
            'prediction': prediction_label
        }
    except Exception as e:
        return {'error': str(e)}