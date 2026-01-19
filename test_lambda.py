import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image

def test_tflite(img_path):
    interpreter = tflite.Interpreter(model_path='pneumonia_model.tflite')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    x = np.array(img, dtype='float32') / 255.0
    x = np.expand_dims(x, axis=0)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"Prediction Probability: {preds[0][0]}")
    print("Result: PNEUMONIA" if preds[0][0] > 0.5 else "Result: NORMAL")

if __name__ == "__main__":
    test_tflite('data/test/PNEUMONIA/person100_bacteria_475.jpeg')