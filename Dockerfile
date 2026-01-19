# 1. Use the stable Python 3.9 Lambda image
FROM public.ecr.aws/lambda/python:3.9

# 2. Update pip to the latest version
RUN pip install --upgrade pip

# 3. Install core dependencies
# We lock these versions for maximum stability
RUN pip install numpy==1.23.5 pillow==10.0.1

# 4. Install tflite-runtime from PyPI (no external URLs)
RUN pip install tflite-runtime==2.11.0

# 5. Copy the model and code
# Ensure pneumonia_model.tflite is in your current folder
COPY pneumonia_model.tflite ${LAMBDA_TASK_ROOT}
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# 6. Set the handler
CMD [ "lambda_function.lambda_handler" ]