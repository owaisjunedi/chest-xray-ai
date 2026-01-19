FROM public.ecr.aws/lambda/python:3.9

# 1. Install system dependencies
RUN yum install -y mesa-libGL

# 2. Install pillow and numpy
RUN pip install numpy==1.23.5 pillow

# 3. Install the specific TFLite wheel compatible with this environment
RUN pip install https://github.com/google-coral/py-repo/raw/master/tflite_runtime-2.5.0.post1-cp39-cp39-linux_x86_64.whl

# 4. Copy files
COPY pneumonia_model.tflite ${LAMBDA_TASK_ROOT}
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# 5. Set the handler
CMD [ "lambda_function.lambda_handler" ]