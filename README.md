# Pneumonia Detection AI: Serverless Deep Learning System

## 🏥 Project Overview
This project is an end-to-end medical imaging system designed to detect Pneumonia from Chest X-ray images. It transitions from traditional machine learning to a modern, scalable **Deep Learning** stack.

## 🚀 Advanced Tech Stack
* **Model:** Convolutional Neural Network (CNN) using **MobileNetV2** (Transfer Learning).
* **Frameworks:** TensorFlow, Keras, Scikit-learn.
* **Optimization:** Model converted to **TensorFlow Lite** for lightweight inference.
* **Deployment:** Containerized with **Docker** and architected for **AWS Lambda (Serverless)**.
* **Scale:** Designed to follow **Kubernetes** principles for high-availability serving.

## 📊 Performance Metrics
The model achieved an overall accuracy of **86%** on the unseen test set, with a critical focus on Recall:
* **Pneumonia Recall:** 0.99 (99% of positive cases detected).
* **Overall Accuracy:** 86%.

## 🏗️ Architecture (Serverless MLOps)
1.  **Training:** Fine-tuned a pre-trained MobileNetV2 brain on 5,000+ medical images.
2.  **Conversion:** Optimized the 25MB model into an 8MB `.tflite` file.
3.  **Dockerization:** Used a specialized AWS Lambda base image to minimize cold-start latency.
4.  **Inference:** Built a Python handler using `tflite-runtime` to process doctor uploads in < 1 second.



## 🛠️ How to Run
1. Build the container: `docker build -t pneumonia-ai .`
2. Run locally: `docker run -p 8080:8080 pneumonia-ai`
3. Test: `python test_lambda.py`