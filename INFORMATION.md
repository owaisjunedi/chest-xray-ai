# 📘 Project Learning Guide: Pneumonia Detection AI

Welcome! This document is a masterclass in modern Machine Learning (ML). It explains the concepts, the technology, and the exact commands used to build this clinical-grade AI.

---

## 🧩 1. The High-Level Concepts

### What is AI/ML?

* **Artificial Intelligence (AI):** The broad concept of machines acting "smart."
* **Machine Learning (ML):** A subset of AI where we don't give the computer rules (e.g., "if there is a white spot, it's pneumonia"). Instead, we show it 5,000 examples and let it figure out the patterns itself.
* **Deep Learning (DL):** A specialized type of ML that uses "Neural Networks" (inspired by the human brain) to look at complex data like images.

### What is a CNN?

We used a **Convolutional Neural Network**. Think of it as a series of digital filters. The first filters see simple lines; middle filters see shapes (like the curve of a rib); final filters see complex textures (like the "cloudiness" of infected lungs).

| CNN | 
|:-----------------------:|
| ![CNN](screenshots/CNN.jpg) | 
---

## 🛠️ 2. The Tech Stack Explained

| Tool | Purpose | Real-world Analogy |
| --- | --- | --- |
| **Python** | The programming language. | The construction material. |
| **Jupyter Notebook** | An interactive "scratchpad" for experiments. | A scientist's lab journal. |
| **TensorFlow/Keras** | The engine that builds the Neural Network. | The power tools. |
| **Pipenv** | Keeps libraries organized in a "virtual environment." | A dedicated toolbox for one specific job. |
| **Docker** | Packages the app so it runs on any computer. | A shipping container. |
| **Streamlit** | Turns Python code into a website. | The storefront for your business. |

---

## 🚀 3. The Execution Roadmap (Terminal Commands)

This is the exact sequence of commands we ran to build this project from scratch.

### **Phase 1: Environment Setup**

First, we created a clean "room" so our libraries wouldn't clash with other projects.

```bash
# Install the environment manager
pip install pipenv

# Create the environment and install core AI libraries
pipenv install tensorflow-cpu pillow numpy flask scikit-learn streamlit

# Activate the environment (the "Virtual Environment")
pipenv shell

```

### **Phase 2: Getting the Data**

We pulled the images from Kaggle (the "LinkedIn for Data Scientists").

```bash
# Install Kaggle tool
pip install kaggle

# Download the specific dataset
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Unzip the data
unzip chest-xray-pneumonia.zip

```

### **Phase 3: The Research Phase**

We used Jupyter to see the images and try different model settings.

```bash
# Launch the lab environment
jupyter notebook

```

*In the notebook, we looked for data imbalances (more sick kids than healthy ones) and plotted the "Confusion Matrix" to see where the AI was getting confused.*

### **Phase 4: Moving to Production**

Notebooks are for research; Python scripts are for production. We moved our logic into structured files.

**Step A: Training**

```bash
# This creates the "brain" (pneumonia_model.h5)
python train.py

```
- __Logic:__ It uses Transfer Learning (MobileNetV2). Instead of teaching the AI what a "line" is, we use a pre-trained brain and only teach it what "Pneumonia textures" look like.


**Step B: The API (The Backend)**

```bash
# This starts a server that waits for a "request" to predict an image
python predict.py

```
- __Logic:__ It runs a Flask Web Server on Port 8080. It stays "alive" to listen for images and return a JSON diagnosis.

**Step C: Testing the API**
In a second terminal window:

```bash
curl -X POST "http://127.0.0.1:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{"url": "data/test/PNEUMONIA/person100_bacteria_475.jpeg"}'

```

### **Phase 5: Containerization (Docker)**

We want this to work on a hospital's server without them installing Python.

```bash
# Create the library list
pipenv run pip freeze > requirements.txt

# Build the "Shipping Container"
docker build -t pneumonia-ai-app .

# Run the container (Map port 8080)
docker run -p 8080:8080 pneumonia-ai-app

```
- __Logic:__ Follows the "Blueprint" in the Dockerfile. <br>
The -v (Volume) flag allows the container to "see" the images on your hard drive.

### **Phase 6: The User Interface**

The final product for the doctor. <br>
__Streamlit__ allows us to build a full UI in pure Python without needing to learn HTML/JavaScript.

```bash
streamlit run app.py

```

---

## 🩺 4. Deep Dive: Transfer Learning (MobileNetV2)

We didn't build a new brain; we "borrowed" one.

1. **MobileNetV2** was already trained on 1.4 million images to recognize shapes.
2. We "froze" its existing knowledge (edges, curves).
3. We added a new "Final Layer" and trained it specifically on X-rays.
4. **The Result:** We achieved **99% Recall** in just 5 minutes of training.

---

## ⚠️ 5. Lessons from the "GLIBC" Struggle

In Module 9, we tried deploying to **AWS Lambda**. We learned a hard lesson:

* **The Conflict:** AI libraries (TFLite) often require modern system libraries (`GLIBC`).
* **The Problem:** Some cloud environments (Amazon Linux 2) are "older" than the libraries we want to use.
* **The Solution:** In the real world, you either upgrade your OS (Amazon Linux 2023) or pivot to a more flexible hosting service like **Streamlit Cloud**, which is what we did for the final demo.

---

## 📂 6. File-by-File Summary

* `predict.py`: The **Dual-Purpose** engine. It provides the function for Streamlit and the Server for the API.
* `app.py`: The **Frontend**. It's just the "face" of the project.
* `pneumonia_model.h5`: The **Brain**. A binary file containing the neural weights.
* `.gitignore`: The **Filter**. Tells GitHub: "Don't upload my 2GB of raw data!"

---

### 📂 7. Repository File Guide

Since you are organizing your files for the long term, here is the updated list of every file in your folder and exactly what it does:

| File | Purpose |
| --- | --- |
| **`data/`** | Raw image data (Train/Test/Val). *Note: Should be ignored by git.* |
| **`screenshots/`** | Visual proof of the application working for your portfolio. |
| **`notebook.ipynb`** | The "Science Lab": EDA, experimental training, and final metrics. |
| **`train.py`** | The "Production Line": Script to recreate the model from scratch. |
| **`predict.py`** | The "Engine": Contains the prediction logic for the UI and the Flask API. |
| **`app.py`** | The "Storefront": The Streamlit web dashboard for users. |
| **`lambda_function.py`** | The "Cloud Logic": Code specifically formatted for AWS Lambda (Serverless). |
| **`test_lambda.py`** | The "Testing Rig": Local verification of the `.tflite` model logic. |
| **`Dockerfile`** | The "Blueprint": Instructions for building the Docker container. |
| **`pneumonia_model.h5`** | The "Full Brain": The complete TensorFlow model used by Streamlit/Flask. |
| **`pneumonia_model.tflite`** | The "Light Brain": Compressed model optimized for Cloud/Mobile use. |
| **`requirements.txt`** | The "Ingredients": List of Python libraries needed to run the app. |
| **`Pipfile` / `Pipfile.lock**` | The "Safe": Ensures everyone uses the exact same library versions. |
| **`.gitignore`** | The "Guard": Prevents 2GB of data from breaking your GitHub upload. |

---
