# 🏥 Chest X-Ray Pneumonia Detection AI

An end-to-end Deep Learning project designed to assist in the diagnosis of Pneumonia using pediatric chest X-rays. This system leverages **Transfer Learning** and a **Streamlit** web interface to provide rapid, high-recall predictions.

## 🎯 Project Impact

In medical imaging, missing a positive case (a False Negative) is a critical error. This model prioritizes **Recall**, achieving a **99% success rate** in identifying pneumonia cases in the test set, ensuring patient safety is the top priority.

---

## 🚀 Technical Stack

* **Deep Learning Framework:** TensorFlow / Keras
* **Model Architecture:** MobileNetV2 (Transfer Learning)
* **Web Interface:** Streamlit
* **Data Processing:** NumPy, Pillow (PIL), Scikit-Learn
* **Training Environment:** GitHub Codespaces (CPU-based)

---

## 📊 Performance Analysis (Module 4 & 8)

The model was trained on over 5,000 images and evaluated against a 624-image test set.

### **Metrics Summary**

| Metric | Score | Why it matters? |
| --- | --- | --- |
| **Pneumonia Recall** | **0.99** | Only 5 out of 390 cases were missed. |
| **Overall Accuracy** | **86%** | High reliability on unseen data. |
| **Training Accuracy** | **97%** | Successful learning of medical textures. |

---

## 🛠️ Project Journey & Lessons Learned

### **1. The Model (Transfer Learning)**

Building a medical model from scratch requires millions of images. Instead, I used **MobileNetV2** (pre-trained on ImageNet) and "fine-tuned" it for X-ray textures. By freezing the early layers and training only the top medical-specific layers, I achieved high accuracy without needing a GPU.

### **2. The Deployment Pivot (Infrastructure)**

Initially, the project targeted a **Serverless (AWS Lambda)** deployment via **Docker**. However, due to low-level system library conflicts (`GLIBC` versioning) inherent in serverless environments for certain AI libraries, I strategically pivoted to a **Streamlit Web Application**. This ensured a robust, stable, and highly interactive deployment that is easier for medical staff to use.

---

## 💻 How to Run the App

1. **Install dependencies:**
```bash
pip install streamlit tensorflow-cpu pillow numpy

```


2. **Launch the application:**
```bash
streamlit run app.py

```


3. **Upload an image:** Use any `.jpeg` X-ray image from the `data/test` folder to see the AI in action.

---

## 🧬 Future Improvements

* **Multi-Class Classification:** Expand the model to distinguish between Bacterial and Viral pneumonia.
* **Heatmap Visualization:** Implement Grad-CAM to show doctors exactly which part of the lung the AI is looking at.
* **Cloud Hosting:** Deploy the Streamlit app to Streamlit Community Cloud or Heroku for public access.

---
