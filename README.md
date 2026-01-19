# machine-learning-zoomcamp-Capstone2
Sandy's ML Zoomcamp Capstone 2 - Deepfake detection tool
# 🛡️ Deepfake Image Detection: A Transfer Learning Approach

## 📖 Project Overview
This project focuses on the binary classification of images to detect AI-generated "Deepfakes." As AI generation becomes more accessible, distinguishing between authentic and synthetic faces is critical. This model uses a deep convolutional neural network to identify subtle artifacts in digital faces.

### 🔗 Links
- **Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/gconsulting78-debug/Capstone2-Deepfake)
- **Training Environment:** Google Colab

---

## 🔬 Methodology

### 1. Architecture: EfficientNetB0
I chose **EfficientNetB0** for its high parameter efficiency and robust performance in facial feature extraction. The model was built using a **Two-Stage Transfer Learning** approach:

- **Stage 1: Feature Extraction:** The base EfficientNetB0 model was frozen, and a custom dense "head" was trained (GlobalAveragePooling2D → Dropout (0.5) → Dense (2, Softmax)).
- **Stage 2: Fine-Tuning:** The top 30 layers of the base model were unfrozen and trained with a significantly lower learning rate ($1 \times 10^{-4}$) to adapt the pre-trained weights to the specific domain of AI-generated artifacts.



### 2. Data Processing & Augmentation
To ensure the model generalized well and didn't simply memorize the dataset, I implemented:
- **Rescaling:** Pixel values normalized to $[0, 1]$.
- **Augmentation:** Random horizontal flips, rotations (0.1 factor), and zoom (0.1 factor).
- **Target Size:** $224 \times 224$ pixels.

---

## 📊 Performance Evaluation
The model was tested against a held-out test set of ~10,000 images.

| Metric | Class: Fake | Class: Real | Overall Accuracy |
| :--- | :--- | :--- | :--- |
| **Precision** | 0.75 | 0.59 | **64%** |
| **Recall** | 0.42 | 0.85 | |
| **F1-Score** | 0.54 | 0.70 | |

**Key Finding:** The model is exceptionally strong at identifying **Real** images (85% recall). While it is more conservative in flagging "Fakes," it provides a high-confidence signal when a fake is detected.

---

## 🛠️ Technical Challenges & Solutions
During the deployment phase to Hugging Face, a critical versioning issue was encountered involving **Keras 3** and **BatchNormalization** layers. 

- **The Problem:** Newer Keras versions triggered a `ValueError` regarding input tensors ($1$ expected, $2$ received) when loading the saved model.
- **The Solution:** 1. Downgraded the environment to **Python 3.12**.
  2. Pinned dependencies to **TensorFlow 2.18.0** and **Keras 3.10.0**.
  3. Implemented a `custom_objects` dictionary in the inference script to handle layer loading and set `compile=False` during `load_model` to bypass unnecessary training-depth checks.

---

## 💻 How to Reproduce

### 1. Clone the Repo
```bash
git clone [https://github.com/YOUR_USERNAME/Capstone2-Deepfake-Detection.git](https://github.com/YOUR_USERNAME/Capstone2-Deepfake-Detection.git)
cd Capstone2-Deepfake-Detection
