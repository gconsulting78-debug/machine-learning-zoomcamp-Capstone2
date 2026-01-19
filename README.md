# 🛡️ Deepfake Image Detection: A Transfer Learning Approach

## 📖 Project Overview
This project focuses on the binary classification of images to detect AI-generated "Deepfakes." As AI generation becomes more sophisticated, the ability to distinguish between authentic and synthetic faces is critical for digital security and media integrity. 

This implementation uses a deep convolutional neural network (**EfficientNetB0**) to identify subtle artifacts in digital faces that are often invisible to the human eye.

### 🔗 Project Links
- **Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/gconsulting78-debug/Capstone2-Deepfake)
- **Training Environment:** Google Colab
- **Dataset Source:** [[Kaggle - Deep Fake Detection](https://www.kaggle.com/code/kameshrasu/deep-fake-detection-with-efficientnet/input]

---

### 1. Data & Inspiration
The model was trained on a comprehensive dataset of real and fake facial images. The primary objective was to leverage the efficiency of the EfficientNet family of models to create a lightweight yet powerful detector capable of running in real-time environments.

### 2. Architecture: EfficientNetB0
I implemented a **Two-Stage Transfer Learning** strategy:
- **Stage 1: Feature Extraction:** The base EfficientNetB0 model was frozen, and a custom dense "head" was trained to map features to our binary classes.
- **Stage 2: Fine-Tuning:** The top 30 layers were unfrozen and trained with a very low learning rate ($1 \times 10^{-4}$) to capture domain-specific textures in synthetic skin and eyes.

---

## 📊 Performance
The model was evaluated on a test set of ~10,000 images, achieving an overall accuracy of **64%**. Notably, it achieved an **85% recall for 'Real' images**, making it a reliable tool for verifying authentic content.

---

## 💻 How to Run Locally

Follow these steps to set up the environment and launch the detector:

```bash
# 1. Clone the repository
git clone [https://github.com/YOUR_USERNAME/Capstone2-Deepfake.git](https://github.com/YOUR_USERNAME/Capstone2-Deepfake.git)
cd Capstone2-Deepfake

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies and launch the app
pip install -r requirements.txt
python app.py
