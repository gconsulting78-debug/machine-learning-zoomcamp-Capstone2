# 🛡️ Deepfake Image Detection: A Transfer Learning Approach

## 📖 Project Overview
This project classifies images as "Real" or "Fake" (AI-generated) using a fine-tuned **EfficientNetB0** model. This was developed as a Capstone project for the MLZoomCamp.

### 🔗 Links
- **Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/gconsulting78-debug/Capstone2-Deepfake)
- **Model Weights:** Hosted on Hugging Face (automatically downloaded on run).

---

## 📊 Performance
The model achieved an **85% recall for 'Real' images**, making it highly reliable at verifying authentic content. Overall accuracy sits at **64%** on a diverse test set of 10,000 images.

---

## 💻 How to Run Locally

Follow these steps to set up the environment and launch the detector on your machine:

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
