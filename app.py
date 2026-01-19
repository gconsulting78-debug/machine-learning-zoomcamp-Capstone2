# app.py

import gradio as gr
import tensorflow as tf
import numpy as np
import os
from huggingface_hub import hf_hub_download

# 1. DOWNLOAD THE MODEL FROM HUGGING FACE
# This keeps your GitHub repo small and avoids the 100MB limit.
REPO_ID = "gconsulting78-debug/Capstone2-Deepfake"
FILENAME = "best_model_finetuned.keras"

print("Downloading model from Hugging Face...")
# This downloads the file to a cache folder and returns the local path
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="space")

# 2. LOAD THE MODEL
# We use custom_objects and safe_mode=False to bypass Keras 3.11+ loading bugs
custom_objects = {"BatchNormalization": tf.keras.layers.BatchNormalization}
model = tf.keras.models.load_model(
    model_path, 
    custom_objects=custom_objects, 
    compile=False,
    safe_mode=False
)

def predict_image(img):
    # Preprocess the image to match EfficientNetB0 requirements
    img = tf.image.resize(img, (224, 224))
    img_array = tf.expand_dims(img, 0) / 255.0
    
    # Run prediction
    prediction = model.predict(img_array)[0]
    
    # Return as a dictionary for Gradio Label component
    return {"Fake": float(prediction[0]), "Real": float(prediction[1])}

# 3. CREATE GRADIO INTERFACE
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=2),
    title="🛡️ Deepfake Detector",
    description="Upload a face image to check if it is Real or AI-generated."
)

if __name__ == "__main__":
    # ssr_mode=False prevents the _DictWrapper error on Hugging Face
    interface.launch(ssr_mode=False)
