# app.py

import gradio as gr
import tensorflow as tf
import numpy as np
import os
from huggingface_hub import hf_hub_download

# 1. DOWNLOAD THE MODEL FROM HUGGING FACE
# This avoids the GitHub 100MB limit issue
REPO_ID = "gconsulting78-debug/Capstone2-Deepfake"
FILENAME = "best_model_finetuned.keras"

print("Downloading model from Hugging Face...")
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="space")

# 2. LOAD THE MODEL
custom_objects = {"BatchNormalization": tf.keras.layers.BatchNormalization}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

def predict_image(img):
    img = tf.image.resize(img, (224, 224))
    img_array = tf.expand_dims(img, 0) / 255.0
    prediction = model.predict(img_array)[0]
    return {"Fake": float(prediction[0]), "Real": float(prediction[1])}

interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=2),
    title="🛡️ Deepfake Detector"
)

if __name__ == "__main__":
    interface.launch(ssr_mode=False)
