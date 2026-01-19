# app.py
import gradio as gr
import numpy as np

def predict_image(img):
    # 1. Resize the image to match model input
    img = tf.image.resize(img, (224, 224))
    
    # 2. Add batch dimension and rescale
    img_array = tf.expand_dims(img, 0) / 255.0
    
    # 3. Predict
    prediction = model.predict(img_array)[0]
    
    # 4. Map back to class names
    # Returns a dictionary like {"Fake": 0.25, "Real": 0.75}
    return {class_names[i]: float(prediction[i]) for i in range(2)}

# Test it locally with the Gradio Interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=2),
    title="Deepfake Detector",
    description="Upload an image to check if it is Real or a Deepfake."
)

interface.launch(share=True)