from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import base64
from io import BytesIO

app = Flask(__name__)

# Load the trained cotton disease detection model
model_path = 'cotton_disease_CNN.h5' 
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Preprocess image using PIL
def preprocess_image(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image = image.resize((64, 64))  
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Predict image class
def predict_image(image_bytes):
    try:
        img_array = preprocess_image(image_bytes)
        if img_array is None:
            return None
        predictions = model.predict(img_array)
        class_labels = ["diseased cotton leaf", "diseased cotton plant", "fresh cotton leaf", "fresh cotton plant"]
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_class_index]
        return predicted_class
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None

# Flask routes
@app.route('/')
def new():
    return render_template('new.html')

@app.route('/upload')
def home():
    return render_template('home.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return "No file uploaded."

        filename = 'temp_image.jpg'
        file_path = os.path.join('uploads', filename)
        uploaded_file.save(file_path)
        print(f"Saved uploaded file to: {file_path}")

        with open(file_path, 'rb') as f:
            image_bytes = f.read()

        predicted_class = predict_image(image_bytes)
        if predicted_class is None:
            return "Prediction failed. Please check the input image."

        print(f"Predicted class: {predicted_class}")

        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return render_template('result.html', img_filename=filename, predicted_class=predicted_class, image=img_str)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)