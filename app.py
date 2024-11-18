from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import cv2
import pickle
from werkzeug.utils import secure_filename
# import tensorflow as tf
app = Flask(__name__,static_folder='static')

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
with open('model/dog_vs_cat_model.pkl', 'rb') as file:
    model = pickle.load(file)

#
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        image = cv2.imread(filepath)
        image_resized = cv2.resize(image, (224, 224))
        image_scaled = image_resized / 255.0
        image_reshaped = np.reshape(image_scaled, (1, 224, 224, 3))

        # Make a prediction
        prediction = model.predict(image_reshaped)
        predicted_label = np.argmax(prediction)

        label = 'Dog' if predicted_label == 1 else 'Cat'
        return render_template('result.html', label=label, image_url=filepath)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
