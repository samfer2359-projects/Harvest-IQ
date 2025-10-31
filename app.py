from flask import Flask, render_template, request, redirect, url_for, session 
import os
import joblib
import numpy as np
from keras.models import load_model
from PIL import Image
from io import BytesIO
import tensorflow as tf
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib.pyplot as plt  # <-- Add this import for plotting
import xarray as xr



app = Flask(__name__)

import secrets
app.secret_key = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(16))

# Load the crop recommendation model
crop_model = joblib.load('crop_recommendation_model.pkl')

# Load the plant disease model (VGG19)
disease_model = load_model('best_plant_disease_model.h5')

# Directory to save uploaded files
UPLOAD_FOLDER = 'static'  # Save to static folder for easy access via URL
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#Home Route
@app.route('/')
def index():
    return render_template('index.html', result='', suggestion='')



import main_sat  # Import your script that computes NDVI

@app.route('/ndvi', methods=['GET', 'POST'])
def ndvi():
    if request.method == 'POST':
        try:
            # Call your main_sat logic to get NDVI paths
            ndvi_paths = main_sat.get_ndvi_paths()  # This should return a dictionary with paths

            # Render the template with paths to the NDVI plot and NetCDF file
            return render_template(
                'ndvi.html',
                ndvi_result="✅ NDVI calculation complete!",
                ndvi_image_path=ndvi_paths["ndvi_plot"],  # Path to NDVI plot
                ndvi_netcdf_path=ndvi_paths["ndvi_netcdf"]  # Path to NDVI NetCDF
            )
        except Exception as e:
            return f"Error processing NDVI: {e}", 500

    return render_template('ndvi.html')






@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username and password == '1234':  # simple demo check
            # Store username in session
            session['username'] = username
            return redirect(url_for('welcome', username=username))
        else:
            return render_template('login.html', error="Invalid username or password")

    return render_template('login.html')

@app.route('/welcome')
def welcome():
    # Fetch username from session
    username = session.get('username', None)  # None is the default value if username isn't found
    if username is None:
        return redirect(url_for('login'))  # If the user is not logged in, redirect to login
    return render_template('welcome.html', username=username)

@app.route('/predict_plant', methods=['GET', 'POST'])
def predict_plant():
    if request.method == 'POST':
        # Handle image upload
        image_file = request.files.get('image')
        if not image_file or image_file.filename == '':
            return render_template('predict_plant.html', result=None, suggestion=None, error="No file uploaded.")

        filename = secure_filename(image_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(filepath)

        # --- Preprocess image for VGG19 ---
        try:
            img = Image.open(filepath)
            # Convert to RGB (handles grayscale / RGBA)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((256, 256))   # Model expects 256x256 input size

            img_array = np.array(img).astype('float32')
        except Exception as e:
            return render_template('predict_plant.html', result=None, suggestion=None, error=f"Image processing error: {e}")

            #Treatment suggestions dictionary
            
        # Expand dims and apply VGG19 preprocessing
        img_array = np.expand_dims(img_array, axis=0)
        try:
            from tensorflow.keras.applications.vgg19 import preprocess_input
        except Exception:
            # fallback if import path differs
            from tensorflow.keras.applications.vgg19 import preprocess_input

        img_array = preprocess_input(img_array)  # important for VGG19

        # Predict using the disease model
        preds = disease_model.predict(img_array)

        # Convert to probabilities (in case model returns logits)
        try:
            probs = tf.nn.softmax(preds[0]).numpy()
        except Exception:
            # if preds is already probabilities or single-dim
            probs = np.array(preds[0])
            # normalize as a safeguard
            probs = probs / (probs.sum() + 1e-12)

        # Get top prediction(s)
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])

        # Map the class to a label (ensure mapping covers indices)
        ref = {
            0: 'Pepper__bell___Bacterial_spot',
            1: 'Pepper__bell___healthy',
            2: 'Potato___Early_blight',
            3: 'Potato___Late_blight',
            4: 'Potato___healthy',
            5: 'Tomato_Bacterial_spot',
            6: 'Tomato_Early_blight',
            7: 'Tomato_Late_blight',
            8: 'Tomato_Leaf_Mold',
            9: 'Tomato_Septoria_leaf_spot',
            10: 'Tomato_Spider_mites_Two_spotted_spider_mite',
            11: 'Tomato__Target_Spot',
            12: 'Tomato__Tomato_YellowLeaf__Curl_Virus',
            13: 'Tomato__Tomato_mosaic_virus',
            14: 'Tomato_healthy'
        }

        disease_prediction = ref.get(top_idx, 'Unknown Disease')

        # Disease Treatment Suggestions (same mapping)
        treatment_suggestions = {
            'Pepper__bell___Bacterial_spot': "Apply bactericides and remove infected leaves.",
            'Pepper__bell___healthy': "No treatment needed. Keep the plant healthy with proper care.",
            'Potato___Early_blight': "Use fungicides and remove affected leaves.",
            'Potato___Late_blight': "Spray fungicides and remove infected plant parts.",
            'Potato___healthy': "No treatment required. Ensure well-draining soil.",
            'Tomato_Bacterial_spot': "Use copper-based fungicides and prune affected areas.",
            'Tomato_Early_blight': "Apply fungicide and remove damaged leaves.",
            'Tomato_Late_blight': "Spray fungicides and discard infected plant material.",
            'Tomato_Leaf_Mold': "Increase airflow and use fungicides to prevent further spread.",
            'Tomato_Septoria_leaf_spot': "Remove infected leaves and apply appropriate fungicides.",
            'Tomato_Spider_mites_Two_spotted_spider_mite': "Use insecticides or natural predators to control spider mites.",
            'Tomato__Target_Spot': "Apply fungicide and remove affected leaves.",
            'Tomato__Tomato_YellowLeaf__Curl_Virus': "No cure, but you can control vector spread using insecticides.",
            'Tomato__Tomato_mosaic_virus': "No cure. Remove infected plants to prevent virus spread.",
            'Tomato_healthy': "No treatment required. Keep the plant healthy with balanced care."
        }

        treatment_suggestion = treatment_suggestions.get(disease_prediction, "Unknown treatment. Consult a specialist.")

        # Also prepare top-3 suggestions for debugging / display if desired
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3 = [(int(i), ref.get(int(i), 'Unknown'), float(probs[int(i)])) for i in top3_idx]

        # Pass prediction, confidence and optionally the top-3 to the template
        return render_template(
            'predict_plant.html',
            result=disease_prediction,
            suggestion=treatment_suggestion,
            confidence=f"{top_prob*100:.2f}%",
            top3=top3
        )

    return render_template('predict_plant.html', result=None, suggestion=None)


@app.route('/recommend_crop', methods=['GET', 'POST'])
def recommend_crop():
    if request.method == 'POST':
        # Extract form data
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Prepare input features for crop recommendation
        input_features = [N, P, K, temperature, humidity, ph, rainfall]

        # Get crop recommendation from model
        recommended_crop = crop_model.predict([input_features])[0]

        return render_template('recommend_crop.html', crop_result=recommended_crop)

    return render_template('recommend_crop.html', crop_result=None)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
