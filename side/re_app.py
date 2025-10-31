import os
import numpy as np
import pickle
from flask import Flask, render_template, request, redirect, url_for, flash
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a more secure key
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to save uploaded images

# Load models
plant_disease_model = load_model('best_plant_disease_model.h5')  # Path to the plant disease model
crop_recommendation_model = pickle.load(open('crop_recommendation_model.pkl', 'rb'))  # Path to the crop recommendation model

# Default login credentials (for now)
DEFAULT_USERNAME = 'admin'
DEFAULT_PASSWORD = 'password'

# Route for login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD:
            return redirect(url_for('welcome'))  # Redirect to the welcome page after successful login
        else:
            flash('Invalid credentials. Please try again.')
            return redirect(url_for('login'))  # Stay on the login page for failed login
    return render_template('login.html')  # Render the login form

# Welcome page after successful login
@app.route('/welcome')
def welcome():
    return render_template('welcome.html')  # Render the welcome page after login

# Home route that shows the main page after login
@app.route('/')
def index():
    return render_template('index.html')  # Main page where users can interact with the app

# Route for plant disease prediction
@app.route('/predict_plant_disease', methods=['POST'])
def predict_plant_disease():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Process the image for prediction
        img = image.load_img(file_path, target_size=(224, 224))  # Resize the image to match model input
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize the image (assuming model expects normalized input)

        # Make prediction
        prediction = plant_disease_model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)

        # Define disease classes and treatments
        disease_classes = ['Apple Scab', 'Black Rot', 'Cedar Apple Rust', 'Healthy']
        treatments = {
            'Apple Scab': 'Apply fungicide and prune infected leaves.',
            'Black Rot': 'Remove infected fruits and leaves. Use fungicide.',
            'Cedar Apple Rust': 'Remove infected leaves and use fungicide.',
            'Healthy': 'No treatment needed.'
        }

        # Map the prediction to a disease and treatment
        disease = disease_classes[predicted_class[0]]
        treatment = treatments[disease]

        # Render the result page with predicted disease and treatment
        return render_template('prediction_result.html', disease=disease, treatment=treatment)

# Route for crop recommendation based on input
@app.route('/recommend_crop', methods=['POST'])
def recommend_crop():
    if request.method == 'POST':
        # Get input values from the form
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        pH = float(request.form['ph'])

        # Prepare the input features array for the recommendation model
        input_features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, pH]])

        # Predict the recommended crop using the model
        recommended_crop = crop_recommendation_model.predict(input_features)
        recommended_crop_label = recommended_crop[0]  # Assuming the model returns a single prediction

        # Render the result page with the recommended crop
        return render_template('crop_recommendation_result.html', crop=recommended_crop_label)

# Run the Flask app
if __name__ == '__main__':
    # Create the upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # Run the application
    app.run(debug=True)
