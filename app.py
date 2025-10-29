from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import joblib
import geemap
import stackstac
import xarray as xr

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load trained model for plant disease prediction
model = tf.keras.models.load_model("best_plant_disease_model.h5")

# Class names from dataset for plant diseases
class_names = [
    'Pepper__bell___Bacterial_spot', 
    'Pepper__bell___healthy', 
    'Potato___Early_blight', 
    'Potato___Late_blight', 
    'Potato___healthy', 
    'Tomato_Bacterial_spot', 
    'Tomato_Early_blight', 
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 
    'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 
    'Tomato__Target_Spot', 
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 
    'Tomato__Tomato_mosaic_virus', 
    'Tomato_healthy'
]

# Treatment suggestions dictionary for plant diseases
treatment_suggestions = {
    'Pepper__bell___Bacterial_spot': (
        "Remove infected leaves, avoid overhead irrigation, use copper-based sprays. \n\n"
        "Source: UC IPM (https://ipm.ucanr.edu/agriculture/peppers/bacterial-spot/?utm_source=chatgpt.com#gsc.tab=0)"
    ),
    'Pepper__bell___healthy': "Plant appears healthy. Continue watering properly and maintain good air circulation.",
    'Potato___Early_blight': (
        "Rotate crops, stake plants for airflow, mulch to prevent soil splash, and apply early fungicide. \n\n"
        "Source: Cornell University (https://www.vegetables.cornell.edu/pest-management/disease-factsheets/managing-tomato-diseases-successfully/)"
    ),
    'Potato___Late_blight': (
        "Destroy infected plants, avoid water on leaves, and apply recommended fungicides. \n\n"
        "Source: Cornell University (https://www.vegetables.cornell.edu/pest-management/disease-factsheets/managing-tomato-diseases-successfully/)"
    ),
    'Potato___healthy': "Healthy plant! Maintain crop rotation and monitor regularly for blight symptoms.",
    'Tomato_Bacterial_spot': (
        "Use certified disease-free seeds, avoid overhead watering, and spray copper fungicides. \n\n"
        "Source: UC IPM (https://ipm.ucanr.edu/agriculture/tomato/bacterial-spot/)"
    ),
    'Tomato_Early_blight': (
        "Remove lower infected leaves, stake plants, rotate crops, and start fungicide early. \n\n"
        "Source: Cornell University (https://www.vegetables.cornell.edu/pest-management/disease-factsheets/managing-tomato-diseases-successfully/)"
    ),
    'Tomato_Late_blight': (
        "Destroy infected plants immediately, use fungicides in cool humid conditions, and rotate crops. \n\n"
        "Source: Cornell University (https://www.vegetables.cornell.edu/pest-management/disease-factsheets/managing-tomato-diseases-successfully/)"
    ),
    'Tomato_Leaf_Mold': (
        "Reduce humidity, increase ventilation, prune lower leaves, and apply fungicides if needed. \n\n"
        "Source: University of Minnesota Extension (https://extension.umn.edu/plant-diseases/tomato-leaf-spot-diseases)"
    ),
    'Tomato_Septoria_leaf_spot': (
        "Remove infected leaves, mulch to avoid soil splash, avoid dense planting, and use fungicide early. \n\n"
        "Source: University of Minnesota Extension (https://extension.umn.edu/plant-diseases/tomato-leaf-spot-diseases)"
    ),
    'Tomato_Spider_mites_Two_spotted_spider_mite': (
        "Spray water to reduce dust, apply insecticidal soap or neem oil, and increase humidity. \n\n"
        "Source: University of Maryland (https://www.extension.umd.edu/resource/expect-see-two-spotted-spider-mites-vegetables/?utm_source=chatgpt.com)"
    ),
    'Tomato__Target_Spot': (
        "Improve airflow, rotate crops, and apply fungicide at early symptoms. \n\n"
        "Source: Bayer Crop Science (https://www.vegetables.bayer.com/us/en-us/resources/growing-tips-and-innovation-articles/agronomic-spotlights/target-spot-of-tomato.html)"
    ),
    'Tomato__Tomato_YellowLeaf__Curl_Virus': (
        "Control whiteflies (the insect vector), remove infected plants, and use resistant varieties. \n\n"
        "Source: North Carolina State University (https://content.ces.ncsu.edu/tomato-yellow-leaf-curl-virus?utm_source=chatgpt.com)"
    ),
    'Tomato__Tomato_mosaic_virus': (
        "Avoid tobacco handling near tomatoes, disinfect tools, and remove infected plants immediately. \n\n"
        "Source: UC IPM (https://ipm.ucanr.edu/agriculture/tomato/mosaic-diseases-caused-by-potyviruses/?utm_source=chatgpt.com#gsc.tab=0)"
    ),
    'Tomato_healthy': "Plant is healthy! Keep practicing crop rotation, watering at the base, and regular monitoring."
}

# ---------------------------
# 1️⃣ Crop Recommendation Model Section
# ---------------------------

# Load the pre-trained crop recommendation model
crop_model = joblib.load('crop_recommendation_model.pkl')

# Crop recommendation function based on soil and weather conditions
def recommend_crop(soil_data):
    try:
        soil_data = np.array([soil_data])  # Ensure it's in the right shape for prediction
        return crop_model.predict(soil_data)[0]  # Make the prediction
    except Exception as e:
        return f"Error in crop recommendation: {e}"

# ---------------------------
# 2️⃣ NDVI (Satellite) Model Section
# ---------------------------

# NDVI processing function using satellite images (Sentinel-2)
def process_ndvi(bbox, time_range):
    try:
        # Load and search Sentinel-2 data (connect to STAC API)
        catalog = geemap.pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=time_range,
            query={"eo:cloud_cover": {"lt": 20}}  # Filter for less than 20% cloud cover
        )

        items = list(search.get_items())
        if len(items) == 0:
            return "No images found in this region and date range."

        # Stack Sentinel-2 imagery (Red and NIR bands for NDVI calculation)
        ds = stackstac.stack(
            items,
            assets=["B04", "B08"],  # Red (B04) and NIR (B08)
            bounds_latlon=bbox,
            epsg=4326,               # WGS84 CRS
            resolution=0.00025
        )

        # Compute NDVI
        red = ds.sel(band="B04")
        nir = ds.sel(band="B08")
        ndvi = (nir - red) / (nir + red)
        return ndvi.median(dim="time")  # Return the median NDVI over time
    except Exception as e:
        return f"Error in NDVI processing: {e}"

# ---------------------------
# Prediction function for plant disease
# ---------------------------
def predict_image(image_path):
    img = Image.open(image_path).resize((256, 256))
    img = np.array(img)

    # Remove alpha channel if present
    if img.shape[-1] == 4:
        img = img[:, :, :3]  # Keep only RGB

    img = tf.keras.applications.vgg19.preprocess_input(img)  # Process image
    img = np.expand_dims(img, axis=0)  # Add extra dimension because model expects a batch of images

    predictions = model.predict(img)
    predicted_index = np.argmax(predictions)
    predicted_label = class_names[predicted_index]
    suggestion = treatment_suggestions.get(predicted_label, "No treatment information available for this disease.")

    return predicted_label, suggestion


# ---------------------------
# Routes for web pages
# ---------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_plant', methods=['GET', 'POST'])
def predict_plant():
    result = suggestion = ''
    
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            result, suggestion = predict_image(filepath)

    return render_template('predict_plant.html', result=result, suggestion=suggestion)

@app.route('/recommend_crop', methods=['GET', 'POST'])
def recommend_crop_route():
    crop_result = ''
    
    if request.method == 'POST':
        try:
            # Collecting all the form data
            nitrogen = request.form.get('N')
            phosphorus = request.form.get('P')
            potassium = request.form.get('K')
            temperature = request.form.get('temperature')
            humidity = request.form.get('humidity')
            ph = request.form.get('ph')
            rainfall = request.form.get('rainfall')

            # Validate the form fields to make sure they're all filled out
            if not all([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]):
                crop_result = "Error: All fields are required!"
            else:
                # Convert the inputs to floats (if they're valid)
                soil_data = [
                    float(nitrogen),
                    float(phosphorus),
                    float(potassium),
                    float(temperature),
                    float(humidity),
                    float(ph),
                    float(rainfall)
                ]
                
                # Call the crop recommendation function
                crop_result = recommend_crop(soil_data)
                
        except Exception as e:
            crop_result = f"Error processing crop recommendation: {e}"

    return render_template('recommend_crop.html', crop_result=crop_result)


@app.route('/ndvi', methods=['GET', 'POST'])
def ndvi_route():
    ndvi_result = ''
    
    if request.method == 'POST':
        try:
            # Collect bounding box (bbox) and time range from form
            lon_min = float(request.form.get('lon_min'))
            lat_min = float(request.form.get('lat_min'))
            lon_max = float(request.form.get('lon_max'))
            lat_max = float(request.form.get('lat_max'))
            time_range = request.form.get('time_range')

            # Process the NDVI (make sure process_ndvi is properly defined in your app)
            ndvi_result = process_ndvi([lon_min, lat_min, lon_max, lat_max], time_range)
        
        except Exception as e:
            ndvi_result = f"Error processing NDVI data: {e}"

    return render_template('ndvi.html', ndvi_result=ndvi_result)



if __name__ == '__main__':
    app.run(debug=True)
