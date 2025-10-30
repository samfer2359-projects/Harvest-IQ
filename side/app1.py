from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load trained model
model = tf.keras.models.load_model("best_plant_disease_model.h5")

# Class names from dataset
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

#Treatment suggestions dictionary
treatment_suggestions = {
    #Pepper Bell
    'Pepper__bell___Bacterial_spot': (
        "Remove infected leaves, avoid overhead irrigation, use copper-based sprays. \n\n"
        "Source: UC IPM (https://ipm.ucanr.edu/agriculture/peppers/bacterial-spot/?utm_source=chatgpt.com#gsc.tab=0)"
    ),
    'Pepper__bell___healthy': (
        "Plant appears healthy. Continue watering properly and maintain good air circulation."
    ),

    #Potato
    'Potato___Early_blight': (
        "Rotate crops, stake plants for airflow, mulch to prevent soil splash, and apply early fungicide. \n\n"
        "Source: Cornell University (https://www.vegetables.cornell.edu/pest-management/disease-factsheets/managing-tomato-diseases-successfully/)"
    ),
    'Potato___Late_blight': (
        "Destroy infected plants, avoid water on leaves, and apply recommended fungicides. \n\n"
        "Source: Cornell University (https://www.vegetables.cornell.edu/pest-management/disease-factsheets/managing-tomato-diseases-successfully/)"
    ),
    'Potato___healthy': (
        "Healthy plant! Maintain crop rotation and monitor regularly for blight symptoms."
    ),

    #Tomato
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
    'Tomato_healthy': (
        "Plant is healthy! Keep practicing crop rotation, watering at the base, and regular monitoring."
    )
}

#Prediction function
def predict_image(image_path):
    img = Image.open(image_path).resize((256, 256))
    img = np.array(img)

    # Remove alpha channel if present
    if img.shape[-1] == 4:
        img = img[:, :, :3]      #keep only RGB

    img = tf.keras.applications.vgg19.preprocess_input(img)   #process image
    img = np.expand_dims(img, axis=0)                         #add extra dimension because module expects a batch of image

    predictions = model.predict(img)
    predicted_index = np.argmax(predictions)
    predicted_label = class_names[predicted_index]
    suggestion = treatment_suggestions.get(predicted_label, "No treatment information available for this disease.")
    return predicted_label, suggestion




#Home Route
@app.route('/')
def index():
    return render_template('index.html', result='', suggestion='')


#Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', result="No file selected.", suggestion='')

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', result="No image provided.", suggestion='')

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    result, suggestion = predict_image(filepath)
    return render_template('index.html', result=result, suggestion=suggestion)


#Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
