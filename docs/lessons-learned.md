# HarvestIQ – Lessons Learned

---

## Challenge: Tight Coupling Between ML Models and Flask Routes

### Problem

ML logic is directly embedded inside Flask routes, which caused:

* Routes cannot be reused outside Flask
* Testing ML logic independently is difficult
* Any ML change requires modifying `app.py`

---

### Supporting Code

### File: `app.py`

```python
@app.route('/fertilizer', methods=['GET', 'POST'])
def fertilizer():
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    ph = float(request.form['ph'])
    moisture = float(request.form['moisture'])

    status, recommendation = recommend_fertilizer(N, P, K, ph, moisture)

    return render_template('fertilizer.html', result=result)
```

---

### Lesson Learned

* Flask routes should act like connectors, not processors
* Separating ML logic makes the system reusable and testable

---

## Challenge: Image Upload Causing Prediction Errors

### Problem

While building the plant disease detection feature, uploaded images did not always work correctly with the model.

Some images caused errors or wrong predictions because:

* Images came in different formats (JPG, PNG, etc.)
* Some images were not RGB (grayscale images)
* Some images had different sizes than what the model expected

Because of this, the model could not always process the image correctly on the first try.

---

### Supporting Code

### File: `app.py`

```python
image_file = request.files.get('image')

img = Image.open(filepath)

if img.mode != 'RGB':
    img = img.convert('RGB')

img = img.resize((256, 256))

img_array = np.array(img).astype('float32')
img_array = np.expand_dims(img_array, axis=0)
```

---

### Lesson Learned

* Image input is not always clean or uniform
* Models require fixed input format to work correctly
* Real-world user uploads are more complex than dataset images

---

## Challenge: Slow Application Startup Due to Heavy Model Loading

### Problem

When running the Flask application, startup was slow because large ML models were loaded directly in `app.py`.

This included:

* Crop recommendation model
* Plant disease detection model (VGG19)

Since these models are large, every restart increased loading time.

---

### Supporting Code

### File: `app.py`

```python
crop_model = joblib.load('crop_recommendation_model.pkl')
disease_model = load_model('best_plant_disease_model.h5')
```

---

### Lesson Learned

* Large ML models slow down application startup
* Loading heavy models at runtime affects development speed
* Model initialization should be handled carefully in ML applications

