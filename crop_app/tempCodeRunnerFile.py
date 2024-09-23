from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = load_model('crop_recommendation_model.h5')
scaler = joblib.load('scaler.pkl')  # Assuming you saved the scaler earlier
label_encoder = joblib.load('label_encoder.pkl')

# List of plant types for encoding
plant_types = ['Wheat', 'Rice', 'Papaya', 'Barley', ...]  # Add the correct classes here

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        plant_type = request.form['plant_type']

        # Scale the feature inputs
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        features_scaled = scaler.transform(features)

        # One-hot encode the plant type
        plant_one_hot = label_encoder.transform([plant_type])

        # Combine features and plant type

        # Predict probabilities
        probabilities = model.predict(features_scaled)

        # Render the result page with predicted probabilities
        print(probabilities[0][plant_one_hot])
        return render_template('index.html', probs=probabilities[0][plant_one_hot], plant_types=plant_types)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)