from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the model
model = pickle.load(open('model1.pkl', 'rb'))

# Load the scalers
scaler_standard = pickle.load(open('standscaler1.pkl', 'rb'))
scaler_minmax = pickle.load(open('minmaxscaler1.pkl', 'rb'))

# Create Flask app
app = Flask(__name__)

# Define the index route
@app.route('/')
def index():
    return render_template("index.html")

# Define the prediction route
@app.route("/predict", methods=['POST'])
def predict():
    # Get the input values from the form
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temperature = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    pH = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    # Scale the input features
    features = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
    features_standardized = scaler_standard.transform(features)
    features_scaled = scaler_minmax.transform(features_standardized)

    # Make prediction
    prediction = model.predict(features_scaled)

    # Define crop dictionary
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    # Get the recommended crop
    if prediction[0] in crop_dict:
        recommended_crop = crop_dict[prediction[0]]
        result = f"{recommended_crop} is the best crop to be cultivated right there"
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    # Render the result in the index.html template
    return render_template('index.html', result=result)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
