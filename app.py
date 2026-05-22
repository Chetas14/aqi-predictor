from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load('aqi_model.pkl')

FEATURES = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']

# AQI Categories
AQI_CATEGORIES = [
    (0, 50, "Good", "#00e400", "Air quality is satisfactory. No health risks."),
    (51, 100, "Satisfactory", "#ffff00", "Air quality is acceptable."),
    (101, 200, "Moderate", "#ff7e00", "Sensitive groups may experience effects."),
    (201, 300, "Poor", "#ff0000", "Everyone may begin to experience health effects."),
    (301, 400, "Very Poor", "#99004c", "Health alert! Everyone may experience serious effects."),
    (401, 500, "Severe", "#7e0023", "Health warnings of emergency conditions.")
]

def get_category(aqi):
    for min_val, max_val, name, color, msg in AQI_CATEGORIES:
        if min_val <= aqi <= max_val:
            return name, color, msg
    return "Severe+", "#4a0015", "Emergency conditions!"

def get_recommendations(category):
    recs = {
        "Good": ["🌳 Enjoy outdoor activities", "🏃 Open windows for fresh air"],
        "Satisfactory": ["👍 Acceptable air quality", "🏠 Sensitive individuals should limit prolonged outdoor exertion"],
        "Moderate": ["⚠️ Unusually sensitive people should reduce outdoor activity", "🏡 Keep windows closed during high traffic hours"],
        "Poor": ["🚫 Reduce outdoor activities", "😷 Wear N95 masks when going outside", "🏠 Keep windows and doors closed"],
        "Very Poor": ["⛔ Avoid all outdoor activities", "😷 Wear N95 masks at all times outside", "🏠 Stay indoors with air purifiers"],
        "Severe": ["🚨 Emergency health alert!", "🏠 Do NOT go outside unless absolutely necessary", "💨 Run air purifiers continuously"]
    }
    return recs.get(category, ["Monitor air quality."])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    features = [
        float(data.get('pm25', 0)),
        float(data.get('pm10', 0)),
        float(data.get('no2', 0)),
        float(data.get('so2', 0)),
        float(data.get('co', 0)),
        float(data.get('o3', 0))
    ]
    
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)[0]
    aqi = round(prediction, 2)
    
    category, color, message = get_category(aqi)
    recommendations = get_recommendations(category)
    
    contributions = {}
    for i, f in enumerate(FEATURES):
        contributions[f] = round(model.coef_[i] * features[i], 2)
    
    return jsonify({
        'success': True,
        'aqi': aqi,
        'category': category,
        'color': color,
        'message': message,
        'recommendations': recommendations,
        'contributions': contributions
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)