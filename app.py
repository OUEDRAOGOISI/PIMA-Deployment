from flask import Flask, request, jsonify
import joblib
import numpy as np

# Charger le modèle et le scaler
model = joblib.load('best_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Les données doivent être au format JSON
    features = np.array(data['features']).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0].tolist()

    return jsonify({
        'prediction': int(prediction),
        'probability': probability
    })

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Utilise le port fourni par Railway
    app.run(debug=True, host='0.0.0.0', port=port)
