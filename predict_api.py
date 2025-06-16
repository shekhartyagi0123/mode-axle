from flask import Flask, request, jsonify
import traceback
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("axle_failure_model.pkl")

@app.route('/')
def home():
    return "✅ Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        voltage = data.get('voltage')
        current = data.get('current')

        # Check inputs
        if voltage is None or current is None:
            return jsonify({'error': 'Missing voltage or current'}), 400

        input_df = pd.DataFrame([[voltage, current]], columns=['Voltage', 'Current'])

        prediction = model.predict(input_df)[0]

        return jsonify({'prediction': 'Failure' if prediction == 1 else 'Normal'})

    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e),
            'trace': traceback.format_exc()
        }), 500
