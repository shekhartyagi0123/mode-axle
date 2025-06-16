from flask import Flask, request, jsonify
import traceback
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model
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

# ✅ Run with correct host and port
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
