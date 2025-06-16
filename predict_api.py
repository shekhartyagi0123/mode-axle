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

        voltage = float(data.get('voltage'))
        current = float(data.get('current'))

        if voltage is None or current is None:
            return jsonify({'error': 'Missing voltage or current'}), 400

        # For API input, simulate a short 3-second data window using same values
        df = pd.DataFrame([{
            'Voltage': voltage,
            'Current': current
        }] * 3)  # Repeat 3 times to mimic a 3s window

        # Feature engineering (same as training)
        df['voltage_avg_3s'] = df['Voltage'].rolling(window=3).mean()
        df['voltage_diff'] = df['Voltage'].diff()
        df['current_avg_3s'] = df['Current'].rolling(window=3).mean()
        df['current_diff'] = df['Current'].diff()

        # Take the last row (with all features filled)
        latest_features = df.dropna().iloc[-1][['voltage_avg_3s', 'voltage_diff', 'current_avg_3s', 'current_diff']]
        input_df = pd.DataFrame([latest_features])

        prediction = model.predict(input_df)[0]

        return jsonify({'prediction': 'Failure' if prediction == 1 else 'Normal'})

    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e),
            'trace': traceback.format_exc()
        }), 500

# Run the server
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
