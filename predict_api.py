from flask import Flask, request, jsonify
import joblib, os

app = Flask(__name__)
model = joblib.load('axle_failure_model.pkl')

@app.route('/')
def home():
    return "✅ Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON payload'}), 400

    try:
        voltage = float(data.get('voltage'))
        current = float(data.get('current'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid voltage or current'}), 422

    features = [[voltage, current]]
    try:
        pred = model.predict(features)[0]
    except Exception as e:
        return jsonify({'error': 'Model prediction failed', 'details': str(e)}), 500

    return jsonify({'prediction': 'Failure Predicted' if pred == 1 else 'Normal'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
