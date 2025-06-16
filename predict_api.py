from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load('axle_failure_model.pkl')

@app.route('/')
def home():
    return "✅ Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    voltage = data.get('voltage')
    current = data.get('current')

    if voltage is None or current is None:
        return jsonify({'error': 'Missing voltage or current'}), 400

    prediction = model.predict([[voltage, current]])
    result = 'Failure Predicted' if prediction[0] == 1 else 'Normal'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
