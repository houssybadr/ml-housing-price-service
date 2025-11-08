import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- import CORS
from model_loader import load_models
from dotenv import load_dotenv
from processing import preprocess
from regression_model import MyRidgeRegression



load_dotenv()
app = Flask(__name__)
FLASK_ENV = os.environ.get("FLASK_ENV", "development")
if FLASK_ENV == "development":
    allowed_origin = os.environ.get("FRONTEND_URL", "http://localhost:3000")
else:
    allowed_origin = os.environ.get("PROD_FRONTEND_URL")

CORS(app, origins=[allowed_origin])

model, scaler = load_models()
@app.route('/predict', methods=['POST'])
def predict():
    if not model and not scaler:
        return jsonify({"error": "model and scaler r not loaded"}), 500
    elif not model:
        return jsonify({"error": "model is not loaded"}), 500
    elif not scaler:
        return jsonify({"error": "scaler is not loaded"}), 500

    data_json = request.get_json()
    if not data_json:
        return jsonify({"error": "JSON empty."}), 400

    print(f"data received : {data_json}")

    processed_data = preprocess(data_json, scaler)
    if processed_data is None:
        return jsonify({"error": "couldnt preprocess data"}), 400

    try:
        prediction = model.predict(processed_data)
        output = float(prediction[0])
        print(f"prediction : {output}D")
        return jsonify({"prediction": output})
    except Exception as e:
        print(f"an error occurred while predicting: {e}")
        return jsonify({"error": "an error occurred while predicting"}), 500


@app.route('/', methods=['GET'])
def home():
    return "wassup"


if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 5000))
        app.run(debug=True, host='0.0.0.0', port=port)
    except ValueError:
        print("Erreur : PORT doit être un nombre valide")
        sys.exit(1)

