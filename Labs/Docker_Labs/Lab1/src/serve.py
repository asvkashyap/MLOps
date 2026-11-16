"""
Serves the trained Wine classifier over HTTP.

GET  /health   -> {"status": "ok"}; used by the Dockerfile's HEALTHCHECK
GET  /         -> plain welcome message
POST /predict  -> body: {"features": [13 numbers, in data/feature_order.json order]}
                  returns the predicted class and the model's class probabilities
"""
import json
import os

import joblib
import numpy as np
from flask import Flask, jsonify, request

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")

app = Flask(__name__)

model = joblib.load(os.path.join(MODEL_DIR, "wine_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
    metadata = json.load(f)

FEATURE_NAMES = metadata["feature_names"]
CLASS_NAMES = metadata["class_names"]


@app.route("/")
def home():
    return (
        f"Wine Classifier API. Trained {metadata['trained_at']}, "
        f"test accuracy {metadata['test_accuracy']}. POST to /predict."
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_trained_at": metadata["trained_at"]})


@app.route("/metadata")
def get_metadata():
    return jsonify(metadata)


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True, silent=True)
    if not body or "features" not in body:
        return jsonify({"error": f"expected JSON body with a 'features' list of "
                                  f"{len(FEATURE_NAMES)} numbers: {FEATURE_NAMES}"}), 400

    features = body["features"]
    if len(features) != len(FEATURE_NAMES):
        return jsonify({"error": f"expected {len(FEATURE_NAMES)} features, got {len(features)}"}), 400

    try:
        x = np.array(features, dtype=float).reshape(1, -1)
    except (TypeError, ValueError):
        return jsonify({"error": "all features must be numbers"}), 400

    x_scaled = scaler.transform(x)
    probabilities = model.predict_proba(x_scaled)[0]
    predicted_index = int(np.argmax(probabilities))

    return jsonify({
        "predicted_class": CLASS_NAMES[predicted_index],
        "probabilities": {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probabilities)},
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
