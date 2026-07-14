"""Flask web application for heart disease risk prediction."""

import os
import json
import pickle
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Feature metadata — matches the 13 features in heart-statlog
# ---------------------------------------------------------------------------
FEATURES = [
    {"name": "age",      "label": "Age",                   "type": "number", "min": 20, "max": 100, "step": 1,   "default": 55},
    {"name": "sex",      "label": "Sex (1=Male, 0=Female)","type": "number", "min": 0,  "max": 1,   "step": 1,   "default": 1},
    {"name": "cp",       "label": "Chest Pain Type (1-4)", "type": "number", "min": 1,  "max": 4,   "step": 1,   "default": 1},
    {"name": "trestbps", "label": "Resting Blood Pressure","type": "number", "min": 80, "max": 200, "step": 1,   "default": 130},
    {"name": "chol",     "label": "Serum Cholesterol (mg/dl)","type": "number", "min": 100, "max": 600, "step": 1, "default": 250},
    {"name": "fbs",      "label": "Fasting Blood Sugar > 120 (1=Yes)","type": "number", "min": 0, "max": 1, "step": 1, "default": 0},
    {"name": "restecg",  "label": "Resting ECG (0-2)",     "type": "number", "min": 0,  "max": 2,   "step": 1,   "default": 0},
    {"name": "thalach",  "label": "Max Heart Rate Achieved","type": "number", "min": 60, "max": 220, "step": 1,  "default": 150},
    {"name": "exang",    "label": "Exercise Induced Angina (1=Yes)","type": "number", "min": 0, "max": 1, "step": 1, "default": 0},
    {"name": "oldpeak",  "label": "ST Depression (oldpeak)","type": "number", "min": 0.0,"max": 7.0, "step": 0.1, "default": 1.0},
    {"name": "slope",    "label": "Slope of Peak ST (1-3)","type": "number", "min": 1,  "max": 3,   "step": 1,   "default": 2},
    {"name": "ca",       "label": "Number of Major Vessels (0-3)","type": "number", "min": 0, "max": 3, "step": 1, "default": 0},
    {"name": "thal",     "label": "Thalassemia (3/6/7)",   "type": "number", "min": 3,  "max": 7,   "step": 1,   "default": 3},
]

# ---------------------------------------------------------------------------
# Available ONNX models
# ---------------------------------------------------------------------------
MODEL_OPTIONS = {
    "logreg":      {"path": "models/logreg.onnx",       "label": "Logistic Regression"},
    "tf_mlp":      {"path": "models/tf_mlp.onnx",       "label": "TensorFlow MLP"},
    "pytorch_mlp": {"path": "models/pytorch_mlp.onnx",  "label": "PyTorch MLP"},
}

# ---------------------------------------------------------------------------
# Load scaler + models at startup
# ---------------------------------------------------------------------------
scaler = None
sessions = {}


def load_artifacts():
    global scaler, sessions
    scaler_path = "models/scaler.pkl"
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    for key, meta in MODEL_OPTIONS.items():
        if os.path.exists(meta["path"]):
            sessions[key] = ort.InferenceSession(meta["path"])


def predict_single(model_key: str, features: list[float]) -> dict:
    """Run inference on a single sample with the chosen ONNX model."""
    if model_key not in sessions:
        return {"error": f"Model '{model_key}' not loaded. Run train.py first."}
    if scaler is None:
        return {"error": "Scaler not found. Run train.py first."}

    X = np.array([features], dtype=np.float32)
    X_scaled = scaler.transform(X).astype(np.float32)

    session = sessions[model_key]
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: X_scaled})
    raw = outputs[0]

    raw = np.array(raw)
    if raw.ndim == 2 and raw.shape[1] == 1:
        prob = float(1 / (1 + np.exp(-raw[0, 0])))  # sigmoid for logits
        if 0 <= raw[0, 0] <= 1:
            prob = float(raw[0, 0])  # already a probability
        pred = int(prob > 0.5)
    elif raw.ndim == 2 and raw.shape[1] > 1:
        pred = int(np.argmax(raw[0]))
        prob = float(raw[0, pred])
    else:
        pred = int(raw.flatten()[0])
        prob = float(pred)

    return {
        "prediction": pred,
        "label": "Heart Disease Detected" if pred == 1 else "No Heart Disease",
        "confidence": round(prob * 100 if pred == 1 else (1 - prob) * 100, 1),
        "risk_score": round(prob * 100, 1),
        "model": MODEL_OPTIONS[model_key]["label"],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    models_available = {k: v["label"] for k, v in MODEL_OPTIONS.items() if k in sessions}
    comparison = None
    comparison_path = "models/comparison.csv"
    if os.path.exists(comparison_path):
        import pandas as pd
        df = pd.read_csv(comparison_path)
        comparison = df.to_dict(orient="records")
    return render_template("index.html", features=FEATURES, models=models_available, comparison=comparison)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        model_key = data.get("model", "logreg")
        values = [float(data["features"][f["name"]]) for f in FEATURES]
        result = predict_single(model_key, values)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/models")
def api_models():
    return jsonify({k: v["label"] for k, v in MODEL_OPTIONS.items() if k in sessions})


if __name__ == "__main__":
    load_artifacts()
    print(f"\n  Models loaded: {list(sessions.keys())}")
    print(f"  Scaler loaded: {scaler is not None}\n")
    app.run(debug=True, port=5000)
