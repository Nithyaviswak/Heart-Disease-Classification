"""Phase 9: /predict endpoint with SHAP explainability."""

import json
import numpy as np
from flask import Blueprint, request, jsonify, g

from backend.auth import token_required
from backend.database import get_connection

predict_bp = Blueprint("predict", __name__)

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


def _get_ml_service():
    from backend.ml.models import MLService
    return MLService(models_dir="models")


@predict_bp.route("/api/predict", methods=["POST"])
@token_required
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body is required"}), 400

    model_name = data.get("model", "Random Forest")
    feature_values = data.get("features", {})

    try:
        features = np.array([float(feature_values.get(f, 0)) for f in FEATURE_NAMES], dtype=np.float32)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid feature values: {e}"}), 400

    ml = _get_ml_service()
    result = ml.predict(model_name, features)

    if "error" in result:
        return jsonify(result), 500

    if "shap" in result and "error" not in result.get("shap", {}):
        shap_json = json.dumps(result["shap"])
    else:
        shap_json = None

    try:
        user_id = g.current_user.get("user_id") if hasattr(g, "current_user") else None
    except Exception:
        user_id = None

    with get_connection() as conn:
        conn.execute(
            """INSERT INTO predictions (user_id, model_name, features, prediction, probability, confidence, shap_json, risk_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                result["model"],
                json.dumps(feature_values),
                result["prediction"],
                result["probability"],
                result["confidence"],
                shap_json,
                result["risk_score"],
            ),
        )

    return jsonify(result)


@predict_bp.route("/api/predict/explain", methods=["POST"])
@token_required
def predict_explain():
    """Get prediction with SHAP explanation for all models."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body is required"}), 400

    feature_values = data.get("features", {})

    try:
        features = np.array([float(feature_values.get(f, 0)) for f in FEATURE_NAMES], dtype=np.float32)
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400

    ml = _get_ml_service()
    results = ml.predict_all(features)
    return jsonify(results)
