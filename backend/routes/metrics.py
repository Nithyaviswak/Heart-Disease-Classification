"""Phase 2 & 9: Model metrics API."""

import os
import json
from flask import Blueprint, jsonify

from backend.auth import token_required

model_metrics_bp = Blueprint("model_metrics", __name__)


@model_metrics_bp.route("/api/model-metrics")
@token_required
def get_metrics():
    path = "models/model_metrics.json"
    if os.path.exists(path):
        with open(path) as f:
            metrics = json.load(f)
        return jsonify(metrics)
    return jsonify({"error": "No model metrics found. Run training first."}), 404


@model_metrics_bp.route("/api/models")
@token_required
def list_models():
    from backend.ml.models import ModelTrainer, MLService
    all_models = list(ModelTrainer.MODELS.keys())
    try:
        ml = MLService(models_dir="models")
        all_models += list(ml.dl_models.keys())
    except Exception:
        pass
    return jsonify(all_models)
