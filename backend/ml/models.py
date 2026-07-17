"""Phase 2 & 3: Multiple model training, comparison, and SHAP explainability."""

import os
import json
import pickle
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


class ModelTrainer:
    """Trains, evaluates, and persists multiple models."""

    MODELS = {
        "Logistic Regression": lambda: LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "SVM": lambda: SVC(kernel="rbf", probability=True, random_state=42, max_iter=2000),
    }

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.best_model_name = None
        self.best_score = 0.0
        self.metrics = {}
        os.makedirs(models_dir, exist_ok=True)

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        results = {}
        for name, builder in self.MODELS.items():
            logger.info("Training %s...", name)
            model = builder()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0

            self.metrics[name] = {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(roc, 4),
            }

            path = os.path.join(self.models_dir, f"{name.lower().replace(' ', '_')}.joblib")
            joblib.dump(model, path)

            results[name] = self.metrics[name]
            results[name]["path"] = path

            if acc > self.best_score:
                self.best_score = acc
                self.best_model_name = name

        self._save_metrics()
        return results

    def _save_metrics(self):
        path = os.path.join(self.models_dir, "model_metrics.json")
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def get_best_model(self):
        if not self.best_model_name:
            return None
        path = os.path.join(self.models_dir, f"{self.best_model_name.lower().replace(' ', '_')}.joblib")
        return joblib.load(path)

    def load_model(self, name: str):
        path = os.path.join(self.models_dir, f"{name.lower().replace(' ', '_')}.joblib")
        if os.path.exists(path):
            return joblib.load(path)
        return None


class MLService:
    """Inference wrapper supporting sklearn, TensorFlow, and PyTorch models."""

    DL_MODELS = {"TensorFlow MLP", "PyTorch MLP"}

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.models = {}
        self.dl_models = {}
        self.scaler = None
        self.metrics = self._load_metrics()
        self._load_all()
        self._load_scaler()

    def _load_metrics(self):
        path = os.path.join(self.models_dir, "model_metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    def _load_scaler(self):
        path = os.path.join(self.models_dir, "scaler.pkl")
        if os.path.exists(path):
            self.scaler = joblib.load(path)

    def _load_all(self):
        for name in ModelTrainer.MODELS:
            path = os.path.join(self.models_dir, f"{name.lower().replace(' ', '_')}.joblib")
            if os.path.exists(path):
                self.models[name] = joblib.load(path)

        self._load_tensorflow()
        self._load_pytorch()

    def _load_tensorflow(self):
        path = os.path.join(self.models_dir, "tensorflow_mlp.keras")
        if not os.path.exists(path):
            return
        try:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            import tensorflow as tf
            model = tf.keras.models.load_model(path)
            self.dl_models["TensorFlow MLP"] = {"type": "tensorflow", "model": model}
        except Exception as e:
            logger.warning("Could not load TensorFlow model: %s", e)

    def _load_pytorch(self):
        path = os.path.join(self.models_dir, "pytorch_mlp.pt")
        if not os.path.exists(path):
            return
        try:
            import torch
            from models.pytorch_mlp import _MLPNet
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = _MLPNet(
                input_dim=checkpoint["input_dim"],
                hidden_layers=checkpoint["hidden_layers"],
                dropout_rate=checkpoint["dropout_rate"],
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self.dl_models["PyTorch MLP"] = {"type": "pytorch", "model": model}
        except Exception as e:
            logger.warning("Could not load PyTorch model: %s", e)

    def predict(self, model_name: str, features: np.ndarray) -> dict:
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if model_name in self.dl_models:
            return self._predict_dl(model_name, features)

        model = self.models.get(model_name)
        if model is None:
            model = self.models.get("Logistic Regression")
            model_name = "Logistic Regression"
            if model is None:
                return {"error": "No models available"}

        pred = int(model.predict(features)[0])
        prob = float(model.predict_proba(features)[0, 1])
        confidence = prob if pred == 1 else 1 - prob

        shap_explanation = ShapExplainer.explain(model, features, model_name)

        return {
            "prediction": pred,
            "label": "Heart Disease Detected" if pred == 1 else "No Heart Disease",
            "probability": round(prob, 4),
            "confidence": round(confidence, 4),
            "risk_score": round(prob * 100, 1),
            "model": model_name,
            "shap": shap_explanation,
        }

    def _predict_dl(self, model_name: str, features: np.ndarray) -> dict:
        info = self.dl_models[model_name]
        X = features.astype(np.float32)

        if info["type"] == "tensorflow":
            import tensorflow as tf
            prob = float(info["model"].predict(X, verbose=0).flatten()[0])
        elif info["type"] == "pytorch":
            import torch
            X_t = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                prob = float(torch.sigmoid(info["model"](X_t)).item())

        pred = int(prob > 0.5)
        confidence = prob if pred == 1 else 1 - prob

        feat_names = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        ]
        contributions = [{"feature": name, "value": 0.0} for name in feat_names]

        return {
            "prediction": pred,
            "label": "Heart Disease Detected" if pred == 1 else "No Heart Disease",
            "probability": round(prob, 4),
            "confidence": round(confidence, 4),
            "risk_score": round(prob * 100, 1),
            "model": model_name,
            "shap": {
                "top_features": contributions[:5],
                "positive_contributors": [],
                "negative_contributors": [],
                "base_value": 0.5,
                "note": "SHAP not supported for deep learning models",
            },
        }

    def predict_all(self, features: np.ndarray) -> dict:
        results = {}
        for name in list(self.models.keys()) + list(self.dl_models.keys()):
            results[name] = self.predict(name, features)
        return results

    def get_metrics(self):
        return self.metrics


class ShapExplainer:
    """Phase 3: SHAP-based explainable AI for model predictions."""

    @staticmethod
    def explain(model, features: np.ndarray, model_name: str) -> dict:
        try:
            import shap
        except ImportError:
            return {"error": "SHAP not installed. Run: pip install shap"}

        if features.ndim == 1:
            features = features.reshape(1, -1)

        try:
            explainer = shap.TreeExplainer(model) if "Forest" in model_name else shap.KernelExplainer(model.predict_proba, features)
            shap_values = explainer.shap_values(features)

            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            shap_values = np.array(shap_values)
            while shap_values.ndim > 2:
                shap_values = shap_values[0]

            vals = shap_values.flatten() if shap_values.ndim > 1 else shap_values
            feat_names = [
                "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal",
            ]

            base_val = 0.5
            if hasattr(explainer, "expected_value"):
                ev = explainer.expected_value
                base_val = float(np.mean(ev)) if isinstance(ev, np.ndarray) else float(ev)

            contributions = []
            positive = []
            negative = []
            for i, (name, val) in enumerate(zip(feat_names, vals)):
                val = float(np.squeeze(val))
                contributions.append({"feature": name, "value": round(val, 4)})
                if val > 0:
                    positive.append({"feature": name, "impact": round(val, 4)})
                elif val < 0:
                    negative.append({"feature": name, "impact": round(val, 4)})

            positive.sort(key=lambda x: x["impact"], reverse=True)
            negative.sort(key=lambda x: x["impact"])

            return {
                "top_features": contributions[:5],
                "positive_contributors": positive[:5],
                "negative_contributors": negative[:5],
                "base_value": round(base_val, 4),
            }
        except Exception as e:
            return {"error": str(e)}
