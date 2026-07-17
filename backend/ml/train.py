"""Training pipeline — train all models (sklearn + DL) and save metrics."""

import os
import sys
import json
import logging
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.ml.models import ModelTrainer
from data import load_heart_disease_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"


def train_sklearn(X_train, X_test, y_train, y_test):
    logger.info("Training sklearn models...")
    trainer = ModelTrainer(models_dir=MODELS_DIR)
    results = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
    return results


def train_tensorflow(X_train, X_test, y_train, y_test):
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        from models.tensorflow_mlp import TensorFlowMLP
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        logger.info("Training TensorFlow MLP...")
        tf_mlp = TensorFlowMLP(input_dim=X_train.shape[1])
        tf_mlp.train(X_train, y_train, X_val=X_test, y_val=y_test, epochs=100, batch_size=32, patience=15)
        eval_result = tf_mlp.evaluate(X_test, y_test)
        y_prob = tf_mlp.model.predict(X_test, verbose=0).flatten()
        y_pred = eval_result["predictions"]

        acc = eval_result["accuracy"]
        metrics = {
            "accuracy": round(acc, 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        }

        model_path = os.path.join(MODELS_DIR, "tensorflow_mlp.keras")
        tf_mlp.model.save(model_path)
        logger.info("TensorFlow MLP saved to %s (acc=%.4f)", model_path, acc)

        return {"TensorFlow MLP": {**metrics, "path": model_path}}
    except Exception as e:
        logger.warning("TensorFlow training failed: %s", e)
        return {}


def train_pytorch(X_train, X_test, y_train, y_test):
    try:
        import torch
        from models.pytorch_mlp import PyTorchMLP
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        logger.info("Training PyTorch MLP...")
        pt_mlp = PyTorchMLP(input_dim=X_train.shape[1])
        pt_mlp.train(X_train, y_train, X_val=X_test, y_val=y_test, epochs=100, batch_size=32, patience=15)
        eval_result = pt_mlp.evaluate(X_test, y_test)
        y_pred = eval_result["predictions"]

        pt_mlp.model.eval()
        X_t = torch.tensor(X_test, dtype=torch.float32).to(pt_mlp.device)
        with torch.no_grad():
            y_prob = torch.sigmoid(pt_mlp.model(X_t)).cpu().numpy().flatten()

        acc = eval_result["accuracy"]
        metrics = {
            "accuracy": round(acc, 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        }

        model_path = os.path.join(MODELS_DIR, "pytorch_mlp.pt")
        torch.save({
            "model_state_dict": pt_mlp.model.state_dict(),
            "input_dim": pt_mlp.input_dim,
            "hidden_layers": pt_mlp.hidden_layers,
            "dropout_rate": pt_mlp.dropout_rate,
        }, model_path)
        logger.info("PyTorch MLP saved to %s (acc=%.4f)", model_path, acc)

        return {"PyTorch MLP": {**metrics, "path": model_path}}
    except Exception as e:
        logger.warning("PyTorch training failed: %s", e)
        return {}


def main():
    logger.info("Loading data...")
    X_train, X_test, y_train, y_test, feature_names, target_names, scaler = load_heart_disease_data()

    os.makedirs(MODELS_DIR, exist_ok=True)

    import pickle
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    all_results = {}

    all_results.update(train_sklearn(X_train, X_test, y_train, y_test))
    all_results.update(train_tensorflow(X_train, X_test, y_train, y_test))
    all_results.update(train_pytorch(X_train, X_test, y_train, y_test))

    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "path"} for k, v in all_results.items()}, f, indent=2)

    print("\n" + "=" * 65)
    print(f"{'Model':25s} {'Accuracy':10s} {'Precision':10s} {'Recall':10s} {'F1':10s} {'ROC AUC':10s}")
    print("=" * 65)
    for name, m in all_results.items():
        print(f"{name:25s} {m['accuracy']:<10.4f} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1_score']:<10.4f} {m['roc_auc']:<10.4f}")
    print("=" * 65)

    best = max(all_results.items(), key=lambda x: x[1]["accuracy"])
    print(f"\nBest model: {best[0]} (accuracy={best[1]['accuracy']:.4f})")


if __name__ == "__main__":
    main()
