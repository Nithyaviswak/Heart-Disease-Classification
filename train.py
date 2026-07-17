"""Train all models, run hyperparameter tuning, export ONNX, and print comparison."""

import os
import json
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from data import load_heart_disease_data
from models import LogisticRegressionModel, TensorFlowMLP, PyTorchMLP


# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------
HP_SPACE = {
    "hidden_layers": [(128, 64), (64, 32)],
    "dropout_rate": [0.2, 0.5],
    "learning_rate": [1e-3, 1e-4],
}


def grid_search(
    X: np.ndarray,
    y: np.ndarray,
    model_cls,
    input_dim: int,
    hp_space: dict,
    n_folds: int = 2,
    epochs: int = 40,
    batch_size: int = 32,
    random_state: int = 42,
) -> tuple[dict, float]:
    """Grid search with cross-validation."""
    keys = list(hp_space.keys())
    combos = list(itertools.product(*[hp_space[k] for k in keys]))

    best_score = 0.0
    best_params = {}
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    print(f"  Grid search: {len(combos)} combos x {n_folds} folds")

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        fold_scores = []
        pruned = False

        for k, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            if best_score > 0.0:
                s_prev = sum(fold_scores)
                max_possible = (s_prev + (n_folds - k) * 1.0) / n_folds
                if max_possible < best_score:
                    pruned = True
                    break

            X_tr, X_vl = X[train_idx], X[val_idx]
            y_tr, y_vl = y[train_idx], y[val_idx]

            model = model_cls(
                input_dim=input_dim,
                hidden_layers=params["hidden_layers"],
                dropout_rate=params["dropout_rate"],
                learning_rate=params["learning_rate"],
                random_state=random_state,
            )
            model.train(
                X_tr, y_tr, X_vl, y_vl,
                epochs=epochs,
                batch_size=batch_size,
                best_score=best_score,
                s_prev=sum(fold_scores),
                k=k,
                n_folds=n_folds,
            )

            if getattr(model, "pruned", False):
                pruned = True
                break

            result = model.evaluate(X_vl, y_vl)
            fold_scores.append(result["accuracy"])

        if not pruned and len(fold_scores) == n_folds:
            mean_score = float(np.mean(fold_scores))
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        if (i + 1) % 3 == 0 or (i + 1) == len(combos):
            print(f"    [{i+1}/{len(combos)}] best so far: {best_score:.4f}")

    return best_params, best_score


def build_comparison_table(results: dict) -> pd.DataFrame:
    """Build a formatted comparison DataFrame."""
    rows = []
    for name, res in results.items():
        rows.append({
            "Model": name,
            "Accuracy": f"{res['accuracy']:.4f}",
        })
    df = pd.DataFrame(rows)
    return df


def main() -> None:
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("=" * 60)
    print("  Heart Disease Classification — Model Comparison")
    print("=" * 60)

    # --- data ---
    print("\n[1/5] Loading data...")
    X_train, X_test, y_train, y_test, feature_names, target_names, scaler = (
        load_heart_disease_data()
    )
    input_dim = X_train.shape[1]
    print(f"  Features: {input_dim}  |  Train: {len(X_train)}  |  Test: {len(y_test)}")

    # Save scaler for inference API (using joblib instead of pickle for security)
    import joblib
    joblib.dump(scaler, "models/scaler.pkl")
    print("  Scaler saved -> models/scaler.pkl (joblib)")

    # Split train into train+val for DL early stopping
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    results = {}

    # --- Logistic Regression ---
    print("\n[2/5] Training Logistic Regression...")
    lr = LogisticRegressionModel()
    lr.train(X_train, y_train)
    lr_res = lr.evaluate(X_test, y_test)
    results["Logistic Regression"] = lr_res
    lr_onnx_path = lr.export_onnx(X_train, path="models/logreg.onnx")
    print(f"  Accuracy: {lr_res['accuracy']:.4f}  |  ONNX: {lr_onnx_path}")

    # --- TensorFlow MLP hyperparameter tuning ---
    print("\n[3/5] Hyperparameter tuning — TensorFlow MLP...")
    tf_best_hp, tf_best_cv = grid_search(
        X_train, y_train, TensorFlowMLP, input_dim, HP_SPACE,
        n_folds=2, epochs=40, batch_size=32,
    )
    print(f"  Best CV accuracy: {tf_best_cv:.4f}")
    print(f"  Best params: {tf_best_hp}")

    print("  Training final TensorFlow MLP with best params...")
    tf_mlp = TensorFlowMLP(
        input_dim=input_dim,
        hidden_layers=tf_best_hp["hidden_layers"],
        dropout_rate=tf_best_hp["dropout_rate"],
        learning_rate=tf_best_hp["learning_rate"],
    )
    tf_train_info = tf_mlp.train(X_tr, y_tr, X_val, y_val, epochs=200, log_dir="logs/tensorflow")
    tf_res = tf_mlp.evaluate(X_test, y_test)
    results["TensorFlow MLP"] = tf_res
    tf_onnx_path = tf_mlp.export_onnx(path="models/tf_mlp.onnx")
    print(f"  Accuracy: {tf_res['accuracy']:.4f}  |  Best epoch: {tf_train_info['best_epoch']}  |  ONNX: {tf_onnx_path}")

    # --- PyTorch MLP hyperparameter tuning ---
    print("\n[4/5] Hyperparameter tuning — PyTorch MLP...")
    pt_best_hp, pt_best_cv = grid_search(
        X_train, y_train, PyTorchMLP, input_dim, HP_SPACE,
        n_folds=2, epochs=40, batch_size=32,
    )
    print(f"  Best CV accuracy: {pt_best_cv:.4f}")
    print(f"  Best params: {pt_best_hp}")

    print("  Training final PyTorch MLP with best params...")
    pt_mlp = PyTorchMLP(
        input_dim=input_dim,
        hidden_layers=pt_best_hp["hidden_layers"],
        dropout_rate=pt_best_hp["dropout_rate"],
        learning_rate=pt_best_hp["learning_rate"],
    )
    pt_train_info = pt_mlp.train(X_tr, y_tr, X_val, y_val, epochs=200, log_dir="logs/pytorch")
    pt_res = pt_mlp.evaluate(X_test, y_test)
    results["PyTorch MLP"] = pt_res
    pt_onnx_path = pt_mlp.export_onnx(path="models/pytorch_mlp.onnx")
    print(f"  Accuracy: {pt_res['accuracy']:.4f}  |  Best epoch: {pt_train_info['best_epoch']}  |  ONNX: {pt_onnx_path}")

    # --- comparison table ---
    print("\n[5/5] Model Comparison")
    print("=" * 40)
    df = build_comparison_table(results)
    print(df.to_string(index=False))
    print("=" * 40)

    # Save results
    df.to_csv("models/comparison.csv", index=False)
    with open("models/best_hyperparameters.json", "w") as f:
        json.dump({"tensorflow": tf_best_hp, "pytorch": pt_best_hp}, f, indent=2)

    print("\nExported artifacts:")
    print("  models/logreg.onnx")
    print("  models/tf_mlp.onnx")
    print("  models/pytorch_mlp.onnx")
    print("  models/scaler.pkl")
    print("  models/comparison.csv")
    print("  models/best_hyperparameters.json")
    print("\nTensorBoard logs:")
    print("  tensorboard --logdir logs/")
    print("\nFrontend:")
    print("  python app.py  ->  http://localhost:5000")
    print("Done.")


if __name__ == "__main__":
    main()
