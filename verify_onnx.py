"""Verify ONNX exports by running inference with onnxruntime."""

import numpy as np
import onnxruntime as ort

from data import load_heart_disease_data


def verify_onnx(model_path: str, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Load an ONNX model, run inference, and return accuracy.

    Handles three output shapes:
      - 1-D int/float array  : direct class labels (skl2onnx LogReg)
      - 2-D (N, 1) float     : sigmoid probabilities (TF / PyTorch)
      - 2-D (N, C) float     : softmax probabilities (argmax)
    """
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    preds = session.run(None, {input_name: X_test})[0]

    preds = np.array(preds)

    if preds.ndim == 2 and preds.shape[1] == 1:
        # Sigmoid output → threshold at 0.5
        preds = (preds.flatten() > 0.5).astype(int)
    elif preds.ndim == 2 and preds.shape[1] > 1:
        # Softmax output → argmax
        preds = preds.argmax(axis=1).astype(int)
    else:
        # Direct label output (e.g. skl2onnx LogReg) → cast to int
        preds = preds.flatten().astype(int)

    accuracy = float(np.mean(preds == y_test.flatten()))
    return accuracy


def main() -> None:
    _, X_test, _, y_test, _, _, _ = load_heart_disease_data()

    models = {
        "Logistic Regression": "models/logreg.onnx",
        "TensorFlow MLP": "models/tf_mlp.onnx",
        "PyTorch MLP": "models/pytorch_mlp.onnx",
    }

    print("ONNX Inference Verification")
    print("=" * 45)
    for name, path in models.items():
        try:
            acc = verify_onnx(path, X_test, y_test)
            print(f"  {name:25s}  ONNX accuracy: {acc:.4f}")
        except FileNotFoundError:
            print(f"  {name:25s}  ONNX file not found: {path}")
        except Exception as e:
            print(f"  {name:25s}  Error: {e}")
    print("=" * 45)


if __name__ == "__main__":
    main()
