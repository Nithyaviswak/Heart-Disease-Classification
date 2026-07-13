"""Logistic Regression baseline model."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class LogisticRegressionModel:
    """Sklearn Logistic Regression with standardized interface."""

    def __init__(self, random_state: int = 42) -> None:
        self.model = LogisticRegression(
            max_iter=1000, random_state=random_state
        )
        self.name = "Logistic Regression"

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        self.model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, self.model.predict(X_train))
        return {"train_accuracy": train_acc}

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return {"accuracy": accuracy, "predictions": y_pred}

    def export_onnx(self, X_train: np.ndarray, path: str = "models/logreg.onnx") -> str:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [("float_input", FloatTensorType([None, X_train.shape[1]]))]
        onnx_model = convert_sklearn(self.model, initial_types=initial_type)
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        return path
