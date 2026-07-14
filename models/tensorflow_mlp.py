"""TensorFlow/Keras MLP model with early stopping, TensorBoard, and ONNX export."""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers


class TensorFlowPruningCallback(keras.callbacks.Callback):
    """Callback to stop training early if there is no chance of getting the best accuracy."""
    def __init__(self, parent, best_score: float, s_prev: float, k: int, n_folds: int, warm_up: int = 15):
        super().__init__()
        self.parent = parent
        self.best_score = best_score
        self.s_prev = s_prev
        self.k = k
        self.n_folds = n_folds
        self.warm_up = warm_up
        self.best_val_acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get("val_accuracy") or logs.get("val_acc", 0.0)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
        
        if epoch + 1 >= self.warm_up:
            max_possible = (self.s_prev + self.best_val_acc + (self.n_folds - self.k - 1) * 1.0) / self.n_folds
            if max_possible < self.best_score:
                self.model.stop_training = True
                self.parent.pruned = True


class TensorFlowMLP:
    """Keras MLP classifier with full training pipeline."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: tuple[int, ...] = (128, 64),
        dropout_rate: float = 0.3,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.name = "TensorFlow MLP"
        self.pruned = False

        tf.random.set_seed(random_state)
        np.random.seed(random_state)

        self.model = self._build_model()

    def _build_model(self) -> keras.Model:
        model = keras.Sequential()
        model.add(layers.Input(shape=(self.input_dim,)))
        for units in self.hidden_layers:
            model.add(layers.Dense(units, activation="relu"))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(1, activation="sigmoid"))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 200,
        batch_size: int = 32,
        log_dir: str = "logs/tensorflow",
        patience: int = 15,
        best_score: float = 0.0,
        s_prev: float = 0.0,
        k: int = 0,
        n_folds: int = 1,
    ) -> dict:
        os.makedirs(log_dir, exist_ok=True)
        self.pruned = False

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True,
            ),
        ]

        if best_score > 0 and X_val is not None:
            callbacks.append(TensorFlowPruningCallback(self, best_score, s_prev, k, n_folds))

        validation_data = (X_val, y_val) if X_val is not None else None

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        best_epoch = np.argmin(history.history.get("val_loss", history.history["loss"])) + 1
        return {
            "best_epoch": best_epoch,
            "epochs_trained": len(history.history["loss"]),
            "final_loss": float(history.history["loss"][-1]),
        }

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        y_pred = (self.model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
        return {"accuracy": float(accuracy), "loss": float(loss), "predictions": y_pred}

    def export_onnx(self, path: str = "models/tf_mlp.onnx") -> str:
        import subprocess
        import shutil

        temp_dir = "temp_saved_model"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        # Export Keras model to SavedModel format (works in Keras 2 and Keras 3)
        try:
            if hasattr(self.model, "export"):
                self.model.export(temp_dir)
            else:
                self.model.save(temp_dir, save_format="tf")
        except Exception:
            self.model.save(temp_dir, save_format="tf")

        # Run tf2onnx command line tool
        cmd = [
            "python", "-m", "tf2onnx.convert",
            "--saved-model", temp_dir,
            "--output", path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Clean up temporary SavedModel directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        if result.returncode != 0:
            raise RuntimeError(f"tf2onnx conversion failed: {result.stderr or result.stdout}")

        return path
