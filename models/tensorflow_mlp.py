"""TensorFlow/Keras MLP model with early stopping, TensorBoard, and ONNX export."""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers


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
    ) -> dict:
        os.makedirs(log_dir, exist_ok=True)

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
        import tf2onnx
        import onnx

        spec = (tf.TensorSpec((None, self.input_dim), tf.float32, name="input"),)
        output_path = path
        model_proto, _ = tf2onnx.convert.from_keras(
            self.model, input_signature=spec, output_path=output_path
        )
        return output_path
