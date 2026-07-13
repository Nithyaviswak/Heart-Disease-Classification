"""PyTorch MLP model with early stopping, TensorBoard, and ONNX export."""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


class _MLPNet(nn.Module):
    """Raw PyTorch MLP architecture."""

    def __init__(self, input_dim: int, hidden_layers: tuple[int, ...], dropout_rate: float) -> None:
        super().__init__()
        layers_list: list[nn.Module] = []
        prev_dim = input_dim
        for units in hidden_layers:
            layers_list.extend([
                nn.Linear(prev_dim, units),
                nn.BatchNorm1d(units),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = units
        layers_list.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PyTorchMLP:
    """PyTorch MLP classifier with full training pipeline."""

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
        self.name = "PyTorch MLP"

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _MLPNet(input_dim, hidden_layers, dropout_rate).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def _to_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 200,
        batch_size: int = 32,
        log_dir: str = "logs/pytorch",
        patience: int = 15,
    ) -> dict:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

        train_loader = self._to_loader(X_train, y_train, batch_size, shuffle=True)
        val_loader = (
            self._to_loader(X_val, y_val, batch_size, shuffle=False)
            if X_val is not None
            else None
        )

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            # --- train ---
            self.model.train()
            train_loss_sum = 0.0
            n_samples = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X_batch), y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss_sum += loss.item() * X_batch.size(0)
                n_samples += X_batch.size(0)

            avg_train_loss = train_loss_sum / n_samples
            writer.add_scalar("loss/train", avg_train_loss, epoch)

            # --- validate ---
            if val_loader is not None:
                self.model.eval()
                val_loss_sum = 0.0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        logits = self.model(X_batch)
                        val_loss_sum += self.criterion(logits, y_batch).item() * X_batch.size(0)
                        preds = (torch.sigmoid(logits) > 0.5).float()
                        val_correct += (preds == y_batch).sum().item()
                        val_total += X_batch.size(0)

                avg_val_loss = val_loss_sum / val_total
                val_acc = val_correct / val_total
                writer.add_scalar("loss/val", avg_val_loss, epoch)
                writer.add_scalar("accuracy/val", val_acc, epoch)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    best_epoch = epoch
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        break

        writer.close()

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        return {
            "best_epoch": best_epoch,
            "epochs_trained": epoch,
            "final_loss": avg_train_loss,
        }

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        self.model.eval()
        X_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            preds = (torch.sigmoid(logits) > 0.5).float()
            loss = self.criterion(logits, y_t).item()
            accuracy = (preds == y_t).float().mean().item()
        return {
            "accuracy": accuracy,
            "loss": loss,
            "predictions": preds.cpu().numpy().flatten().astype(int),
        }

    def export_onnx(self, path: str = "models/pytorch_mlp.onnx") -> str:
        self.model.eval()
        dummy = torch.randn(1, self.input_dim).to(self.device)
        torch.onnx.export(
            self.model,
            dummy,
            path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=14,
        )
        return path
