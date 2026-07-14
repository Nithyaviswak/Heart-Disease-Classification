# Heart Disease Classification

Deep learning and classical ML models compared on the UCI Heart Disease dataset.

## Models

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 0.8033 |
| TensorFlow MLP | 0.7705 |
| PyTorch MLP | 0.8197 |

## Features

- **Deep Learning**: TensorFlow/Keras MLP and PyTorch MLP with BatchNorm + Dropout
- **Early Stopping**: Restores best weights on validation loss plateau
- **TensorBoard**: Live training metrics at `logs/tensorflow/` and `logs/pytorch/`
- **Hyperparameter Tuning**: Grid search over architecture, dropout, and learning rate
- **ONNX Export**: All three models exported to ONNX format for cross-platform inference
- **ONNX Verification**: `verify_onnx.py` confirms ONNX models match native accuracy

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train all models, tune hyperparameters, export ONNX
python train.py

# Verify ONNX exports match native accuracy
python verify_onnx.py

# Launch TensorBoard
tensorboard --logdir logs/
```

## Project Structure

```
heart-disease-classification/
├── data.py                 # Data loading and preprocessing
├── models/
│   ├── __init__.py
│   ├── logistic_regression.py   # Sklearn Logistic Regression
│   ├── tensorflow_mlp.py        # TensorFlow/Keras MLP
│   └── pytorch_mlp.py           # PyTorch MLP
├── train.py                # Main training + comparison pipeline
├── verify_onnx.py          # ONNX export verification
└── requirements.txt
```

## Tech Stack

- **TensorFlow** / Keras — deep learning MLP
- **PyTorch** — deep learning MLP
- **scikit-learn** — Logistic Regression baseline, preprocessing, hyperparameter tuning
- **ONNX** / onnxruntime — cross-framework model export and inference
- **TensorBoard** — training visualization
