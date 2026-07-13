"""Data loading and preprocessing for heart disease classification."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_heart_disease_data(
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Load and preprocess the UCI Heart Disease dataset.

    Uses the Cleveland dataset from the UCI ML Repository via OpenML.
    Returns scaled train/test splits with feature and target names.
    """
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name="heart-statlog", version=1, as_frame=True, parser="auto")
    X = data.data.values.astype(np.float32)
    y = data.target.values.astype(np.int32)

    # Convert target to binary: 1 = disease present, 0 = absent
    y = (y > 0).astype(np.int32)

    feature_names = list(data.feature_names)
    target_names = ["no_disease", "disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, X_test, y_train, y_test, feature_names, target_names
