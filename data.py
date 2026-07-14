"""Data loading and preprocessing for heart disease classification."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_heart_disease_data(
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str], StandardScaler]:
    """Load and preprocess the local UCI Heart Disease dataset.

    Reads the dataset from data/heart_disease.csv.
    Returns scaled train/test splits, feature/target names, and the fitted scaler.
    """
    import os
    import pandas as pd
    import numpy as np

    csv_path = os.path.join("data", "heart_disease.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at {csv_path}. Please place the dataset there first.")

    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int32)

    # Convert target to binary: 1 = disease present, 0 = absent
    y = (y > 0).astype(np.int32)

    feature_names = list(df.columns[:-1])
    target_names = ["no_disease", "disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, X_test, y_train, y_test, feature_names, target_names, scaler
