"""Phase 5: Generate Power BI datasets and analytics CSV files."""

import os
import json
import csv
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "analytics")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_datasets():
    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), "..", "heart_disease.db"))
    conn.row_factory = sqlite3.Row

    # 1. Predictions Dataset
    try:
        df_pred = pd.read_sql("SELECT * FROM predictions ORDER BY created_at DESC", conn)
        df_pred.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
        print(f"  predictions.csv: {len(df_pred)} rows")
    except Exception as e:
        print(f"  predictions.csv: skipped ({e})")
        df_pred = pd.DataFrame()

    # 2. Model Metrics Dataset
    try:
        df_metrics = pd.read_sql("SELECT * FROM model_metrics ORDER BY accuracy DESC", conn)
        df_metrics.to_csv(os.path.join(OUTPUT_DIR, "model_metrics.csv"), index=False)
        print(f"  model_metrics.csv: {len(df_metrics)} rows")
    except Exception as e:
        print(f"  model_metrics.csv: skipped ({e})")

    # 3. Patients Dataset (from features JSON in predictions)
    if not df_pred.empty and "features" in df_pred.columns:
        rows = []
        for _, r in df_pred.iterrows():
            try:
                feats = json.loads(r["features"])
                rows.append({
                    "prediction_id": r["id"],
                    "age": feats.get("age"),
                    "sex": feats.get("sex"),
                    "cp": feats.get("cp"),
                    "trestbps": feats.get("trestbps"),
                    "chol": feats.get("chol"),
                    "fbs": feats.get("fbs"),
                    "restecg": feats.get("restecg"),
                    "thalach": feats.get("thalach"),
                    "exang": feats.get("exang"),
                    "oldpeak": feats.get("oldpeak"),
                    "slope": feats.get("slope"),
                    "ca": feats.get("ca"),
                    "thal": feats.get("thal"),
                    "prediction": r["prediction"],
                    "probability": r["probability"],
                    "risk_score": r["risk_score"],
                    "model_name": r["model_name"],
                    "created_at": r["created_at"],
                })
            except Exception:
                pass
        df_patients = pd.DataFrame(rows)
        df_patients.to_csv(os.path.join(OUTPUT_DIR, "patients.csv"), index=False)
        print(f"  patients.csv: {len(df_patients)} rows")

    # 4. Dashboard Summary Dataset
    rd = _risk_distribution(df_pred)
    with open(os.path.join(OUTPUT_DIR, "dashboard_summary.json"), "w") as f:
        json.dump(rd, f, indent=2)
    print(f"  dashboard_summary.json saved")

    # 5. Age Buckets Dataset
    age_buckets = _age_buckets(df_patients if not df_pred.empty else pd.DataFrame())
    if age_buckets:
        pd.DataFrame(age_buckets).to_csv(os.path.join(OUTPUT_DIR, "age_buckets.csv"), index=False)
        print(f"  age_buckets.csv: {len(age_buckets)} rows")

    # 6. Correlation Matrix CSV
    if not df_patients.empty:
        numeric_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "prediction"]
        corr = df_patients[numeric_cols].corr()
        corr.to_csv(os.path.join(OUTPUT_DIR, "correlation_matrix.csv"))
        print(f"  correlation_matrix.csv saved")

    conn.close()
    print(f"\nDatasets saved to: {OUTPUT_DIR}")


def _risk_distribution(df):
    if df.empty:
        return {"total": 0, "high_risk": 0, "low_risk": 0}
    total = len(df)
    high = int(df[df["prediction"] == 1].shape[0])
    return {
        "total_patients": total,
        "high_risk": high,
        "high_risk_pct": round(high / total * 100, 1) if total else 0,
        "low_risk": total - high,
        "low_risk_pct": round((total - high) / total * 100, 1) if total else 0,
    }


def _age_buckets(df):
    if df.empty or "age" not in df.columns:
        return []
    bins = [0, 30, 40, 50, 60, 70, 100]
    labels = ["<30", "30-39", "40-49", "50-59", "60-69", "70+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)
    grouped = df.groupby("age_group", observed=True).agg(
        total=("prediction", "count"),
        disease=("prediction", lambda x: (x == 1).sum()),
    ).reset_index()
    grouped["disease_pct"] = (grouped["disease"] / grouped["total"] * 100).round(1)
    return grouped.to_dict(orient="records")


if __name__ == "__main__":
    generate_datasets()
