"""Phase 7: Analytics engine — risk distribution, age buckets, hospital stats."""

import os
import json
import numpy as np
import pandas as pd

from backend.database import get_connection


class AnalyticsEngine:
    """Generates analytics data for dashboards and reports."""

    @staticmethod
    def risk_distribution(predictions_df: pd.DataFrame = None) -> dict:
        if predictions_df is None:
            predictions_df = AnalyticsEngine._get_predictions_df()
        if predictions_df.empty:
            return {}
        total = len(predictions_df)
        high_risk = int(predictions_df[predictions_df["prediction"] == 1].shape[0])
        low_risk = total - high_risk
        return {
            "total_patients": total,
            "high_risk": high_risk,
            "high_risk_pct": round(high_risk / total * 100, 1),
            "low_risk": low_risk,
            "low_risk_pct": round(low_risk / total * 100, 1),
        }

    @staticmethod
    def age_bucket_distribution(predictions_df: pd.DataFrame = None) -> list:
        if predictions_df is None:
            predictions_df = AnalyticsEngine._get_predictions_df()
        if predictions_df.empty or "age" not in predictions_df.columns:
            return []
        bins = [0, 30, 40, 50, 60, 70, 100]
        labels = ["<30", "30-39", "40-49", "50-59", "60-69", "70+"]
        predictions_df["age_group"] = pd.cut(predictions_df["age"], bins=bins, labels=labels)
        dist = predictions_df.groupby("age_group", observed=True).agg(
            total=("prediction", "count"),
            disease=("prediction", lambda x: (x == 1).sum()),
        ).reset_index()
        dist["disease_pct"] = (dist["disease"] / dist["total"] * 100).round(1)
        return dist.to_dict(orient="records")

    @staticmethod
    def gender_distribution(predictions_df: pd.DataFrame = None) -> dict:
        if predictions_df is None:
            predictions_df = AnalyticsEngine._get_predictions_df()
        if predictions_df.empty or "sex" not in predictions_df.columns:
            return {}
        male = int(predictions_df[predictions_df["sex"] == 1].shape[0])
        female = int(predictions_df[predictions_df["sex"] == 0].shape[0])
        return {
            "male": male,
            "male_disease": int(predictions_df[(predictions_df["sex"] == 1) & (predictions_df["prediction"] == 1)].shape[0]),
            "female": female,
            "female_disease": int(predictions_df[(predictions_df["sex"] == 0) & (predictions_df["prediction"] == 1)].shape[0]),
        }

    @staticmethod
    def model_performance_summary() -> dict:
        path = "models/model_metrics.json"
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        from backend.ml.models import ModelTrainer
        return ModelTrainer.MODELS.keys()

    @staticmethod
    def hospital_summary(predictions_df: pd.DataFrame = None) -> dict:
        rd = AnalyticsEngine.risk_distribution(predictions_df)
        gd = AnalyticsEngine.gender_distribution(predictions_df)
        age = AnalyticsEngine.age_bucket_distribution(predictions_df)
        return {
            "risk": rd,
            "gender": gd,
            "age_buckets": age,
            "avg_age": round(predictions_df["age"].mean(), 1) if predictions_df is not None and not predictions_df.empty and "age" in predictions_df.columns else 0,
        }

    @staticmethod
    def _get_predictions_df() -> pd.DataFrame:
        try:
            conn = get_connection()
            rows = conn.execute("SELECT * FROM predictions ORDER BY created_at DESC LIMIT 10000").fetchall()
            conn.close()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame([dict(r) for r in rows])
            features_df = df["features"].apply(lambda x: pd.Series(json.loads(x)) if x else {})
            return pd.concat([df.drop(columns=["features"]), features_df], axis=1)
        except Exception:
            return pd.DataFrame()
