"""Phase 5, 7 & 9: Dashboard and Analytics API."""

from flask import Blueprint, jsonify

from backend.auth import token_required
from backend.analytics import AnalyticsEngine

dashboard_bp = Blueprint("dashboard", __name__)
analytics_bp = Blueprint("analytics", __name__)


@dashboard_bp.route("/api/dashboard/summary")
@token_required
def dashboard_summary():
    eng = AnalyticsEngine()
    rd = eng.risk_distribution()
    gd = eng.gender_distribution()
    age = eng.age_bucket_distribution()
    ms = eng.model_performance_summary()
    return jsonify({
        "risk_distribution": rd,
        "gender_distribution": gd,
        "age_buckets": age,
        "model_performance": ms,
    })


@dashboard_bp.route("/api/dashboard/kpi")
@token_required
def dashboard_kpi():
    from backend.database import get_connection
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) as c FROM predictions").fetchone()["c"]
        high = conn.execute("SELECT COUNT(*) as c FROM predictions WHERE prediction=1").fetchone()["c"]
        avg_age = conn.execute("SELECT AVG(CAST(json_extract(features, '$.age') AS REAL)) as a FROM predictions").fetchone()["a"]
        avg_conf = conn.execute("SELECT AVG(confidence) as c FROM predictions").fetchone()["c"]
    return jsonify({
        "total_patients": total,
        "high_risk_patients": high,
        "avg_age": round(avg_age or 0, 1),
        "avg_confidence": round(avg_conf or 0, 3),
        "heart_disease_pct": round(high / total * 100, 1) if total else 0,
    })


@analytics_bp.route("/api/analytics/summary")
@token_required
def analytics_summary():
    eng = AnalyticsEngine()
    df = eng._get_predictions_df()
    return jsonify({
        "hospital_summary": eng.hospital_summary(df),
        "risk_distribution": eng.risk_distribution(df),
        "age_buckets": eng.age_bucket_distribution(df),
        "gender": eng.gender_distribution(df),
    })
