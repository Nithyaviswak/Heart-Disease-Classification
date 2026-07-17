"""Phase 4 & 9: Prediction history API."""

from flask import Blueprint, request, jsonify, g

from backend.auth import token_required
from backend.database import get_connection

history_bp = Blueprint("history", __name__)


@history_bp.route("/api/history")
@token_required
def get_history():
    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 50, type=int), 100)  # Cap at 100
    model_filter = request.args.get("model", "")
    user_id = g.current_user.get("user_id")

    with get_connection() as conn:
        query = "SELECT * FROM predictions WHERE user_id = ?" if user_id else "SELECT * FROM predictions WHERE 1=1"
        params = [user_id] if user_id else []

        if model_filter:
            query += " AND model_name = ?"
            params.append(model_filter)

        total = conn.execute(f"SELECT COUNT(*) as c FROM ({query}) src", params).fetchone()["c"]
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([per_page, (page - 1) * per_page])
        rows = conn.execute(query, params).fetchall()

    return jsonify({
        "data": [dict(r) for r in rows],
        "page": page,
        "per_page": per_page,
        "total": total,
    })


@history_bp.route("/api/history/stats")
@token_required
def history_stats():
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) as c FROM predictions").fetchone()["c"]
        by_model = conn.execute(
            "SELECT model_name, COUNT(*) as count, AVG(confidence) as avg_conf FROM predictions GROUP BY model_name"
        ).fetchall()
        daily = conn.execute(
            "SELECT DATE(created_at) as date, COUNT(*) as count FROM predictions GROUP BY DATE(created_at) ORDER BY date DESC LIMIT 30"
        ).fetchall()
    return jsonify({
        "total_predictions": total,
        "by_model": [dict(r) for r in by_model],
        "daily": [dict(r) for r in daily],
    })
