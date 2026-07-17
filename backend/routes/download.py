"""Phase 8 & 9: Download endpoints for CSV, Excel, PDF."""

from flask import Blueprint, send_file, jsonify

from backend.auth import token_required
from backend.reports import ReportGenerator

download_bp = Blueprint("download", __name__)


@download_bp.route("/api/download/csv/<table>")
@token_required
def download_csv(table: str = "predictions"):
    try:
        path = ReportGenerator.export_csv(table)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not path:
        return jsonify({"error": "No data found"}), 404
    return send_file(path, as_attachment=True, download_name=f"{table}.csv")


@download_bp.route("/api/download/excel")
@token_required
def download_excel():
    path = ReportGenerator.export_excel()
    return send_file(path, as_attachment=True, download_name="heart_disease_report.xlsx")


@download_bp.route("/api/download/pdf/<report_type>")
@token_required
def download_pdf(report_type: str = "summary"):
    try:
        path = ReportGenerator.export_pdf(report_type)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if path.endswith(".html"):
        return send_file(path, as_attachment=True, download_name="report.html")
    return send_file(path, as_attachment=True, download_name="report.pdf")
