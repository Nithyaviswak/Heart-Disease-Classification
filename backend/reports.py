"""Phase 8: Report generation — PDF, CSV, Excel exports."""

import os
import io
import csv
import json
from datetime import datetime

import numpy as np
import pandas as pd

from backend.database import get_connection

# Security: whitelist of allowed table names to prevent SQL injection
ALLOWED_TABLES = {"predictions", "model_metrics", "patients", "users"}
ALLOWED_REPORT_TYPES = {"summary", "detailed", "clinical"}


class ReportGenerator:
    """Generates exportable reports in multiple formats."""

    EXPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

    @classmethod
    def _ensure_dir(cls):
        os.makedirs(cls.EXPORT_DIR, exist_ok=True)

    @classmethod
    def _validate_table(cls, table: str) -> str:
        """Validate table name against whitelist to prevent SQL injection."""
        if table not in ALLOWED_TABLES:
            raise ValueError(
                f"Invalid table name: '{table}'. "
                f"Allowed tables: {', '.join(sorted(ALLOWED_TABLES))}"
            )
        return table

    @classmethod
    def _validate_report_type(cls, report_type: str) -> str:
        """Validate report type against whitelist."""
        if report_type not in ALLOWED_REPORT_TYPES:
            raise ValueError(
                f"Invalid report type: '{report_type}'. "
                f"Allowed types: {', '.join(sorted(ALLOWED_REPORT_TYPES))}"
            )
        return report_type

    @classmethod
    def export_csv(cls, table: str = "predictions") -> str:
        cls._ensure_dir()
        table = cls._validate_table(table)
        with get_connection() as conn:
            rows = conn.execute(f"SELECT * FROM {table} ORDER BY created_at DESC").fetchall()
        if not rows:
            return None
        df = pd.DataFrame([dict(r) for r in rows])
        # Exclude sensitive columns from user exports
        if table == "users":
            df = df.drop(columns=["password_hash"], errors="ignore")
        path = os.path.join(cls.EXPORT_DIR, f"{table}_{datetime.now():%Y%m%d_%H%M%S}.csv")
        df.to_csv(path, index=False)
        return path

    @classmethod
    def export_excel(cls) -> str:
        cls._ensure_dir()
        path = os.path.join(cls.EXPORT_DIR, f"full_report_{datetime.now():%Y%m%d_%H%M%S}.xlsx")
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for table in ["predictions", "model_metrics"]:
                table = cls._validate_table(table)
                try:
                    with get_connection() as conn:
                        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
                    if rows:
                        df = pd.DataFrame([dict(r) for r in rows])
                        df.to_excel(writer, sheet_name=table, index=False)
                except Exception:
                    pass
        return path

    @classmethod
    def export_pdf(cls, report_type: str = "summary") -> str:
        cls._ensure_dir()
        report_type = cls._validate_report_type(report_type)
        path = os.path.join(cls.EXPORT_DIR, f"{report_type}_{datetime.now():%Y%m%d_%H%M%S}.pdf")
        try:
            from weasyprint import HTML
        except ImportError:
            html = cls._generate_html_report(report_type)
            path = path.replace(".pdf", ".html")
            with open(path, "w") as f:
                f.write(html)
            return path

        html = cls._generate_html_report(report_type)
        HTML(string=html).write_pdf(path)
        return path

    @classmethod
    def _generate_html_report(cls, report_type: str) -> str:
        with get_connection() as conn:
            pred_count = conn.execute("SELECT COUNT(*) as c FROM predictions").fetchone()["c"]
            high_risk = conn.execute("SELECT COUNT(*) as c FROM predictions WHERE prediction=1").fetchone()["c"]
            metrics_rows = conn.execute("SELECT * FROM model_metrics ORDER BY accuracy DESC").fetchall()

        metrics_rows = [dict(r) for r in metrics_rows] if metrics_rows else []

        rows_html = ""
        for m in metrics_rows:
            rows_html += f"<tr><td>{m['model_name']}</td><td>{m.get('accuracy', 'N/A')}</td><td>{m.get('roc_auc', 'N/A')}</td></tr>"

        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Heart Disease Report</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; padding: 40px; color: #1e293b; }}
h1 {{ color: #1e3a5f; border-bottom: 2px solid #2563eb; padding-bottom: 10px; }}
table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
th {{ background: #1e3a5f; color: white; padding: 10px; text-align: left; }}
td {{ padding: 8px 10px; border-bottom: 1px solid #e2e8f0; }}
.footer {{ margin-top: 40px; color: #94a3b8; font-size: 12px; }}
</style></head><body>
<h1>Clinical Analytics Report</h1>
<p>Generated: {datetime.now():%B %d, %Y at %H:%M}</p>
<div style="display:flex;gap:20px;margin:20px 0;">
<div style="background:#f8fafc;padding:20px;border-radius:8px;flex:1;text-align:center;">
<h3>Total Predictions</h3><p style="font-size:32px;font-weight:700;">{pred_count}</p></div>
<div style="background:#fef2f2;padding:20px;border-radius:8px;flex:1;text-align:center;">
<h3>High Risk</h3><p style="font-size:32px;font-weight:700;color:#dc2626;">{high_risk}</p></div>
</div>
<h2>Model Performance</h2>
<table><thead><tr><th>Model</th><th>Accuracy</th><th>ROC AUC</th></tr></thead>
<tbody>{rows_html}</tbody></table>
<div class="footer">Heart Disease Analytics Platform &mdash; Confidential</div>
</body></html>"""
