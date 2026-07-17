"""Enterprise Healthcare Analytics Platform — Main Application.

Combines prediction API, authentication, analytics, and report generation.
Preserves backward compatibility with existing routes.

Security hardened:
- CORS restricted to configured origins
- Security headers added (CSP, X-Content-Type-Options, X-Frame-Options, etc.)
- Rate limiting on auth endpoints
"""

import os
import logging
from flask import Flask, render_template, jsonify
from flask_cors import CORS

from backend.config import Config
from backend.database import init_db

# Route blueprints
from backend.routes.auth import auth_bp
from backend.routes.predict import predict_bp
from backend.routes.history import history_bp
from backend.routes.dashboard import dashboard_bp, analytics_bp
from backend.routes.download import download_bp
from backend.routes.metrics import model_metrics_bp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _add_security_headers(app):
    """Add security headers to all responses."""

    @app.after_request
    def set_security_headers(response):
        # Prevent MIME-type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        # Enable XSS filter
        response.headers["X-XSS-Protection"] = "1; mode=block"
        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Permissions policy
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://accounts.google.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self' https://accounts.google.com; "
            "frame-src https://accounts.google.com;"
        )
        # HSTS (only in production)
        if os.environ.get("FLASK_ENV") == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


def create_app():
    app = Flask(
        __name__,
        static_folder="../frontend/dist" if os.path.exists("../frontend/dist") else "static",
        template_folder="templates",
    )
    app.config.from_object(Config)

    # Fix #6: Restrict CORS to configured origins
    cors_origins = Config.CORS_ORIGINS
    if cors_origins:
        origins_list = [o.strip() for o in cors_origins.split(",") if o.strip()]
        CORS(app, origins=origins_list, supports_credentials=True)
    else:
        # Production with no CORS_ORIGINS set: allow same-origin only
        CORS(app, origins=[], supports_credentials=True)

    # Fix #12: Add security headers
    _add_security_headers(app)

    # Initialize database
    init_db()

    # Register API blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(download_bp)
    app.register_blueprint(model_metrics_bp)

    # Health check (public, no auth required)
    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "version": "2.0.0"})

    # Legacy Flask template route (backward compatible)
    @app.route("/")
    def index():
        try:
            return render_template("index.html")
        except Exception:
            return jsonify({"message": "Heart Disease Analytics API", "version": "2.0.0"})

    logger.info("Healthcare Analytics Platform started")
    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
