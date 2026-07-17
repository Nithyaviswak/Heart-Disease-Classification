import os
import secrets
import logging
from datetime import timedelta


class Config:
    # Security: generate a strong random fallback for development only.
    # In production (FLASK_ENV=production), SECRET_KEY and JWT_SECRET_KEY
    # MUST be set via environment variables — the app will refuse to start
    # with weak/default secrets in production mode.
    _is_production = os.environ.get("FLASK_ENV", "development") == "production"

    _default_secret = secrets.token_hex(32) if not _is_production else None
    _default_jwt = secrets.token_hex(32) if not _is_production else None

    SECRET_KEY = os.environ.get("SECRET_KEY", _default_secret)
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", _default_jwt)

    # Fail hard if secrets are missing in production
    if _is_production and (not SECRET_KEY or not JWT_SECRET_KEY):
        raise RuntimeError(
            "SECURITY ERROR: SECRET_KEY and JWT_SECRET_KEY environment variables "
            "must be set in production. Generate with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )

    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", "sqlite:///heart_disease.db"
    )
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=2)
    GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "..", "reports")
    MODELS_FOLDER = os.path.join(os.path.dirname(__file__), "..", "models")
    DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data")
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
    POWERBI_FOLDER = os.path.join(os.path.dirname(__file__), "..", "powerbi")

    # CORS allowed origins (restrict in production)
    CORS_ORIGINS = os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5000" if not _is_production else ""
    )
