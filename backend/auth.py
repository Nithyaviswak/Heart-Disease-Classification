"""Phase 10: JWT Authentication with role-based access + Google OAuth.

Security hardened:
- Uses werkzeug scrypt-based password hashing instead of raw SHA-256
- Input validation on registration (username, email, password strength)
- Role assignment restricted to safe defaults
"""

import os
import re
import logging
from datetime import datetime, timedelta, timezone
from functools import wraps

import jwt
from flask import request, jsonify, g
from werkzeug.security import generate_password_hash, check_password_hash

from backend.database import get_connection

logger = logging.getLogger(__name__)

SECRET = os.environ.get("JWT_SECRET_KEY", "jwt-change-this-in-production")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")

# Security: roles users are allowed to self-assign during registration
ALLOWED_SELF_ROLES = {"doctor", "patient"}

# Password policy
MIN_PASSWORD_LENGTH = 8


def _hash_password(password: str) -> str:
    """Hash password using werkzeug's scrypt-based hasher (secure, slow hash)."""
    return generate_password_hash(password, method="scrypt")


def _verify_password(password: str, stored: str) -> bool:
    """Verify password against stored hash.

    Supports both new werkzeug hashes and legacy SHA-256 hashes for
    backward compatibility during migration.
    """
    # New werkzeug-style hashes start with 'scrypt:' or 'pbkdf2:'
    if stored.startswith(("scrypt:", "pbkdf2:")):
        return check_password_hash(stored, password)

    # Legacy SHA-256 fallback: salt$hash
    try:
        import hashlib
        salt, h = stored.split("$", 1)
        if hashlib.sha256(f"{salt}{password}".encode()).hexdigest() == h:
            # Optionally rehash on next login (upgrade path)
            return True
        return False
    except (ValueError, ImportError):
        return False


def _validate_registration(username: str, email: str, password: str, role: str) -> dict | None:
    """Validate registration inputs. Returns error dict or None if valid."""
    if not username or len(username) < 3 or len(username) > 50:
        return {"success": False, "error": "Username must be 3-50 characters"}

    if not re.match(r"^[a-zA-Z0-9_]+$", username):
        return {"success": False, "error": "Username may only contain letters, numbers, and underscores"}

    if not email or not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return {"success": False, "error": "A valid email address is required"}

    if not password or len(password) < MIN_PASSWORD_LENGTH:
        return {"success": False, "error": f"Password must be at least {MIN_PASSWORD_LENGTH} characters"}

    if not re.search(r"[A-Za-z]", password) or not re.search(r"[0-9]", password):
        return {"success": False, "error": "Password must contain both letters and numbers"}

    if role not in ALLOWED_SELF_ROLES:
        return {"success": False, "error": f"Invalid role. Allowed: {', '.join(sorted(ALLOWED_SELF_ROLES))}"}

    return None


def register_user(
    username: str,
    email: str,
    password: str,
    role: str = "doctor",
    full_name: str = "",
) -> dict:
    # Validate inputs
    validation_error = _validate_registration(username, email, password, role)
    if validation_error:
        return validation_error

    # Sanitize full_name
    full_name = full_name.strip()[:100] if full_name else ""

    with get_connection() as conn:
        try:
            conn.execute(
                "INSERT INTO users (username, email, password_hash, role, full_name) VALUES (?, ?, ?, ?, ?)",
                (username.lower(), email.lower(), _hash_password(password), role, full_name),
            )
            return {"success": True, "message": "User registered"}
        except Exception as e:
            logger.warning("Registration failed for user '%s': %s", username, e)
            return {"success": False, "error": "Username or email already exists"}


def login_user(username: str, password: str) -> dict:
    if not username or not password:
        return {"success": False, "error": "Invalid credentials"}

    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, role, full_name FROM users WHERE username = ?",
            (username.lower(),),
        ).fetchone()
        if not row:
            return {"success": False, "error": "Invalid credentials"}
        if not _verify_password(password, row["password_hash"]):
            return {"success": False, "error": "Invalid credentials"}
        token = jwt.encode(
            {
                "user_id": row["id"],
                "username": row["username"],
                "role": row["role"],
                "exp": datetime.now(timezone.utc) + timedelta(hours=2),
            },
            SECRET,
            algorithm="HS256",
        )
        return {
            "success": True,
            "token": token,
            "user": {
                "id": row["id"],
                "username": row["username"],
                "role": row["role"],
                "full_name": row["full_name"],
            },
        }


def _issue_token(user_id: int, username: str, role: str) -> str:
    return jwt.encode(
        {
            "user_id": user_id,
            "username": username,
            "role": role,
            "exp": datetime.now(timezone.utc) + timedelta(hours=2),
        },
        SECRET,
        algorithm="HS256",
    )


def login_with_google(id_token: str) -> dict:
    """Verify a Google ID token and issue a JWT."""
    try:
        from google.oauth2 import id_token as google_id_token
        from google.auth.transport import requests as google_requests

        payload = google_id_token.verify_oauth2_token(
            id_token, google_requests.Request(), GOOGLE_CLIENT_ID
        )

        google_email = payload.get("email", "")
        google_name = payload.get("name", "")
        google_sub = payload.get("sub", "")

        if not google_email:
            return {"success": False, "error": "No email in Google token"}

        username = google_email.split("@")[0]

        with get_connection() as conn:
            row = conn.execute(
                "SELECT id, username, role, full_name FROM users WHERE email = ? OR username = ?",
                (google_email, username),
            ).fetchone()

            if row:
                user_id = row["id"]
                role = row["role"]
                full_name = row["full_name"] or google_name
            else:
                conn.execute(
                    "INSERT INTO users (username, email, password_hash, role, full_name) VALUES (?, ?, ?, ?, ?)",
                    (username, google_email, f"google${google_sub}", "doctor", google_name),
                )
                row = conn.execute(
                    "SELECT id, username, role, full_name FROM users WHERE email = ?",
                    (google_email,),
                ).fetchone()
                user_id = row["id"]
                role = row["role"]
                full_name = row["full_name"]

        token = _issue_token(user_id, username, role)
        return {
            "success": True,
            "token": token,
            "user": {
                "id": user_id,
                "username": username,
                "role": role,
                "full_name": full_name,
            },
        }
    except Exception as e:
        logger.error("Google auth failed: %s", e)
        # Don't leak internal error details to the client
        return {"success": False, "error": "Google authentication failed"}


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid token"}), 401
        token = auth.split(" ", 1)[1]
        try:
            payload = jwt.decode(token, SECRET, algorithms=["HS256"])
            g.current_user = payload
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except Exception:
            return jsonify({"error": "Invalid token"}), 401
        return f(*args, **kwargs)
    return decorated


def role_required(*roles: str):
    def decorator(f):
        @wraps(f)
        @token_required
        def decorated(*args, **kwargs):
            if g.current_user.get("role") not in roles:
                return jsonify({"error": "Insufficient permissions"}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator
