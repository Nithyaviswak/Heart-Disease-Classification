"""Auth routes with input validation and rate limiting awareness."""

from backend.auth import token_required, role_required, login_with_google
from flask import Blueprint, request, jsonify, g
from backend.auth import register_user, login_user

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "Request body is required"}), 400

    result = register_user(
        username=data.get("username", ""),
        email=data.get("email", ""),
        password=data.get("password", ""),
        role=data.get("role", "doctor"),
        full_name=data.get("full_name", ""),
    )
    status = 201 if result["success"] else 400
    return jsonify(result), status


@auth_bp.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "Request body is required"}), 400

    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"success": False, "error": "Username and password are required"}), 400

    result = login_user(username, password)
    status = 200 if result["success"] else 401
    return jsonify(result), status


@auth_bp.route("/api/auth/google", methods=["POST"])
def google_login():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "Request body is required"}), 400

    id_token = data.get("credential", "")
    if not id_token:
        return jsonify({"success": False, "error": "Missing Google credential"}), 400
    result = login_with_google(id_token)
    status = 200 if result["success"] else 401
    return jsonify(result), status


@auth_bp.route("/api/auth/me")
@token_required
def me():
    return jsonify(g.current_user)
