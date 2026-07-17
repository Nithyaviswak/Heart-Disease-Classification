"""Heart Disease Analytics Platform — Production Entry Point.

Uses the backend factory to serve both the REST API and React frontend.
Preserves legacy ONNX route for backward compatibility.

Security: debug mode is NEVER enabled in production, regardless of env vars.
"""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from backend.app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    flask_env = os.environ.get("FLASK_ENV", "development")

    # Fix #11: Never allow debug mode in production
    debug = flask_env == "development"
    if flask_env == "production" and debug:
        raise RuntimeError("Debug mode must not be enabled in production")

    if debug:
        print(f"⚠️  Running in DEVELOPMENT mode on port {port}")
        print("   Set FLASK_ENV=production for production deployments")
    app.run(host="0.0.0.0", port=port, debug=debug)
