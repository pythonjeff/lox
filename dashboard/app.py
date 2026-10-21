"""
LOX FUND Dashboard
Flask app for investor-facing P&L dashboard.

Memory-optimized for Heroku Basic (512MB):
- No background threads on startup
- Regime/Palmer/MC data loaded lazily on request
- Only 5 endpoints are used by the frontend
"""

import os
import sys

# Add parent directory to path to import lox modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load .env file explicitly (required for gunicorn)
from dotenv import load_dotenv
_env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(_env_path)
print(f"[Dashboard] Loaded .env from: {_env_path}")

from flask import Flask
from flask_login import LoginManager

from dashboard.models import db, bcrypt, User
from dashboard.security import csrf, limiter
from dashboard.auth import auth as auth_blueprint
from dashboard.routes.pages import pages
from dashboard.routes.positions_api import positions_api
from dashboard.routes.market_api import market_api

# ═══════════════════════════════════════════════════════════════
# Flask app setup
# ═══════════════════════════════════════════════════════════════
app = Flask(__name__)
_secret_key = os.environ.get("FLASK_SECRET_KEY")
if not _secret_key:
    # No guessable default: an unset key gets a random per-process one (sessions reset on
    # restart). Set FLASK_SECRET_KEY in production so sessions survive restarts/multiple workers.
    import secrets
    _secret_key = secrets.token_hex(32)
    print("[Dashboard] WARNING: FLASK_SECRET_KEY not set — using a temporary random key.")
app.config["SECRET_KEY"] = _secret_key

_db_url = os.environ.get(
    "DATABASE_URL",
    "sqlite:///" + os.path.join(os.path.dirname(__file__), "..", "data", "lox_users.db"),
)
if _db_url.startswith("postgres://"):
    _db_url = _db_url.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = _db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 300

# ═══════════════════════════════════════════════════════════════
# Extensions
# ═══════════════════════════════════════════════════════════════
db.init_app(app)
bcrypt.init_app(app)
csrf.init_app(app)
limiter.init_app(app)

# Heroku terminates TLS at its router: trust one proxy hop so rate limiting sees the real client IP
# and cookies/URLs use https.
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = bool(os.environ.get("DYNO"))  # only Heroku is https

login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message = "Please sign in to access the dashboard."
login_manager.login_message_category = "info"
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id: str):
    return User.query.get(int(user_id))


# ═══════════════════════════════════════════════════════════════
# Blueprints (only what the frontend uses)
# ═══════════════════════════════════════════════════════════════
app.register_blueprint(auth_blueprint)
app.register_blueprint(pages)
app.register_blueprint(positions_api)
app.register_blueprint(market_api)

# Create tables on first run
with app.app_context():
    db.create_all()
    print("[Dashboard] Database tables ready.")

# No background threads — all data is loaded lazily on request.
# This keeps memory under 512MB on Heroku Basic.
print("[Dashboard] Ready (lazy-load mode, no background threads)")


# Backward compat aliases (no-ops)
def start_background_threads():
    pass

start_palmer_background = start_background_threads


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)
