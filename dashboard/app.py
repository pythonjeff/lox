"""
LOX FUND Dashboard
Flask app for investor-facing P&L dashboard (updates every 5 minutes).
Palmer analysis is server-cached and refreshes every 30 minutes automatically.
"""

import os
import sys
import threading
import time

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
from dashboard.auth import auth as auth_blueprint
from dashboard.routes.pages import pages
from dashboard.routes.positions_api import positions_api
from dashboard.routes.regime_api import regime_api
from dashboard.routes.market_api import market_api
from dashboard.routes.lii_api import lii_api

# ═══════════════════════════════════════════════════════════════
# Flask app setup
# ═══════════════════════════════════════════════════════════════
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "lox-dev-secret-change-me")

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

login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message = "Please sign in to access the dashboard."
login_manager.login_message_category = "info"
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id: str):
    return User.query.get(int(user_id))


# ═══════════════════════════════════════════════════════════════
# Blueprints
# ═══════════════════════════════════════════════════════════════
app.register_blueprint(auth_blueprint)
app.register_blueprint(pages)
app.register_blueprint(positions_api)
app.register_blueprint(regime_api)
app.register_blueprint(market_api)
app.register_blueprint(lii_api)

# Create tables on first run
with app.app_context():
    db.create_all()
    print("[Dashboard] Database tables ready.")


# ═══════════════════════════════════════════════════════════════
# Background threads
# ═══════════════════════════════════════════════════════════════
_background_threads_started = False


def start_background_threads():
    """Initialize caches and start background refresh threads."""
    global _background_threads_started
    if _background_threads_started:
        return
    _background_threads_started = True

    from dashboard.cache import PALMER_REFRESH_INTERVAL, MC_REFRESH_INTERVAL
    from dashboard.palmer import refresh_palmer_cache, palmer_background_refresh
    from dashboard.monte_carlo import refresh_mc_cache, mc_background_refresh

    print(f"[Palmer] Starting background refresh (interval: {PALMER_REFRESH_INTERVAL}s)")
    print(f"[MC] Starting background refresh (interval: {MC_REFRESH_INTERVAL}s = 1 hour)")
    print(f"[Regimes] Starting background prefetch (interval: 540s = 9 min)")

    # Initial refreshes
    threading.Thread(target=lambda: refresh_palmer_cache(app=app), daemon=True).start()
    threading.Thread(target=refresh_mc_cache, daemon=True).start()

    # Regime cache warm + history backfill
    def deferred_regime_init():
        try:
            from lox.config import load_settings
            from dashboard.regime_utils import _build_regime_cache
            settings = load_settings()
            _build_regime_cache(settings, refresh=False)
            print("[Regimes] Initial cache warmed")
        except Exception as e:
            print(f"[Regimes] Initial cache warm failed (non-fatal): {e}")
        try:
            from dashboard.regime_history import backfill_regime_history
            backfill_regime_history(app)
        except Exception as e:
            print(f"[RegimeHistory] Backfill error (non-fatal): {e}")

    threading.Thread(target=deferred_regime_init, daemon=True).start()

    # Recurring refresh threads
    threading.Thread(
        target=lambda: palmer_background_refresh(app=app), daemon=True
    ).start()
    threading.Thread(target=mc_background_refresh, daemon=True).start()

    def regime_background_refresh():
        while True:
            try:
                time.sleep(540)
                from lox.config import load_settings
                from dashboard.regime_utils import _build_regime_cache
                settings = load_settings()
                _build_regime_cache(settings, refresh=False)
                print("[Regimes] Background cache refreshed")
            except Exception as e:
                print(f"[Regimes] Background refresh error: {e}")
                time.sleep(60)

    threading.Thread(target=regime_background_refresh, daemon=True).start()


# Backward compat alias
start_palmer_background = start_background_threads

# ═══════════════════════════════════════════════════════════════
# Startup
# ═══════════════════════════════════════════════════════════════
start_background_threads()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)
