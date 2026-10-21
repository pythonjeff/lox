"""/api/investors must never return another investor's row to a non-admin."""
from flask import Flask
from flask_login import LoginManager, UserMixin

import dashboard.cache as cache
import dashboard.investors as investors_mod
from dashboard.cache import INVESTORS_CACHE, INVESTORS_CACHE_LOCK, is_admin_request
from dashboard.routes.market_api import market_api

LEDGER = {
    "investors": [{"code": "AAA", "value": 100.0}, {"code": "BBB", "value": 200.0}],
    "total_capital": 300.0,
    "equity": 310.0,
}


class _User(UserMixin):
    def __init__(self, uid, investor_code=None, is_admin=False):
        self.id = uid
        self.investor_code = investor_code
        self.is_admin = is_admin


USERS = {
    "a": _User("a", "AAA"),
    "nocode": _User("nocode", None),
    "admin": _User("admin", None, is_admin=True),
}


def _client(monkeypatch, uid):
    app = Flask(__name__)
    app.config.update(SECRET_KEY="test", TESTING=True)
    lm = LoginManager(app)
    lm.user_loader(lambda i: USERS.get(i))
    app.register_blueprint(market_api)
    monkeypatch.setattr(investors_mod, "get_investor_data", lambda: LEDGER)
    with INVESTORS_CACHE_LOCK:
        INVESTORS_CACHE["data"], INVESTORS_CACHE["timestamp"] = None, None
    c = app.test_client()
    with c.session_transaction() as s:
        s["_user_id"] = uid
        s["_fresh"] = True
    return c


def _codes(resp):
    return [i["code"] for i in resp.get_json()["investors"]]


def test_investor_sees_only_own_row(monkeypatch):
    c = _client(monkeypatch, "a")
    assert _codes(c.get("/api/investors")) == ["AAA"]
    # second call is served from the shared cache and must still be scoped
    assert _codes(c.get("/api/investors")) == ["AAA"]


def test_user_without_code_sees_nobody(monkeypatch):
    assert _codes(_client(monkeypatch, "nocode").get("/api/investors")) == []


def test_admin_sees_everyone(monkeypatch):
    assert _codes(_client(monkeypatch, "admin").get("/api/investors")) == ["AAA", "BBB"]


def test_admin_secret_unset_denies(monkeypatch):
    monkeypatch.setattr(cache, "ADMIN_SECRET", None)
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "test"
    LoginManager(app).user_loader(lambda i: None)
    with app.test_request_context():
        assert is_admin_request("") is False
        assert is_admin_request("lox-admin-2026") is False
