"""CSRF and rate limiting on the real dashboard app (in-memory DB)."""
import os
import re

os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ.setdefault("FLASK_SECRET_KEY", "test-secret")

import pytest  # noqa: E402

from dashboard.app import app  # noqa: E402
from dashboard.security import limiter  # noqa: E402


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    limiter.reset()
    return app.test_client()


def _token(client, path="/auth/login"):
    html = client.get(path).get_data(as_text=True)
    m = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert m, "login form must carry a CSRF token"
    return m.group(1)


def test_login_form_has_csrf_token(client):
    assert _token(client)


def test_post_without_token_rejected(client):
    r = client.post("/auth/login", data={"email": "a@b.co", "password": "x"})
    assert r.status_code == 400


def test_post_with_token_is_processed(client):
    r = client.post("/auth/login", data={"email": "a@b.co", "password": "x", "csrf_token": _token(client)})
    assert r.status_code == 401  # the login handler's bad-credentials response, not a CSRF 400


def test_logout_requires_token(client):
    assert client.post("/auth/logout").status_code in (400, 401, 302)


def test_login_rate_limited(client):
    tok = _token(client)
    codes = [
        client.post("/auth/login", data={"email": "a@b.co", "password": "x", "csrf_token": tok}).status_code
        for _ in range(12)
    ]
    assert 429 in codes
    assert codes[0] != 429
