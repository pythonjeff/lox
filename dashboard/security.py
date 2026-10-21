"""Shared security extensions: CSRF protection and rate limiting.

Kept in its own module so blueprints (auth) can import the limiter without importing app.py.
Rate-limit state is in-process memory, which is correct for the Procfile's single gunicorn worker;
move `storage_uri` to Redis if the app is ever scaled to multiple workers/dynos.
"""
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect

csrf = CSRFProtect()
limiter = Limiter(key_func=get_remote_address, default_limits=[], storage_uri="memory://")
