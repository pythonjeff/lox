"""
LOX FUND - Database Models
SQLAlchemy ORM models for user authentication, session tracking, and invites.
"""

import uuid
from datetime import datetime, timezone, timedelta

from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import UserMixin

# ============ EXTENSIONS ============
# Initialized here, bound to app in app.py via init_app()
db = SQLAlchemy()
bcrypt = Bcrypt()


# ============ USER MODEL ============

class User(UserMixin, db.Model):
    """Registered user account."""

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(128), nullable=False)

    is_active = db.Column(db.Boolean, default=True, nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)

    # Link to investor ledger code (e.g. "JL", "AK") -- nullable for admin-only users
    investor_code = db.Column(db.String(10), nullable=True, index=True)

    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    last_login_at = db.Column(db.DateTime, nullable=True)

    # Relationship
    sessions = db.relationship("UserSession", back_populates="user", lazy="dynamic")

    # ------------------------------------------------------------------
    # Password helpers
    # ------------------------------------------------------------------

    def set_password(self, password: str) -> None:
        """Hash and store a plaintext password using bcrypt."""
        self.password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password: str) -> bool:
        """Verify a plaintext password against the stored hash."""
        return bcrypt.check_password_hash(self.password_hash, password)

    # ------------------------------------------------------------------
    # flask-login integration
    # ------------------------------------------------------------------

    @property
    def is_active_user(self):
        """flask-login calls get_id(); is_active comes from UserMixin."""
        return self.is_active

    def __repr__(self) -> str:
        return f"<User {self.email!r} (id={self.id})>"


# ============ SESSION MODEL ============

class UserSession(db.Model):
    """
    Tracks individual login sessions for audit and revocation.
    Separate from Flask's cookie-based session — this is the DB record.
    """

    __tablename__ = "sessions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    session_token = db.Column(
        db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4())
    )

    ip_address = db.Column(db.String(45), nullable=True)   # IPv4 or IPv6
    user_agent = db.Column(db.String(512), nullable=True)

    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    is_revoked = db.Column(db.Boolean, default=False, nullable=False)

    # Relationship
    user = db.relationship("User", back_populates="sessions")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    SESSION_LIFETIME_DAYS = 5

    @classmethod
    def create_for_user(cls, user: User, request) -> "UserSession":
        """Create a new session row from a Flask request."""
        session = cls(
            user_id=user.id,
            ip_address=request.remote_addr,
            user_agent=str(request.user_agent)[:512],
            expires_at=datetime.now(timezone.utc) + timedelta(days=cls.SESSION_LIFETIME_DAYS),
        )
        db.session.add(session)
        db.session.commit()
        return session

    def revoke(self) -> None:
        """Mark this session as revoked."""
        self.is_revoked = True
        db.session.commit()

    @property
    def is_expired(self) -> bool:
        now = datetime.utcnow()
        exp = self.expires_at.replace(tzinfo=None) if self.expires_at.tzinfo else self.expires_at
        return now > exp

    @property
    def is_valid(self) -> bool:
        return not self.is_revoked and not self.is_expired

    def __repr__(self) -> str:
        return f"<UserSession token={self.session_token[:8]}… user_id={self.user_id}>"


# ============ INVITE MODEL ============

class Invite(db.Model):
    """
    Investor invite token.
    Created by admin via CLI; consumed when the investor registers.
    """

    __tablename__ = "invites"

    INVITE_LIFETIME_DAYS = 30

    id = db.Column(db.Integer, primary_key=True)
    investor_code = db.Column(db.String(10), nullable=False)
    email = db.Column(db.String(255), nullable=False)
    token = db.Column(
        db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4())
    )

    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    accepted_at = db.Column(db.DateTime, nullable=True)  # set when investor registers

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, *, investor_code: str, email: str) -> "Invite":
        """Create and persist a new invite."""
        invite = cls(
            investor_code=investor_code.strip().upper(),
            email=email.strip().lower(),
            expires_at=datetime.now(timezone.utc) + timedelta(days=cls.INVITE_LIFETIME_DAYS),
        )
        db.session.add(invite)
        db.session.commit()
        return invite

    @property
    def is_expired(self) -> bool:
        now = datetime.utcnow()
        exp = self.expires_at.replace(tzinfo=None) if self.expires_at.tzinfo else self.expires_at
        return now > exp

    @property
    def is_accepted(self) -> bool:
        return self.accepted_at is not None

    @property
    def is_valid(self) -> bool:
        return not self.is_expired and not self.is_accepted

    def accept(self) -> None:
        """Mark this invite as used."""
        self.accepted_at = datetime.now(timezone.utc)
        db.session.commit()

    def __repr__(self) -> str:
        status = "accepted" if self.is_accepted else ("expired" if self.is_expired else "pending")
        return f"<Invite {self.investor_code} -> {self.email} ({status})>"
