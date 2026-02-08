"""
LOX FUND - Database Models
SQLAlchemy ORM models for user authentication, session tracking, invites,
and regime history snapshots.
"""

import uuid
from datetime import datetime, timezone, timedelta, date as date_type

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
    first_name = db.Column(db.String(80), nullable=False, default="")
    last_name = db.Column(db.String(80), nullable=False, default="")
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


# ============ REGIME SNAPSHOT MODEL ============

class RegimeSnapshot(db.Model):
    """
    Daily snapshot of market regime state.
    Used to correlate trading performance with macro conditions.
    """

    __tablename__ = "regime_snapshots"

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, unique=True, nullable=False, index=True)
    regime = db.Column(db.String(20), nullable=False)  # RISK-ON, CAUTIOUS, RISK-OFF

    # Key macro indicators at snapshot time
    vix = db.Column(db.Float, nullable=True)
    hy_oas = db.Column(db.Float, nullable=True)          # basis points
    yield_10y = db.Column(db.Float, nullable=True)        # percent
    cpi_yoy = db.Column(db.Float, nullable=True)          # percent
    curve_2s10s = db.Column(db.Float, nullable=True)      # basis points

    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize snapshot to dict."""
        return {
            "date": self.date.isoformat(),
            "regime": self.regime,
            "vix": self.vix,
            "hy_oas": self.hy_oas,
            "yield_10y": self.yield_10y,
            "cpi_yoy": self.cpi_yoy,
            "curve_2s10s": self.curve_2s10s,
        }

    @classmethod
    def upsert(cls, *, snapshot_date: date_type, regime: str,
               vix: float = None, hy_oas: float = None,
               yield_10y: float = None, cpi_yoy: float = None,
               curve_2s10s: float = None) -> "RegimeSnapshot":
        """Insert or update a regime snapshot for a given date."""
        existing = cls.query.filter_by(date=snapshot_date).first()
        if existing:
            existing.regime = regime
            if vix is not None:
                existing.vix = vix
            if hy_oas is not None:
                existing.hy_oas = hy_oas
            if yield_10y is not None:
                existing.yield_10y = yield_10y
            if cpi_yoy is not None:
                existing.cpi_yoy = cpi_yoy
            if curve_2s10s is not None:
                existing.curve_2s10s = curve_2s10s
            db.session.commit()
            return existing
        snap = cls(
            date=snapshot_date,
            regime=regime,
            vix=vix,
            hy_oas=hy_oas,
            yield_10y=yield_10y,
            cpi_yoy=cpi_yoy,
            curve_2s10s=curve_2s10s,
        )
        db.session.add(snap)
        db.session.commit()
        return snap

    @classmethod
    def get_regime_for_date(cls, target_date: date_type) -> str | None:
        """Return regime label for a given date, or the closest prior date."""
        snap = cls.query.filter(cls.date <= target_date).order_by(cls.date.desc()).first()
        return snap.regime if snap else None

    @classmethod
    def get_all_ordered(cls) -> list["RegimeSnapshot"]:
        """Return all snapshots ordered by date ascending."""
        return cls.query.order_by(cls.date.asc()).all()

    @classmethod
    def get_regime_bands(cls) -> list[dict]:
        """Return contiguous regime bands [{start, end, regime}, ...]."""
        snaps = cls.get_all_ordered()
        if not snaps:
            return []
        bands = []
        current = {"start": snaps[0].date.isoformat(), "regime": snaps[0].regime}
        for s in snaps[1:]:
            if s.regime != current["regime"]:
                current["end"] = s.date.isoformat()
                bands.append(current)
                current = {"start": s.date.isoformat(), "regime": s.regime}
        # Close last band with today
        current["end"] = datetime.now(timezone.utc).date().isoformat()
        bands.append(current)
        return bands

    def __repr__(self) -> str:
        return f"<RegimeSnapshot {self.date} {self.regime} VIX={self.vix}>"
