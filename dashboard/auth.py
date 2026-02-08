"""
LOX FUND - Authentication Blueprint
Handles user registration (invite-only), login, and logout.
"""

from datetime import datetime, timezone

from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user

from dashboard.models import db, User, UserSession, Invite

auth = Blueprint("auth", __name__, url_prefix="/auth")


# ============ LOGIN ============

@auth.route("/login", methods=["GET", "POST"])
def login():
    """Show login form / process login."""
    # Already authenticated — go to dashboard
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Email and password are required.", "error")
            return render_template("login.html"), 400

        user = User.query.filter_by(email=email).first()

        if user is None or not user.check_password(password):
            flash("Invalid email or password.", "error")
            return render_template("login.html"), 401

        if not user.is_active:
            flash("This account has been deactivated.", "error")
            return render_template("login.html"), 403

        # Record login timestamp
        user.last_login_at = datetime.now(timezone.utc)
        db.session.commit()

        # Create a DB session record for audit
        UserSession.create_for_user(user, request)

        # Log in via flask-login (sets session cookie)
        login_user(user, remember=True)

        # Redirect to the page the user originally wanted, or dashboard
        next_page = request.args.get("next")
        if next_page and _is_safe_redirect(next_page):
            return redirect(next_page)
        return redirect(url_for("my_account"))

    return render_template("login.html")


# ============ REGISTER (invite-only) ============

@auth.route("/register", methods=["GET", "POST"])
def register():
    """Invite-only registration. Requires a valid invite token in ?invite=<token>."""
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    # --- Validate invite token ---
    token = request.args.get("invite") or request.form.get("invite", "")
    if not token:
        flash("Registration requires an invite link. Contact your fund administrator.", "error")
        return redirect(url_for("auth.login"))

    invite = Invite.query.filter_by(token=token).first()
    if invite is None:
        flash("Invalid invite link.", "error")
        return redirect(url_for("auth.login"))

    if invite.is_accepted:
        flash("This invite has already been used.", "error")
        return redirect(url_for("auth.login"))

    if invite.is_expired:
        flash("This invite has expired. Contact your fund administrator for a new one.", "error")
        return redirect(url_for("auth.login"))

    if request.method == "POST":
        # Email comes from the invite (locked); name + password from form
        email = invite.email
        first_name = request.form.get("first_name", "").strip()
        last_name = request.form.get("last_name", "").strip()
        password = request.form.get("password", "")
        password_confirm = request.form.get("password_confirm", "")

        # --- Validation ---
        errors = []

        if not first_name:
            errors.append("First name is required.")
        if not last_name:
            errors.append("Last name is required.")
        if not password:
            errors.append("Password is required.")
        if len(password) < 8:
            errors.append("Password must be at least 8 characters.")
        if password != password_confirm:
            errors.append("Passwords do not match.")

        if User.query.filter_by(email=email).first():
            errors.append("An account with that email already exists.")

        if errors:
            for e in errors:
                flash(e, "error")
            return render_template(
                "register.html", invite=invite,
                first_name=first_name, last_name=last_name,
            ), 400

        # --- Create user linked to investor code ---
        # Username auto-derived from email (used internally, not shown to user)
        username = email.split("@")[0]
        if User.query.filter_by(username=username).first():
            import uuid as _uuid
            username = f"{username}_{_uuid.uuid4().hex[:6]}"

        user = User(
            email=email,
            username=username,
            first_name=first_name,
            last_name=last_name,
            investor_code=invite.investor_code,
        )
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        # Mark invite as used
        invite.accept()

        # Auto-login after registration
        user.last_login_at = datetime.now(timezone.utc)
        db.session.commit()

        UserSession.create_for_user(user, request)
        login_user(user, remember=True)

        flash(f"Welcome! Your account is linked to investor code {invite.investor_code}.", "success")
        return redirect(url_for("my_account"))

    # GET — render form pre-filled with invite data
    return render_template("register.html", invite=invite)


# ============ LOGOUT ============

@auth.route("/logout", methods=["POST"])
@login_required
def logout():
    """Revoke current DB session and log out."""
    # Revoke all active sessions for this user (simple approach)
    active_sessions = UserSession.query.filter_by(
        user_id=current_user.id, is_revoked=False
    ).all()
    for s in active_sessions:
        s.is_revoked = True
    db.session.commit()

    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))


# ============ HELPERS ============

def _is_safe_redirect(target: str) -> bool:
    """Prevent open-redirect attacks — only allow relative URLs."""
    from urllib.parse import urlparse

    parsed = urlparse(target)
    return parsed.scheme == "" and parsed.netloc == ""
