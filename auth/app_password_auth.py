"""
Gmail App Password login — authenticates via IMAP to verify credentials,
then stores them in the Flask session for SMTP/IMAP operations.
"""
import imaplib
from flask import Blueprint, request, jsonify, session, redirect, url_for
from flask_login import login_user, UserMixin
from config import Config

apppass_auth_bp = Blueprint("apppass_auth", __name__)


# ── User model ────────────────────────────────────────────────────────────────
class AppPasswordUser(UserMixin):
    def __init__(self, email: str):
        self.id = email
        self.email = email
        self.name = email.split("@")[0]
        self.auth_type = "app_password"

    @staticmethod
    def from_session(data: dict) -> "AppPasswordUser":
        return AppPasswordUser(email=data["email"])

    def to_dict(self) -> dict:
        return {
            "email": self.email,
            "name": self.name,
            "auth_type": "app_password",
            # NOTE: we never store the raw password in session —
            # it is kept only in the encrypted session cookie under "app_pass"
        }


# ── Route ─────────────────────────────────────────────────────────────────────
@apppass_auth_bp.route("/login/app-password", methods=["POST"])
def app_password_login():
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    # Verify credentials with IMAP
    try:
        mail = imaplib.IMAP4_SSL(Config.GMAIL_IMAP_HOST)
        mail.login(email, password)
        mail.logout()
    except imaplib.IMAP4.error:
        return jsonify({"error": "Invalid credentials"}), 401
    except Exception as exc:
        return jsonify({"error": f"Connection failed: {exc}"}), 503

    user = AppPasswordUser(email=email)
    session["user"] = user.to_dict()
    # Store encrypted password separately in session (Flask signs the cookie)
    session["app_pass"] = password
    login_user(user, remember=True)

    return jsonify({"message": "Login successful", "redirect": url_for("dashboard")}), 200
