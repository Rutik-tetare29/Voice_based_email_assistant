"""
Google OAuth 2.0 authentication using google-auth-oauthlib.
Stores OAuth credentials in Flask session so they survive requests.
"""
from __future__ import annotations
import json
from flask import Blueprint, redirect, request, session, url_for
from flask_login import login_user, UserMixin
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import google.oauth2.id_token
import google.auth.transport.requests as google_requests
from config import Config

google_auth_bp = Blueprint("google_auth", __name__)


# ── User model ────────────────────────────────────────────────────────────────
class GoogleUser(UserMixin):
    """Lightweight user object stored in the Flask session."""

    def __init__(self, email: str, name: str, credentials_dict: dict):
        self.id = email
        self.email = email
        self.name = name
        self.credentials_dict = credentials_dict  # serialised google Credentials
        self.auth_type = "google"

    # Rebuild from session dict
    @staticmethod
    def from_session(data: dict) -> "GoogleUser":
        return GoogleUser(
            email=data["email"],
            name=data["name"],
            credentials_dict=data["credentials"],
        )

    def to_dict(self) -> dict:
        return {
            "email": self.email,
            "name": self.name,
            "credentials": self.credentials_dict,
            "auth_type": "google",
        }

    def get_credentials(self) -> Credentials:
        """Return a refreshed Credentials object."""
        creds = Credentials(**self.credentials_dict)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            # Persist refreshed token back to session
            session["user"]["credentials"] = _creds_to_dict(creds)
        return creds


# ── Helper ────────────────────────────────────────────────────────────────────
def _build_flow(state: str = None) -> Flow:
    flow = Flow.from_client_secrets_file(
        Config.GOOGLE_CLIENT_SECRETS_FILE,
        scopes=Config.GOOGLE_SCOPES,
        state=state,
    )
    flow.redirect_uri = url_for("google_auth.oauth_callback", _external=True)
    return flow


def _creds_to_dict(creds: Credentials) -> dict:
    return {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes or []),
    }


# ── Routes ─────────────────────────────────────────────────────────────────────
@google_auth_bp.route("/login/google")
def google_login():
    flow = _build_flow()
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    session["oauth_state"] = state
    return redirect(auth_url)


@google_auth_bp.route("/login/google/callback")
def oauth_callback():
    state = session.get("oauth_state", "")
    flow = _build_flow(state=state)

    try:
        flow.fetch_token(authorization_response=request.url)
    except Exception as exc:
        return {"error": f"OAuth token exchange failed: {exc}"}, 400

    creds = flow.credentials
    id_info = google.oauth2.id_token.verify_oauth2_token(
        creds.id_token,
        google_requests.Request(),
        clock_skew_in_seconds=10,
    )

    user = GoogleUser(
        email=id_info["email"],
        name=id_info.get("name", id_info["email"]),
        credentials_dict=_creds_to_dict(creds),
    )
    session["user"] = user.to_dict()
    login_user(user, remember=True)

    return redirect(url_for("dashboard"))
