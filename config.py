import os
from dotenv import load_dotenv

# Always load .env from the project root regardless of CWD where Python is launched
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)


class Config:
    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    # Vosk model path — always resolve against BASE_DIR if relative
    _vosk_raw = os.getenv(
        "VOSK_MODEL_PATH",
        os.path.join(BASE_DIR, "model", "vosk-model-small-en-us-0.15"),
    )
    VOSK_MODEL_PATH = _vosk_raw if os.path.isabs(_vosk_raw) else os.path.join(BASE_DIR, _vosk_raw)

    # Audio temp storage
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "audio")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Google OAuth
    GOOGLE_CLIENT_SECRETS_FILE = os.getenv(
        "GOOGLE_CLIENT_SECRETS_FILE",
        os.path.join(BASE_DIR, "client_secrets.json"),
    )
    GOOGLE_SCOPES = [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send",
        "https://www.googleapis.com/auth/userinfo.email",
        "openid",
    ]
    OAUTHLIB_INSECURE_TRANSPORT = os.getenv("OAUTHLIB_INSECURE_TRANSPORT", "1")

    # Pinned OAuth redirect URI — must match exactly what is registered in
    # Google Cloud Console → APIs & Services → Credentials → Authorised redirect URIs
    GOOGLE_REDIRECT_URI = os.getenv(
        "GOOGLE_REDIRECT_URI",
        "http://127.0.0.1:5000/login/google/callback",
    )

    # Gmail IMAP / SMTP (App Password flow)
    GMAIL_IMAP_HOST = "imap.gmail.com"
    GMAIL_SMTP_HOST = "smtp.gmail.com"
    GMAIL_SMTP_PORT = 587
