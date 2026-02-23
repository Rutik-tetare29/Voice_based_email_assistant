import os
from dotenv import load_dotenv

# Always load .env from the project root regardless of CWD where Python is launched
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)


class Config:
    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

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

    # Whisper STT model
    # Options: tiny (~75MB), base (~145MB, recommended), small (~465MB), medium (~1.5GB)
    # Model is auto-downloaded to ~/.cache/whisper/ on first run.
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

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
