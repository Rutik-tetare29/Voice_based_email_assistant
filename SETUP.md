# VoiceMail — Setup Guide

## 1. Install dependencies
```bash
cd voice_email_app
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## 2. Whisper model
No manual download needed. The Whisper model is **automatically downloaded** to `~/.cache/whisper/` on first run.
Default model: `base` (~145 MB). You can change it in `.env` (`WHISPER_MODEL=tiny/base/small/medium`).

## 3. Configure environment
```bash
copy .env.example .env
# Edit .env — set SECRET_KEY (and optionally WHISPER_MODEL)
```

## 4. Google OAuth setup (optional — for Gmail API login)
1. Go to https://console.cloud.google.com/
2. Create a project → Enable **Gmail API**
3. Credentials → OAuth 2.0 Client ID → Web Application
4. Add `http://localhost:5000/login/google/callback` as redirect URI
5. Download JSON → save as `voice_email_app/client_secrets.json`

## 5. Gmail App Password (optional — for SMTP/IMAP login)
1. Enable 2FA on your Google Account
2. Go to https://myaccount.google.com/apppasswords
3. Generate a password for "Mail" — use it in the login form

## 6. Run the app
```bash
python app.py
# Open http://localhost:5000
```

## Quick command reference
| Voice command | Action |
|---|---|
| "read my emails" | Fetches & reads inbox |
| "send an email" | Guides to compose form |
| "help" | Lists available commands |
| "logout" | Signs you out |
