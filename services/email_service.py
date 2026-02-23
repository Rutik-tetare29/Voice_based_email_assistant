"""
Email service — fetch and send emails.

Supports two auth flows:
  • Google OAuth  → uses Gmail REST API via googleapiclient
  • App Password  → uses IMAP / SMTP directly
"""
import base64
import imaplib
import email as email_lib
import smtplib
import logging
import re
from email.mime.text import MIMEText
from email.header import decode_header
from html.parser import HTMLParser

from googleapiclient.discovery import build

from config import Config

logger = logging.getLogger(__name__)

MAX_EMAILS = 10


# ── Public API ─────────────────────────────────────────────────────────────────
def fetch_emails(session: dict, limit: int = MAX_EMAILS) -> list[dict]:
    """Return the latest `limit` emails as a list of dicts."""
    user_data = session.get("user", {})
    auth_type = user_data.get("auth_type")

    if auth_type == "google":
        return _fetch_gmail_api(session, limit)
    elif auth_type == "app_password":
        return _fetch_imap(
            email_addr=user_data["email"],
            password=session.get("app_pass", ""),
            limit=limit,
        )
    return []


def send_email(session: dict, to_addr: str, subject: str, body: str) -> tuple[bool, str]:
    """Send an email. Returns (success, message)."""
    user_data = session.get("user", {})
    auth_type = user_data.get("auth_type")

    if auth_type == "google":
        return _send_gmail_api(session, to_addr, subject, body)
    elif auth_type == "app_password":
        return _send_smtp(
            from_addr=user_data["email"],
            password=session.get("app_pass", ""),
            to_addr=to_addr,
            subject=subject,
            body=body,
        )
    return False, "Unknown auth type"


# ── Gmail API (OAuth) ──────────────────────────────────────────────────────────
def _gmail_service(session: dict):
    from auth.google_auth import GoogleUser
    user = GoogleUser.from_session(session["user"])
    creds = user.get_credentials()
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def _fetch_gmail_api(session: dict, limit: int = MAX_EMAILS) -> list[dict]:
    try:
        service = _gmail_service(session)
        result = (
            service.users()
            .messages()
            .list(userId="me", maxResults=limit, labelIds=["INBOX"])
            .execute()
        )
        messages = result.get("messages", [])
        emails = []
        for msg in messages:
            msg_data = (
                service.users()
                .messages()
                .get(userId="me", id=msg["id"], format="full")
                .execute()
            )
            headers = {
                h["name"]: h["value"]
                for h in msg_data.get("payload", {}).get("headers", [])
            }
            body = _extract_gmail_body(msg_data.get("payload", {}))
            emails.append({
                "id": msg["id"],
                "from": headers.get("From", "Unknown"),
                "subject": headers.get("Subject", "(No subject)"),
                "date": headers.get("Date", ""),
                "body": body,
                "snippet": body[:200] if body else msg_data.get("snippet", ""),
            })
        return emails
    except Exception as exc:
        logger.error("Gmail API fetch error: %s", exc)
        return []


def _extract_gmail_body(payload: dict) -> str:
    """Recursively extract plain-text body from a Gmail API payload part."""
    mime = payload.get("mimeType", "")
    parts = payload.get("parts", [])

    if mime == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")

    if mime == "text/html":
        data = payload.get("body", {}).get("data", "")
        if data:
            html = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
            return _html_to_text(html)

    # Recurse into multipart
    plain = ""
    html_fallback = ""
    for part in parts:
        t = part.get("mimeType", "")
        if t == "text/plain":
            plain = _extract_gmail_body(part)
        elif t == "text/html":
            html_fallback = _extract_gmail_body(part)
        elif t.startswith("multipart/"):
            result = _extract_gmail_body(part)
            if result:
                plain = result
    return plain or html_fallback


def _send_gmail_api(session: dict, to_addr: str, subject: str, body: str) -> tuple[bool, str]:
    try:
        service = _gmail_service(session)
        msg = MIMEText(body)
        msg["to"] = to_addr
        msg["subject"] = subject
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        service.users().messages().send(
            userId="me", body={"raw": raw}
        ).execute()
        return True, "Email sent successfully"
    except Exception as exc:
        logger.error("Gmail API send error: %s", exc)
        return False, str(exc)


# ── IMAP (App Password) ────────────────────────────────────────────────────────
def _decode_header_value(value: str) -> str:
    decoded, encoding = decode_header(value)[0]
    if isinstance(decoded, bytes):
        return decoded.decode(encoding or "utf-8", errors="replace")
    return decoded


def _fetch_imap(email_addr: str, password: str, limit: int = MAX_EMAILS) -> list[dict]:
    try:
        mail = imaplib.IMAP4_SSL(Config.GMAIL_IMAP_HOST)
        mail.login(email_addr, password)
        mail.select("inbox")

        _, data = mail.search(None, "ALL")
        ids = data[0].split()
        latest_ids = ids[-limit:] if len(ids) >= limit else ids
        latest_ids = list(reversed(latest_ids))  # newest first

        emails = []
        for eid in latest_ids:
            _, msg_data = mail.fetch(eid, "(RFC822)")
            raw = msg_data[0][1]
            msg = email_lib.message_from_bytes(raw)
            body = _get_body(msg)
            emails.append({
                "id": eid.decode(),
                "from": _decode_header_value(msg.get("From", "Unknown")),
                "subject": _decode_header_value(msg.get("Subject", "(No subject)")),
                "date": msg.get("Date", ""),
                "body": body,
                "snippet": body[:200] if body else "",
            })

        mail.logout()
        return emails

    except Exception as exc:
        logger.error("IMAP fetch error: %s", exc)
        return []


class _HTMLStripper(HTMLParser):
    """Minimal HTML → plain text converter."""
    def __init__(self):
        super().__init__()
        self._parts = []
    def handle_data(self, data):
        self._parts.append(data)
    def get_text(self):
        return re.sub(r'\n{3,}', '\n\n', "\n".join(
            p.strip() for p in self._parts if p.strip()
        ))


def _html_to_text(html: str) -> str:
    """Strip HTML tags and return readable plain text."""
    try:
        stripper = _HTMLStripper()
        stripper.feed(html)
        return stripper.get_text()
    except Exception:
        # Fallback: crude regex strip
        return re.sub(r'<[^>]+>', ' ', html).strip()


def _get_body(msg) -> str:
    """Return the full plain-text body of an email.message.Message object."""
    plain = ""
    html  = ""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain" and not plain:
                payload = part.get_payload(decode=True)
                if payload:
                    plain = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
            elif ct == "text/html" and not html:
                payload = part.get_payload(decode=True)
                if payload:
                    html = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            text = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
            if msg.get_content_type() == "text/html":
                html = text
            else:
                plain = text

    if plain:
        return plain.strip()
    if html:
        return _html_to_text(html)
    return "(No message body)"
    return ""


# ── SMTP (App Password) ────────────────────────────────────────────────────────
def _send_smtp(
    from_addr: str, password: str, to_addr: str, subject: str, body: str
) -> tuple[bool, str]:
    try:
        msg = MIMEText(body)
        msg["From"] = from_addr
        msg["To"] = to_addr
        msg["Subject"] = subject

        with smtplib.SMTP(Config.GMAIL_SMTP_HOST, Config.GMAIL_SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(from_addr, password)
            server.sendmail(from_addr, to_addr, msg.as_string())

        return True, "Email sent successfully"
    except smtplib.SMTPAuthenticationError:
        return False, "Authentication failed — check app password"
    except Exception as exc:
        logger.error("SMTP send error: %s", exc)
        return False, str(exc)
