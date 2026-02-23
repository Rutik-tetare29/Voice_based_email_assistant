"""
Voice command processor.

Pipeline:
    1. Save uploaded audio to disk as WAV
    2. Run Vosk STT → get transcription text
    3. Match an intent from the text
    4. Execute intent → produce response text
    5. Run pyttsx3 TTS → save response audio
    6. Return JSON payload
"""
import os
import uuid
import logging
import difflib
from werkzeug.datastructures import FileStorage

from services.stt_vosk import transcribe, _model as _vosk_model
from services.tts_engine import speak_to_file
from services.email_service import fetch_emails, send_email
from config import Config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Intent keyword tables
# Each list contains the canonical word PLUS every common Vosk mis-transcription
# for that word when spoken clearly by an Indian-English speaker with a small
# model (vosk-model-small-en-us-0.15).
# ─────────────────────────────────────────────────────────────────────────────

_INTENTS = {
    # send_email must be listed before read_email so that keywords like
    # "cent" (mishearing of "send") are matched before the generic "email"
    # substring in read_email grabs the whole phrase.
    "send_email": [
        "send", "cent", "sent", "sand", "ends",
        "compose", "composed",
        "write", "right", "wrote",
        "new email", "new mail",
    ],
    "read_email": [
        "read", "reed", "red", "raid", "rid",
        "check", "czech", "checked",
        "inbox", "in box",
        "emails", "email", "e-mail", "e mail", "mails", "mail",
        "show", "open", "get", "fetch", "list",
    ],
    "logout": [
        "logout", "log out", "log-out",
        "sign out", "sign-out",
        "bye", "by", "buy", "bi",
        "exit", "exist", "quite", "quit",
        "goodbye", "good bye",
    ],
    "help": [
        "help", "held", "heap", "hell",
        "what can", "commands", "command",
    ],
}

# ── Stop-reading signals ───────────────────────────────────────────────────────
# "stop" is often heard as: top, stock, shop, cop, drop, prop, stuff, step,
#  stoop, store, storm, sport, spot,ktop, scop, stab, stub, stomp, stoppe
_STOP_EXACT = {
    "stop", "top", "stock", "shop", "cop", "drop", "prop",
    "stuff", "step", "stoop", "store", "stopped", "stopping",
    "stab", "stub", "spot", "stomp", "pause", "paws", "halt",
    "quiet", "quite", "quite", "silence", "silent",
    "enough", "that's enough", "that is enough",
    "shut up", "be quiet", "stop it", "stop reading",
    "pause reading", "stop the email", "no more",
}

# ── Cancel signals ────────────────────────────────────────────────────────────
# "cancel" → council, console, consul, cancel, camel, counsel, counsel
_CANCEL_EXACT = {
    "cancel", "council", "console", "consul", "camel", "counsel",
    "cancelled", "cancelling",
    "abort", "a board", "aboard",
    "never mind", "nevermind", "never mine",
    "forget it", "forget", "forget that",
    "don't send", "do not send", "don't do it",
    "no", "nope", "nah", "not",
    "stop sending", "cancel email", "cancel sending", "cancel it",
}

# ── Confirm signals ───────────────────────────────────────────────────────────
# "yes" → yet, yes, yep, yeah, ya, yah, yea, jest, chest
# "confirm" → conform, conformed, confirmed
# "ok" → oak, okay, ok, o.k.
_CONFIRM_EXACT = {
    "yes", "yet", "yep", "yeah", "ya", "yah", "yea", "jest",
    "confirm", "confirmed", "conform", "conformed",
    "ok", "okay", "o.k.", "oak",
    "send it", "do it", "go ahead", "go", "proceed",
    "yes please", "please send", "absolutely", "sure", "correct",
}


def _fuzzy_match(word: str, targets: set, cutoff: float = 0.72) -> bool:
    """
    Return True if `word` is close enough to any string in `targets`.
    Uses SequenceMatcher ratio — tolerates 1-2 character substitutions.
    """
    word = word.strip()
    if word in targets:
        return True
    matches = difflib.get_close_matches(word, targets, n=1, cutoff=cutoff)
    return bool(matches)


def _any_token_matches(text: str, targets: set, cutoff: float = 0.72) -> bool:
    """
    Check every individual word in `text` AND the full phrase against `targets`.
    Short utterances (≤3 words) get a slightly more lenient cutoff, but
    never below 0.78 — prevents common words ('send','read') fuzzy-matching
    into the stop-word set.
    """
    words = text.lower().split()
    # short utterance → be more lenient, but floor at 0.78
    if len(words) <= 3:
        cutoff = min(cutoff, 0.78)

    # full phrase check
    if _fuzzy_match(text.lower(), targets, cutoff):
        return True
    # per-word check
    for w in words:
        if _fuzzy_match(w, targets, cutoff):
            return True
    return False


def _detect_intent(text: str, session: dict) -> str:
    lower = text.lower().strip()
    if not lower:
        return "unknown"

    # ── While email compose is active ────────────────────────────────────────
    if session.get("email_compose"):
        # Cancel beats everything
        if _any_token_matches(lower, _CANCEL_EXACT):
            return "cancel_email"
        # At confirm step, check for yes/no explicitly
        step = session["email_compose"].get("step")
        if step == "confirm":
            if _any_token_matches(lower, _CONFIRM_EXACT):
                return "send_email"   # processor handles actual send
            # Anything non-confirm at confirm step = cancel
            if _any_token_matches(lower, _CANCEL_EXACT):
                return "cancel_email"
        return "send_email"   # all other utterances feed the compose flow

    # ── Stop reading (checked before general intents) ─────────────────────────
    if _any_token_matches(lower, _STOP_EXACT):
        return "stop_reading"

    # ── Standard intent matching ──────────────────────────────────────────────
    # First try exact substring (fast path)
    for intent, keywords in _INTENTS.items():
        if any(kw in lower for kw in keywords):
            return intent

    # Fuzzy fallback — try every token against every keyword list
    words = lower.split()
    for intent, keywords in _INTENTS.items():
        kw_set = set(keywords)
        for w in words:
            if _fuzzy_match(w, kw_set, cutoff=0.70):
                return intent

    return "unknown"


# ── Intent handlers ────────────────────────────────────────────────────────────
def _handle_read_email(session: dict) -> str:
    # Fetch only the 1 most recent email
    emails = fetch_emails(session, limit=1)
    if not emails:
        return "Your inbox is empty or I could not retrieve your emails."

    # Read only the latest email with full content
    latest = emails[0]
    sender  = latest.get("from", "Unknown")
    subject = latest.get("subject", "No subject")
    body    = latest.get("body") or latest.get("snippet") or "No content"

    # Trim body to a reasonable spoken length (~800 chars)
    if len(body) > 800:
        body = body[:800] + "... message continues."

    return (
        f"Your latest email is from {sender}. "
        f"Subject: {subject}. "
        f"Message: {body}"
    )


def _handle_stop_reading() -> str:
    # Return empty string — frontend stops audio instantly, no TTS needed.
    # Generating TTS for "stop" adds ~500 ms latency for no benefit.
    return ""


def _handle_cancel_email(session: dict) -> str:
    session.pop("email_compose", None)
    return "Email cancelled. What else can I help you with?"


def _handle_send_email(session: dict, transcription: str) -> str:
    """Multi-step voice-guided email compose with cancel support at every step."""
    lower    = transcription.lower()
    compose  = session.get("email_compose")

    # ── Step 0: start the flow ────────────────────────────────────────────────
    if compose is None:
        session["email_compose"] = {"step": "to", "to": "", "subject": "", "body": ""}
        return (
            "Sure! Let's compose an email. "
            "Who would you like to send it to? Please say the recipient's email address."
        )

    step = compose["step"]

    # ── Step 1: recipient ─────────────────────────────────────────────────────
    if step == "to":
        to_addr = transcription.strip()
        session["email_compose"] = dict(compose, to=to_addr, step="subject")
        return f"Got it — sending to {to_addr}. What is the subject?"

    # ── Step 2: subject ───────────────────────────────────────────────────────
    elif step == "subject":
        subject = transcription.strip()
        session["email_compose"] = dict(compose, subject=subject, step="body")
        return f"Subject: {subject}. What is your message?"

    # ── Step 3: body ─────────────────────────────────────────────────────────
    elif step == "body":
        body = transcription.strip()
        session["email_compose"] = dict(compose, body=body, step="confirm")
        to      = compose["to"]
        subject = compose["subject"]
        return (
            f"Ready to send. "
            f"To: {to}. Subject: {subject}. "
            f"Message: {body}. "
            f"Say confirm to send, or cancel to abort."
        )

    # ── Step 4: confirm ───────────────────────────────────────────────────────
    elif step == "confirm":
        if _any_token_matches(lower, _CONFIRM_EXACT):
            to      = compose["to"]
            subject = compose["subject"]
            body    = compose["body"]
            session.pop("email_compose", None)
            try:
                send_email(session, to, subject, body)
                return f"Email sent successfully to {to}!"
            except Exception as exc:
                logger.error("Send email failed: %s", exc)
                return "Sorry, I could not send the email. Please check your settings and try again."
        else:
            # Didn't confirm → treat as implicit cancel
            session.pop("email_compose", None)
            return "Email cancelled."

    # fallback
    session.pop("email_compose", None)
    return "Something went wrong. Email compose reset. Please try again."


def _handle_logout() -> str:
    return "You have been logged out. Goodbye!"


def _handle_help() -> str:
    return (
        "You can say: read email, send email, logout, or help. "
        "I will carry out your request right away."
    )


def _handle_unknown(text: str) -> str:
    if not text:
        # Silence / nothing transcribed — return empty so no TTS fires
        return ""
    return f"I heard: {text}. I am not sure what you want. Try saying read email or send email."


# ── Main entry point ───────────────────────────────────────────────────────────
def process_voice_command(audio_file: FileStorage, session: dict) -> dict:
    """
    Accepts a Werkzeug FileStorage WAV upload, transcribes it,
    detects intent, generates a spoken response, and returns a dict:
        {
            "transcription": str,
            "intent": str,
            "response_text": str,
            "audio_url": str | None
        }
    """
    # 1 — Save raw upload
    temp_path = os.path.join(Config.UPLOAD_FOLDER, f"input_{uuid.uuid4().hex}.wav")
    audio_file.save(temp_path)

    # Early exit if Vosk model is not loaded
    if _vosk_model is None:
        tts_path = speak_to_file("Vosk model not loaded. Please download the model and set the path in your environment file.")
        audio_url = f"/static/audio/{os.path.basename(tts_path)}" if tts_path else None
        try: os.remove(temp_path)
        except OSError: pass
        return {
            "transcription": "",
            "intent": "error",
            "response_text": "Vosk model not loaded. Download a model from alphacephei.com/vosk/models and set VOSK_MODEL_PATH in your .env file.",
            "audio_url": audio_url,
        }

    # 2 — Transcribe
    transcription = transcribe(temp_path)
    logger.info("Transcription: %r", transcription)

    # 3 — Detect intent (context-aware: checks session for active email compose)
    intent = _detect_intent(transcription, session)

    # 4 — Execute intent
    intent_map = {
        "read_email":   lambda: _handle_read_email(session),
        "send_email":   lambda: _handle_send_email(session, transcription),
        "stop_reading": _handle_stop_reading,
        "cancel_email": lambda: _handle_cancel_email(session),
        "logout":       _handle_logout,
        "help":         _handle_help,
        "unknown":      lambda: _handle_unknown(transcription),
    }
    response_text = intent_map.get(intent, lambda: _handle_unknown(transcription))()

    # 5 — TTS (skip if response is empty, e.g. stop_reading)
    audio_url = None
    if response_text:
        tts_path = speak_to_file(response_text)
        if tts_path:
            audio_url = f"/static/audio/{os.path.basename(tts_path)}"

    # Clean up input file
    try:
        os.remove(temp_path)
    except OSError:
        pass

    return {
        "transcription": transcription,
        "intent":        intent,
        "response_text": response_text,
        "audio_url":     audio_url,
        # Tells the frontend which compose step we are on (or null)
        "email_step":    (session.get("email_compose") or {}).get("step"),
    }
