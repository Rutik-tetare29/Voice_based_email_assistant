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
import re
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


# ── Number-word → digit table (covers 0-19 and tens up to 90) ─────────────────
_NUM_WORDS = {
    "zero":"0","one":"1","two":"2","three":"3","four":"4",
    "five":"5","six":"6","seven":"7","eight":"8","nine":"9",
    "ten":"10","eleven":"11","twelve":"12","thirteen":"13",
    "fourteen":"14","fifteen":"15","sixteen":"16","seventeen":"17",
    "eighteen":"18","nineteen":"19",
    "twenty":"20","thirty":"30","forty":"40","fifty":"50",
    "sixty":"60","seventy":"70","eighty":"80","ninety":"90",
}
# compound tens: "twenty one" … "ninety nine"
_TENS = ["twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]
_ONES = ["one","two","three","four","five","six","seven","eight","nine"]


def _replace_number_words(t: str) -> str:
    """Replace spoken number words with digits, e.g. 'twenty nine' → '29'."""
    # compound tens first ("twenty one" → "21")
    for ten in _TENS:
        for one in _ONES:
            t = re.sub(rf'\b{ten}\s+{one}\b',
                       str(int(_NUM_WORDS[ten]) + int(_NUM_WORDS[one])), t)
    # single words
    for word, digit in _NUM_WORDS.items():
        t = re.sub(rf'\b{word}\b', digit, t)
    return t


# ── Known domain spoken-form fixes ────────────────────────────────────────────
_DOMAIN_FIXES = [
    # Gmail variants
    (r'\bg\s*mail\b',       'gmail'),
    (r'\bgemail\b',         'gmail'),
    (r'\bg-mail\b',         'gmail'),
    # Hotmail / Outlook
    (r'\bhot\s*mail\b',     'hotmail'),
    (r'\bout\s*look\b',     'outlook'),
    # Yahoo
    (r'\byah+oo\b',         'yahoo'),
    # TLD: "com" mis-heard
    (r'\b(?:calm|come|comma|khan|con|gom|cam)\b', 'com'),
    # TLD: "in" short form
    (r'\b(?:inn|an|and)$',  'in'),
    # TLD: "net"
    (r'\b(?:naet|neat|met)\b', 'net'),
    # TLD: "org"
    (r'\b(?:org|aura|alba)\b', 'org'),
    # TLD: "edu"
    (r'\b(?:edu|eddo|ado)\b', 'edu'),
]


def _normalize_email_address(text: str) -> str:
    """
    Convert a spoken email address to a valid format.
    Handles the many ways Vosk (small model) mis-transcribes email components.

    Pronunciation guide spoken to the user:
        name  [at / at the rate / @ / add / hat]  domain  [dot / period / full stop]  tld
    """
    t = text.lower().strip()

    # ── 0. Number words → digits ──────────────────────────────────────────────
    t = _replace_number_words(t)

    # ── 1. Domain spoken-form fixes (before @ replacement so 'add' isn't
    #        accidentally turned into a digit) ─────────────────────────────────
    for pattern, replacement in _DOMAIN_FIXES:
        t = re.sub(pattern, replacement, t)

    # ── 2. @ substitutes ─────────────────────────────────────────────────────
    # "at the rate (of)?"
    t = re.sub(r'\bat\s+the\s+rate\s+(?:of\s+)?', '@', t)
    # "at sign" / "at symbol" / "@ symbol"
    t = re.sub(r'\bat\s+(?:sign|symbol|mark)\b', '@', t)
    # "commercial at"
    t = re.sub(r'\bcommercial\s+at\b', '@', t)
    # Vosk mis-hears "at" as "add" / "hat" / "that" / "had" / "rat" / "bat"
    t = re.sub(r'\b(?:add|hat|that|had|rat|bat|cat|fat|sat|@)\b', '@', t)
    # Plain "at" between two non-space sequences
    t = re.sub(r'(?<=\S)\s+at\s+(?=\S)', '@', t)
    # Remove stray "at" at start if it survived
    t = re.sub(r'^at\s+', '@', t)

    # ── 3. Dot substitutes ────────────────────────────────────────────────────
    # "full stop", "period", "point", "dot"
    t = re.sub(r'\s*\b(?:dot|period|full\s+stop|point|por)\b\s*', '.', t)

    # ── 4. Special character names ────────────────────────────────────────────
    t = re.sub(r'\s*\bunderscore\b\s*', '_', t)
    t = re.sub(r'\s*\b(?:dash|hyphen|minus)\b\s*', '-', t)
    t = re.sub(r'\s*\bplus\b\s*', '+', t)

    # ── 5. Strip filler words that creep in ───────────────────────────────────
    # "my email is", "send to", "the address is", etc.
    t = re.sub(r'^(?:my\s+)?(?:email\s+(?:is\s+|address\s+is\s+)?'  \
               r'|address\s+is\s+|send\s+(?:it\s+)?to\s+|to\s+)?', '', t)

    # ── 6. Collapse whitespace inside the address ─────────────────────────────
    # At this point anything left that is a space inside the email is wrong
    t = re.sub(r'\s+', '', t)

    # ── 7. Cleanup double punctuation / leading-trailing junk ─────────────────
    t = re.sub(r'\.{2,}', '.', t)   # ".." → "."
    t = re.sub(r'@{2,}', '@', t)    # "@@" → "@"
    t = t.strip('.@_-')

    return t


def _is_valid_email(addr: str) -> bool:
    """Basic sanity check — must have exactly one @ and at least one dot after it."""
    return bool(re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', addr))


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
        raw     = transcription.strip()
        to_addr = _normalize_email_address(raw)
        logger.info("Compose 'to': raw=%r  normalised=%r", raw, to_addr)

        retries = compose.get("to_retries", 0)

        if not _is_valid_email(to_addr):
            # Keep step as "to", increment retry counter
            new_retries = retries + 1
            session["email_compose"] = dict(compose, to_retries=new_retries)
            session.modified = True
            if new_retries >= 2:
                # After 2 failures, suggest typing
                return (
                    f"I heard: {raw!r} — that doesn't look like a valid email address. "
                    "Please type the address in the text box that appeared below the mic, "
                    "then say continue."
                )
            return (
                f"I heard: {raw!r} — that doesn't look like a valid email address. "
                "Please say it again clearly. For example: "
                "r u t i k at gmail dot com."
            )

        session["email_compose"] = dict(compose, to=to_addr, step="subject", to_retries=0)
        session.modified = True
        readable = to_addr.replace("@", " at ").replace(".", " dot ")
        return f"Got it — sending to {readable}. What is the subject?"

    # ── Step 2: subject ───────────────────────────────────────────────────────
    elif step == "subject":
        subject = transcription.strip()
        session["email_compose"] = dict(compose, subject=subject, step="body")
        session.modified = True
        return f"Subject: {subject}. What is your message?"

    # ── Step 3: body ─────────────────────────────────────────────────────────
    elif step == "body":
        body    = transcription.strip()
        session["email_compose"] = dict(compose, body=body, step="confirm")
        session.modified = True
        to      = compose["to"]
        subject = compose["subject"]
        readable_to = to.replace("@", " at ").replace(".", " dot ")
        return (
            f"Ready to send. "
            f"To: {readable_to}. Subject: {subject}. "
            f"Message: {body}. "
            f"Say yes or confirm to send, or cancel to abort."
        )

    # ── Step 4: confirm ───────────────────────────────────────────────────────
    elif step == "confirm":
        if _any_token_matches(lower, _CONFIRM_EXACT):
            to      = compose["to"]
            subject = compose["subject"]
            body    = compose["body"]
            session.pop("email_compose", None)
            session.modified = True
            try:
                success, message = send_email(session, to, subject, body)
                if success:
                    readable_to = to.replace("@", " at ").replace(".", " dot ")
                    return f"Email sent successfully to {readable_to}!"
                else:
                    logger.error("Send email returned failure: %s", message)
                    return f"Failed to send email. {message}. Please try again."
            except Exception as exc:
                logger.error("Send email exception: %s", exc)
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


# ── Text-input path for compose fields (bypasses STT) ─────────────────────────
def process_text_compose_input(field: str, value: str, session: dict) -> dict:
    """
    Accepts a typed value for one compose field and advances the flow.
    Returns the same dict shape as process_voice_command.
    """
    compose = session.get("email_compose")

    # Safety: if no compose session is active, start one
    if compose is None:
        session["email_compose"] = {"step": "to", "to": "", "subject": "", "body": ""}
        session.modified = True
        compose = session["email_compose"]

    response_text = ""

    if field == "to":
        # Validate the typed address (basic sanity, not full RFC)
        if not _is_valid_email(value):
            response_text = (
                f"'{value}' doesn't look like a valid email address. "
                "Please check and try again."
            )
        else:
            session["email_compose"] = dict(compose, to=value, step="subject", to_retries=0)
            session.modified = True
            readable = value.replace("@", " at ").replace(".", " dot ")
            response_text = f"Got it — sending to {readable}. Now say the subject."

    elif field == "subject":
        session["email_compose"] = dict(compose, subject=value, step="body")
        session.modified = True
        response_text = f"Subject: {value}. Now say your message."

    elif field == "body":
        to      = compose.get("to", "")
        subject = compose.get("subject", "")
        session["email_compose"] = dict(compose, body=value, step="confirm")
        session.modified = True
        readable_to = to.replace("@", " at ").replace(".", " dot ")
        response_text = (
            f"Ready to send. To: {readable_to}. Subject: {subject}. "
            f"Message: {value}. Say yes to confirm or cancel to abort."
        )

    elif field == "confirm":
        # value should be "yes" or similar — frontend already filtered "no"
        to      = compose.get("to", "")
        subject = compose.get("subject", "")
        body_v  = compose.get("body", "")
        session.pop("email_compose", None)
        session.modified = True
        try:
            success, message = send_email(session, to, subject, body_v)
            if success:
                readable_to = to.replace("@", " at ").replace(".", " dot ")
                response_text = f"Email sent successfully to {readable_to}!"
            else:
                response_text = f"Failed to send email. {message}. Please try again."
        except Exception as exc:
            logger.error("Text compose send error: %s", exc)
            response_text = "Sorry, I could not send the email. Please check your settings."

    else:
        response_text = "Unknown field."

    audio_url = None
    if response_text:
        tts_path = speak_to_file(response_text)
        if tts_path:
            audio_url = f"/static/audio/{os.path.basename(tts_path)}"

    return {
        "transcription": f"[typed] {value}",
        "intent":        "send_email",
        "response_text": response_text,
        "audio_url":     audio_url,
        "email_step":    (session.get("email_compose") or {}).get("step"),
    }
