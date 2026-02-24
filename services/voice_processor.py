"""
Voice command processor.

Pipeline:
    1. Save uploaded audio to disk as WAV
    2. Run Whisper STT → get transcription text
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

from services.stt_whisper import transcribe, _model as _whisper_model
from services.tts_engine import speak_to_file
from services.email_service import fetch_emails, send_email
from config import Config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Intent keyword tables
# Each list contains the canonical word PLUS every common Whisper mis-transcription
# for that word when spoken clearly by an Indian-English speaker.
# ─────────────────────────────────────────────────────────────────────────────

_INTENTS = {
    # ── Email navigation — listed FIRST so they override read/send keywords ──
    "list_emails": [
        "list emails", "list email", "list my emails", "list my email",
        "show emails", "show email", "show inbox", "show my emails",
        "what emails", "how many emails", "emails in inbox",
        "check inbox", "check my inbox", "what is in my inbox",
        "whats in my inbox", "inbox summary", "email summary",
    ],
    "next_email": [
        # ── single-word shortcuts (must appear before multi-word so substring
        #    check hits correctly when user says just "next") ──────────────────
        "next email", "next mail", "read next", "read next email",
        "the next one", "next one", "go next", "move next",
        "forward", "forwards",
        "email 2", "email two", "second email",
        "email 3", "email three", "third email",
        "email 4", "email four", "fourth email",
        "email 5", "email five", "fifth email",
        "next",                   # bare "next"
    ],
    "prev_email": [
        "previous email", "previous mail", "read previous",
        "go back", "the previous one", "email before",
        "earlier email", "before that", "read previous email",
        "previous", "prev",       # bare single-word forms
        "back",
    ],
    "read_more": [
        "read more", "continue reading", "more of this", "keep reading",
        "rest of the email", "rest of email", "keep going",
        "read the rest", "what else", "more please",
        "continue",               # bare single-word form
        "more",                   # bare "more"
        "next part", "next chunk",
    ],
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
    Handles the many ways Whisper mis-transcribes email components.

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
    # Whisper sometimes mis-hears "at" as "add" / "hat" / "that" / "had" / "rat" / "bat"
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
    # Use a null-byte placeholder for intentional dashes (spoken as "dash"/"hyphen")
    # so they survive the Whisper-separator removal in step 4b below.
    t = re.sub(r'\s*\b(?:dash|hyphen|minus)\b\s*', '\x00', t)
    t = re.sub(r'\s*\bplus\b\s*', '+', t)

    # ── 4b. Strip Whisper-inserted letter-separator hyphens ───────────────────
    # When the user spells out their email letter-by-letter, Whisper inserts
    # hyphens between the letters: "rutikte-t-e-t-k-r-e" → "rutiktetekre".
    # Intentional dashes were protected as '\x00' in step 4, so we can safely
    # strip all remaining bare hyphens from the local part only.
    # (Domain hyphens like "x-y.com" come from the spoken domain text and are
    # rarely user-spoken letter-by-letter, but to be safe we only strip the local.)
    if '@' in t:
        _lp, _rest = t.split('@', 1)
        t = _lp.replace('-', '') + '@' + _rest
    else:
        t = t.replace('-', '')

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
    t = t.strip('.@_-\x00')

    # ── 8. Restore intentional dashes (spoken as "dash"/"hyphen") ─────────────
    t = t.replace('\x00', '-')

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

    # ── "read email N" / "email number N" — positional navigation ────────────
    _num_map = {
        "one": 1, "1": 1, "two": 2, "2": 2, "three": 3, "3": 3,
        "four": 4, "4": 4, "five": 5, "5": 5,
    }
    m = re.search(
        r'(?:read|open|show|play)\s+(?:email|mail|message|number|no\.?)\s*'
        r'(one|two|three|four|five|1|2|3|4|5)\b',
        lower
    )
    if not m:
        m = re.search(r'(?:email|message)\s+(?:number\s+)?(one|two|three|four|five|1|2|3|4|5)\b', lower)
    if m:
        session["_goto_email_idx"] = _num_map.get(m.group(1), 1) - 1
        session.modified = True
        return "next_email"

    # ── Standard intent matching — list_emails / next / prev / read_more come
    #    first in _INTENTS so they match before read_email's generic keywords ─
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


# ── Email reading helpers ──────────────────────────────────────────────────────
_CHUNK_SIZE = 400   # chars per spoken navigation chunk (~30 s at 165 WPM)

# Server-side cache to avoid Flask session-cookie overflow (4 KB limit).
# Keyed by user email address — survives the request lifetime of the process.
_EMAIL_STORE: dict[str, list] = {}


def _store_key(session: dict) -> str:
    """Return a cache key unique to the logged-in user."""
    # Primary: session["user"]["email"] (set by both GoogleUser and AppPasswordUser)
    user_dict = session.get("user") or {}
    email = user_dict.get("email") if isinstance(user_dict, dict) else None
    return email or session.get("user_email") or session.get("email") or "anon"


def _cache_emails(session: dict, limit: int = 5) -> list:
    """Fetch emails and store in _EMAIL_STORE (NOT in the cookie-based session)."""
    emails = fetch_emails(session, limit=limit)
    key = _store_key(session)
    _EMAIL_STORE[key] = emails
    # Only store lightweight navigation pointers in the session cookie
    session["_email_cache_key"]  = key
    session["_email_read_idx"]   = 0
    session["_email_read_chunk"] = 0
    session.modified = True
    return emails


def _get_cached_emails(session: dict) -> list | None:
    """Return the cached email list for this user, or None if not cached."""
    key = session.get("_email_cache_key") or _store_key(session)
    return _EMAIL_STORE.get(key)


# ── TTS-safe text helpers ──────────────────────────────────────────────────────

def _clean_sender(from_str: str) -> str:
    """
    Convert a raw RFC-2822 From header into a short, TTS-safe spoken form.

    Examples
    --------
    '"Do not reply" <no-reply@iirs.gov.in>'  →  'no-reply at iirs.gov.in'
    'Rutik Tetare <rutik@gmail.com>'          →  'Rutik Tetare'
    'rutik@gmail.com'                         →  'rutik at gmail.com'
    """
    s = from_str.strip()

    # 1. Extract display name and address parts
    m = re.match(r'^(.*?)<([^>]+)>', s)
    if m:
        display = m.group(1).strip().strip('"').strip("'").strip()
        addr    = m.group(2).strip()
        # If display name is meaningful (not empty / not equal to addr), use it
        if display and display.lower() != addr.lower() and len(display) > 1:
            # Limit to first 60 chars to avoid absurdly long TTS intros
            display = display[:60].rstrip()
            return _tts_safe(display)
        # Otherwise speak the address in a readable way
        return _tts_safe(addr.replace("@", " at ").replace(".", " dot "))
    # No angle brackets — might be a plain address or plain name
    if "@" in s:
        return _tts_safe(s.replace("@", " at ").replace(".", " dot "))
    return _tts_safe(s[:80])


def _tts_safe(text: str) -> str:
    """
    Strip characters that confuse pyttsx3/SAPI5 SSML parser and clean up
    whitespace so the engine produces reliable untruncated audio.

    SAPI5 interprets < > as XML/SSML tags; encountering a malformed tag
    silently aborts audio generation — hence the 'stops at sender' bug.
    """
    # Remove SSML/XML angle-bracket constructs entirely
    text = re.sub(r'<[^>]*>', ' ', text)
    # Replace remaining stray < > & with safe equivalents
    text = text.replace('&', ' and ').replace('<', ' ').replace('>', ' ')
    # Strip markdown-style formatting
    text = re.sub(r'[*_`#~]', '', text)
    # Collapse repeated punctuation
    text = re.sub(r'[.]{2,}', '.', text)
    text = re.sub(r'[-]{2,}', '-', text)
    # URLs are unreadable — replace with "link"
    text = re.sub(r'https?://\S+', 'link', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _read_email_at(email: dict, idx: int, total: int, chunk: int = 0) -> str:
    """Speak one email, paginating the body into chunks."""
    sender  = _clean_sender(email.get("from", "Unknown"))
    subject = _tts_safe(email.get("subject", "No subject"))
    body    = _tts_safe(
        (email.get("body") or email.get("snippet") or "No content").strip()
    )

    start     = chunk * _CHUNK_SIZE
    body_part = body[start : start + _CHUNK_SIZE]
    has_more  = len(body) > start + _CHUNK_SIZE

    ordinals = ["first", "second", "third", "fourth", "fifth"]
    label    = ordinals[idx] if idx < len(ordinals) else f"email {idx + 1}"

    if chunk == 0:
        result = (
            f"Reading your {label} email. "
            f"From: {sender}. "
            f"Subject: {subject}. "
            f"Message: {body_part}"
        )
    else:
        result = f"Continuing email {idx + 1}. {body_part}"

    if has_more:
        result += " Say 'read more' to continue."
    else:
        if idx < total - 1:
            result += f" End of message. Say 'next' for email {idx + 2}."
        else:
            result += " That was your last email."
    return result


# ── Intent handlers ────────────────────────────────────────────────────────────
def _handle_list_emails(session: dict) -> str:
    """List subjects + senders so user knows what's in inbox before reading."""
    emails = _cache_emails(session, limit=5)
    if not emails:
        return "Your inbox is empty or I could not retrieve your emails."
    lines = []
    for i, e in enumerate(emails, 1):
        lines.append(
            f"Email {i}: from {_clean_sender(e.get('from', 'Unknown'))}. "
            f"Subject: {_tts_safe(e.get('subject', 'No subject'))}."
        )
    return (
        f"You have {len(emails)} email{'s' if len(emails) > 1 else ''} loaded. "
        + " ".join(lines)
        + " Say 'read email 1' or 'next' to read them."
    )


def _handle_read_email(session: dict) -> str:
    """Read the first (latest) email and cache all 5 for navigation."""
    emails = _cache_emails(session, limit=5)
    if not emails:
        return "Your inbox is empty or I could not retrieve your emails."
    return _read_email_at(emails[0], 0, len(emails), chunk=0)


def _handle_next_email(session: dict) -> str:
    """Read the next email, or a specific one if _goto_email_idx was set."""
    emails = _get_cached_emails(session)
    if not emails:
        emails = _cache_emails(session, limit=5)

    # Honour positional jump ("read email 3")
    goto = session.pop("_goto_email_idx", None)
    if goto is not None:
        idx = int(goto)
    else:
        idx = session.get("_email_read_idx", 0) + 1

    if idx >= len(emails):
        return (
            f"You've reached the end. There are only {len(emails)} emails loaded. "
            "Say 'list emails' to hear the subjects again."
        )

    session["_email_read_idx"]   = idx
    session["_email_read_chunk"] = 0
    session.modified = True
    return _read_email_at(emails[idx], idx, len(emails), chunk=0)


def _handle_prev_email(session: dict) -> str:
    """Go back to the previous email."""
    emails = _get_cached_emails(session)
    if not emails:
        return "No emails loaded yet. Say 'read emails' to load your inbox first."

    idx = session.get("_email_read_idx", 0) - 1
    if idx < 0:
        return "You're already at the first email."

    session["_email_read_idx"]   = idx
    session["_email_read_chunk"] = 0
    session.modified = True
    return _read_email_at(emails[idx], idx, len(emails), chunk=0)


def _handle_read_more(session: dict) -> str:
    """Read the next chunk of the current email body."""
    emails = _get_cached_emails(session)
    idx    = session.get("_email_read_idx", 0)
    chunk  = session.get("_email_read_chunk", 0) + 1

    if not emails or idx >= len(emails):
        return "No email is currently being read. Say 'read emails' to start."

    body  = (emails[idx].get("body") or emails[idx].get("snippet") or "").strip()
    start = chunk * _CHUNK_SIZE
    if start >= len(body):
        nxt = idx + 1
        if nxt < len(emails):
            return f"That's the end of this email. Say 'next' for email {nxt + 1}."
        return "That's the end of this email and your last loaded message."

    session["_email_read_chunk"] = chunk
    session.modified = True
    return _read_email_at(emails[idx], idx, len(emails), chunk=chunk)



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
        "You can say: "
        "Read emails — to hear your inbox summary. "
        "Next — to move to the next email. "
        "Previous — to go back. "
        "Read more — to hear the rest of a long email. "
        "Read email 2 — to jump to a specific email. "
        "Send email — to compose a new email. "
        "Stop — to interrupt me while I am speaking. "
        "Logout — to sign out."
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

    # Early exit if Whisper model is not loaded
    if _whisper_model is None:
        tts_path = speak_to_file("Whisper model failed to load. Please run: pip install openai-whisper")
        audio_url = f"/static/audio/{os.path.basename(tts_path)}" if tts_path else None
        try: os.remove(temp_path)
        except OSError: pass
        return {
            "transcription": "",
            "intent": "error",
            "response_text": "Whisper model not loaded. Run: pip install openai-whisper",
            "audio_url": audio_url,
        }

    # 2 — Transcribe
    transcription = transcribe(temp_path)
    logger.info("Transcription: %r", transcription)

    # 3 — Detect intent (context-aware: checks session for active email compose)
    intent = _detect_intent(transcription, session)

    # 4 — Execute intent
    intent_map = {
        "list_emails":  lambda: _handle_list_emails(session),
        "read_email":   lambda: _handle_read_email(session),
        "next_email":   lambda: _handle_next_email(session),
        "prev_email":   lambda: _handle_prev_email(session),
        "read_more":    lambda: _handle_read_more(session),
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
