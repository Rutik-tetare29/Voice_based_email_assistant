import os
import uuid
from flask import Flask, request, jsonify, session, send_from_directory, render_template
from flask_login import LoginManager, login_required, current_user, logout_user
from config import Config
from auth.google_auth import google_auth_bp, GoogleUser
from auth.app_password_auth import apppass_auth_bp, AppPasswordUser
from services.voice_processor import process_voice_command, process_text_compose_input
from services.email_service import fetch_emails, send_email

# ── App factory ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.from_object(Config)

# Allow OAuth over plain HTTP during local development
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = app.config["OAUTHLIB_INSECURE_TRANSPORT"]

# ── Flask-Login setup ─────────────────────────────────────────────────────────
login_manager = LoginManager(app)
login_manager.login_view = "index"


@login_manager.user_loader
def load_user(user_id: str):
    """Rebuild user object from session data stored at login."""
    if "user" not in session:
        return None
    user_data = session["user"]
    if user_data.get("auth_type") == "app_password":
        return AppPasswordUser.from_session(user_data)
    return GoogleUser.from_session(user_data)


# ── Blueprints ────────────────────────────────────────────────────────────────
app.register_blueprint(google_auth_bp)
app.register_blueprint(apppass_auth_bp)


# ── Page routes ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login")
def login_page():
    return render_template("login.html")


# ── Voice login transcription (no @login_required) ────────────────────────────

def _normalize_app_password(raw: str) -> str:
    """
    Convert Whisper's transcription of a letter-by-letter App Password
    back to the actual 16-character string.

    Handles all common Whisper outputs for Indian-English speakers:
      • Single letters: 'a', 'b', 'c' …
      • Phonetic names: 'bee', 'see', 'dee', 'ef', 'gee', 'aitch', 'jay', 'kay',
                        'el', 'em', 'en', 'oh', 'pee', 'cue', 'are', 'ess',
                        'tee', 'you', 'vee', 'ex', 'why', 'zee', 'zed'
      • Aye/ay vs eye: 'ay'/'aye' → 'a',  'eye' → 'i'
      • NATO alphabet: 'alpha', 'bravo', 'charlie', ... 'zulu'
      • Stray punctuation Whisper inserts: 'a.', 'B,' etc.
      • Digit words: 'zero'…'nine'  (App Passwords sometimes include digits)
      • Multi-word: 'double you' → 'w', 'x ray' → 'x'
    """
    import re
    text = raw.strip().lower()
    # Strip punctuation Whisper inserts after single letters ('A.' 'B,' 'C;')
    text = re.sub(r'[.,;:!?\'"()\[\]{}]', ' ', text)
    # Handle two-word phonetics before splitting
    text = re.sub(r'\bdouble\s+(?:you|u)\b', 'w', text)
    text = re.sub(r'\bx[\s\-]ray\b', 'x', text)
    text = re.sub(r'\s+', ' ', text).strip()

    LETTER_NAMES = {
        # ── keep bare single letters as-is ───────────────────────────────────
        **{c: c for c in 'abcdefghijklmnopqrstuvwxyz0123456789'},
        # ── A ────────────────────────────────────────────────────────────────
        'ay': 'a', 'aye': 'a', 'alpha': 'a',
        # ── B ────────────────────────────────────────────────────────────────
        'bee': 'b', 'be': 'b', 'bravo': 'b',
        # ── C ────────────────────────────────────────────────────────────────
        'see': 'c', 'sea': 'c', 'si': 'c', 'charlie': 'c',
        # ── D ────────────────────────────────────────────────────────────────
        'dee': 'd', 'de': 'd', 'delta': 'd',
        # ── E ────────────────────────────────────────────────────────────────
        'ee': 'e', 'echo': 'e',
        # ── F ────────────────────────────────────────────────────────────────
        'ef': 'f', 'eff': 'f', 'foxtrot': 'f',
        # ── G ────────────────────────────────────────────────────────────────
        'gee': 'g', 'ji': 'g', 'golf': 'g',
        # ── H ────────────────────────────────────────────────────────────────
        'aitch': 'h', 'haitch': 'h', 'hotel': 'h',
        # ── I ────────────────────────────────────────────────────────────────
        'eye': 'i', 'india': 'i',
        # ── J ────────────────────────────────────────────────────────────────
        'jay': 'j', 'juliett': 'j', 'juliet': 'j',
        # ── K ────────────────────────────────────────────────────────────────
        'kay': 'k', 'kilo': 'k',
        # ── L ────────────────────────────────────────────────────────────────
        'el': 'l', 'ell': 'l', 'lima': 'l',
        # ── M ────────────────────────────────────────────────────────────────
        'em': 'm', 'mike': 'm',
        # ── N ────────────────────────────────────────────────────────────────
        'en': 'n', 'november': 'n',
        # ── O ────────────────────────────────────────────────────────────────
        'oh': 'o', 'owe': 'o', 'oscar': 'o',
        # ── P ────────────────────────────────────────────────────────────────
        'pee': 'p', 'pe': 'p', 'papa': 'p',
        # ── Q ────────────────────────────────────────────────────────────────
        'cue': 'q', 'queue': 'q', 'quebec': 'q',
        # ── R ────────────────────────────────────────────────────────────────
        'are': 'r', 'ar': 'r', 'romeo': 'r',
        # ── S ────────────────────────────────────────────────────────────────
        'ess': 's', 'es': 's', 'sierra': 's',
        # ── T ────────────────────────────────────────────────────────────────
        'tee': 't', 'ti': 't', 'tango': 't',
        # ── U ────────────────────────────────────────────────────────────────
        'you': 'u', 'yoo': 'u', 'uniform': 'u',
        # ── V ────────────────────────────────────────────────────────────────
        'vee': 'v', 've': 'v', 'victor': 'v',
        # ── W ────────────────────────────────────────────────────────────────
        'whiskey': 'w',
        # ── X ────────────────────────────────────────────────────────────────
        'ex': 'x', 'eks': 'x',
        # ── Y ────────────────────────────────────────────────────────────────
        'why': 'y', 'yankee': 'y',
        # ── Z ────────────────────────────────────────────────────────────────
        'zee': 'z', 'zed': 'z', 'zulu': 'z',
        # ── digit words ──────────────────────────────────────────────────────
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    }

    tokens = text.split()
    out = []
    for tok in tokens:
        # Strip any residual punctuation glued to the token
        tok_clean = re.sub(r'[^a-z0-9]', '', tok)
        if not tok_clean:
            continue
        if tok_clean in LETTER_NAMES:
            out.append(LETTER_NAMES[tok_clean])
        else:
            # Unknown word (Whisper sometimes groups letters into a run like "abc")
            # Keep it as-is; it may already be the correct characters
            out.append(tok_clean)
    return ''.join(out)


@app.route("/voice/login-transcribe", methods=["POST"])
def voice_login_transcribe():
    """
    Transcribes a WAV blob for the login page (no auth required).
    Form fields: audio=<wav>, step=email|password
    Returns: { "text": "<raw>", "normalized": "<cleaned>" }
    """
    from services.stt_whisper import transcribe
    from services.voice_processor import _normalize_email_address

    step = request.form.get("step", "email")
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    tmp_path = os.path.join(Config.UPLOAD_FOLDER, f"login_{uuid.uuid4().hex}.wav")
    audio_file.save(tmp_path)
    try:
        raw_text = transcribe(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if step == "email":
        normalized = _normalize_email_address(raw_text)
    elif step in ("email-correct", "yesno"):
        # Return raw lowercased text for client-side matching
        normalized = raw_text.lower().strip()
    else:
        # App Passwords: map phonetic letter-names → actual characters
        normalized = _normalize_app_password(raw_text)

    return jsonify({"text": raw_text, "normalized": normalized})


# ── Voice email correction ─────────────────────────────────────────────────────

# Full phonetic/spoken-letter → actual character map (shared by correction + password)
_PHONETIC_CHARS = {
    # single bare letters kept as-is
    **{c: c for c in 'abcdefghijklmnopqrstuvwxyz0123456789'},
    # ── A ─── 
    'ay': 'a', 'aye': 'a', 'alpha': 'a',
    # ── B ───
    'bee': 'b', 'be': 'b', 'bravo': 'b',
    # ── C ───
    'see': 'c', 'sea': 'c', 'si': 'c', 'charlie': 'c',
    # ── D ───
    'dee': 'd', 'de': 'd', 'delta': 'd',
    # ── E ───
    'ee': 'e', 'echo': 'e',
    # ── F ───
    'ef': 'f', 'eff': 'f', 'foxtrot': 'f',
    # ── G ───
    'gee': 'g', 'ji': 'g', 'golf': 'g',
    # ── H ───
    'aitch': 'h', 'haitch': 'h', 'hotel': 'h',
    # ── I ───
    'eye': 'i', 'india': 'i',
    # ── J ───
    'jay': 'j', 'juliett': 'j', 'juliet': 'j',
    # ── K ───
    'kay': 'k', 'kilo': 'k',
    # ── L ───
    'el': 'l', 'ell': 'l', 'lima': 'l',
    # ── M ───
    'em': 'm', 'mike': 'm',
    # ── N ───
    'en': 'n', 'november': 'n',
    # ── O ───
    'oh': 'o', 'owe': 'o', 'oscar': 'o',
    # ── P ───
    'pee': 'p', 'pe': 'p', 'papa': 'p',
    # ── Q ───
    'cue': 'q', 'queue': 'q', 'quebec': 'q',
    # ── R ───
    'are': 'r', 'ar': 'r', 'romeo': 'r',
    # ── S ───
    'ess': 's', 'es': 's', 'sierra': 's',
    # ── T ───
    'tee': 't', 'ti': 't', 'tango': 't',
    # ── U ───
    'you': 'u', 'yoo': 'u', 'uniform': 'u',
    # ── V ───
    'vee': 'v', 've': 'v', 'victor': 'v',
    # ── W ───
    'double you': 'w', 'double u': 'w', 'whiskey': 'w',
    # ── X ───
    'ex': 'x', 'eks': 'x', 'x ray': 'x',
    # ── Y ───
    'why': 'y', 'yankee': 'y',
    # ── Z ───
    'zee': 'z', 'zed': 'z', 'zulu': 'z',
    # ── special chars ───
    'at sign': '@', 'at the rate': '@', 'at': '@',
    'dot': '.', 'period': '.', 'full stop': '.', 'point': '.',
    'dash': '-', 'hyphen': '-', 'minus': '-',
    'underscore': '_', 'under score': '_',
    'plus': '+',
    # ── digit words ───
    'zero': '0', 'one': '1', 'two': '2', 'to': '2', 'too': '2',
    'three': '3', 'four': '4', 'for': '4', 'five': '5',
    'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
}


def _vc_clean(text: str) -> str:
    """
    Convert a spoken phrase into the actual email characters it represents.
    Handles: phonetic letter names, NATO alphabet, digit words, special char names,
    multi-token phrases like 'double you', bare letters/digits, and mixed strings.
    """
    import re
    text = text.strip().lower()
    # Strip leading spoken fillers
    text = re.sub(
        r'^(?:the letter|letter|the number|number|the character|character|the digit|digit|the)\s+',
        '', text
    )
    # Normalise punctuation Whisper inserts ('A.' 'B,')
    text = re.sub(r'[.,;:!?\'"()\[\]{}]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Try full phrase first (e.g. "double you", "at sign", "x ray")
    if text in _PHONETIC_CHARS:
        return _PHONETIC_CHARS[text]

    # Tokenise and map each token
    tokens = text.split()
    out = []
    i = 0
    while i < len(tokens):
        # Try two-token phrases first
        if i + 1 < len(tokens):
            two = tokens[i] + ' ' + tokens[i + 1]
            if two in _PHONETIC_CHARS:
                out.append(_PHONETIC_CHARS[two])
                i += 2
                continue
        tok = re.sub(r'[^a-z0-9]', '', tokens[i])
        if tok in _PHONETIC_CHARS:
            out.append(_PHONETIC_CHARS[tok])
        elif tok:
            out.append(tok)   # unknown → keep as-is (may already be correct chars)
        i += 1
    return ''.join(out)


@app.route("/voice/correct-email", methods=["POST"])
def voice_correct_email():
    """
    Apply a voice correction command to an email address.
    JSON body: { "email": "<current>", "command": "<voice command text>" }
    Returns:   { "corrected": "<new email>", "message": "<readable result>", "changed": bool }

    Supported command patterns (case-insensitive, phonetic-letter-aware):
      replace/change/fix/make/turn/set/correct X to/with/as/into Y
      add/insert/put/place/type X before Y
      add/insert/put/place/type X after Y
      add/insert/put/place/append/prepend X at the end / beginning / start
      add X at position N
      remove/delete/take out/drop/erase/eliminate/strip X
      remove/delete the last letter / first letter
      move/shift X to the end / beginning
      the email/address is X             → replace entire local part
      the domain is/should be X          → replace domain
      redo / retype / whole email is X   → rewrite entire address
    """
    import re
    data    = request.json or {}
    email   = data.get("email", "").strip().lower()
    command = data.get("command", "").strip().lower()
    if not email or not command:
        return jsonify({"error": "Missing email or command"}), 400

    # Ensure we always have a local + domain even for malformed input
    if "@" in email:
        local, domain = email.split("@", 1)
    else:
        local, domain = email, "gmail.com"

    def _ok(new_local, new_domain, message):
        c = new_local + "@" + new_domain
        return jsonify({"corrected": c, "message": message, "changed": True})

    def _no_change(reason):
        return jsonify({"corrected": email, "message": reason, "changed": False})

    def _find_and_replace(source, old, new):
        """Replace first occurrence of old in source; return (new_source, found)."""
        if old in source:
            return source.replace(old, new, 1), True
        return source, False

    # ── Whole-email rewrite: "the email is X" / "redo as X" / "whole email is X" ─
    m = re.search(
        r'(?:whole email is|redo(?:\s+as)?|retype(?:\s+as)?|entire(?:\s+address)? is|'
        r'email(?:\s+address)? (?:is|should be)|my email is)\s+(.+)', command)
    if m:
        raw = m.group(1).strip()
        # If they say the full address (contains "at" or "@"), normalise it
        new_email = _vc_clean(raw) if '@' not in raw else raw.replace(' ', '').lower()
        if '@' not in new_email:
            # They probably only said the local part
            new_local = new_email; new_domain = domain
        else:
            new_local, new_domain = new_email.split('@', 1)
        return _ok(new_local, new_domain, "Email rewritten to " + new_local + "@" + new_domain)

    # ── Domain replacement: "the domain is/should be X" / "change domain to X" ──
    m = re.search(
        r'(?:domain (?:is|should be|to|as)|change domain to|set domain to|'
        r'domain name is)\s+(.+)', command)
    if m:
        raw_domain = m.group(1).strip()
        # Map spoken words: "gmail" → "gmail.com", "yahoo" → "yahoo.com", etc.
        domain_map = {
            'gmail': 'gmail.com', 'google mail': 'gmail.com',
            'yahoo': 'yahoo.com', 'hotmail': 'hotmail.com',
            'outlook': 'outlook.com', 'icloud': 'icloud.com',
            'proton': 'protonmail.com', 'protonmail': 'protonmail.com',
        }
        new_domain = domain_map.get(raw_domain, _vc_clean(raw_domain))
        return _ok(local, new_domain, "Domain changed to " + new_domain)

    # ── replace / change / fix / make / turn / set / correct X to/with/as Y ──────
    m = re.search(
        r'(?:replace|change|fix|make|turn|set|correct|swap|edit)\s+(.+?)'
        r'\s+(?:with|to|by|as|into|for)\s+(.+)', command)
    if m:
        old = _vc_clean(m.group(1).strip())
        new = _vc_clean(m.group(2).strip())
        new_local, found = _find_and_replace(local, old, new)
        if found:
            return _ok(new_local, domain, "Replaced '" + old + "' with '" + new + "'")
        new_domain, found = _find_and_replace(domain, old, new)
        if found:
            return _ok(local, new_domain, "Replaced '" + old + "' with '" + new + "' in domain")
        return _no_change("Could not find '" + old + "' in the email address")

    # ── "fix X" alone (shorthand for "fix X" with no replacement — same as remove) ─
    # Only treat as standalone fix if no "to/with" clause was found above
    m_fix = re.search(r'^(?:fix|correct)\s+(.+)$', command)

    # ── add / insert X before Y ──────────────────────────────────────────────
    m = re.search(r'(?:add|insert|put|place|type)\s+(.+?)\s+before\s+(.+)', command)
    if m:
        char = _vc_clean(m.group(1).strip())
        ref  = _vc_clean(m.group(2).strip())
        if ref in local:
            pos   = local.find(ref)
            return _ok(local[:pos] + char + local[pos:], domain,
                       "Added '" + char + "' before '" + ref + "'")
        if ref in domain:
            pos   = domain.find(ref)
            return _ok(local, domain[:pos] + char + domain[pos:],
                       "Added '" + char + "' before '" + ref + "' in domain")
        return _no_change("Could not find '" + ref + "' in the email address")

    # ── add / insert X after Y ───────────────────────────────────────────────
    m = re.search(r'(?:add|insert|put|place|type)\s+(.+?)\s+after\s+(.+)', command)
    if m:
        char = _vc_clean(m.group(1).strip())
        ref  = _vc_clean(m.group(2).strip())
        if ref in local:
            pos   = local.find(ref) + len(ref)
            return _ok(local[:pos] + char + local[pos:], domain,
                       "Added '" + char + "' after '" + ref + "'")
        if ref in domain:
            pos   = domain.find(ref) + len(ref)
            return _ok(local, domain[:pos] + char + domain[pos:],
                       "Added '" + char + "' after '" + ref + "' in domain")
        return _no_change("Could not find '" + ref + "' in the email address")

    # ── add X at position N ──────────────────────────────────────────────────
    m = re.search(r'(?:add|insert|put)\s+(.+?)\s+at\s+position\s+(\d+)', command)
    if m:
        char = _vc_clean(m.group(1).strip())
        pos  = int(m.group(2)) - 1   # 1-based → 0-based
        pos  = max(0, min(pos, len(local)))
        new_local = local[:pos] + char + local[pos:]
        return _ok(new_local, domain, "Inserted '" + char + "' at position " + str(pos + 1))

    # ── add / append X at end ────────────────────────────────────────────────
    m = re.search(
        r'(?:add|append|put|insert|place|type)\s+(.+?)\s+'
        r'(?:at the end|at end|to the end|to end|at last|in the end)', command)
    if m:
        char = _vc_clean(m.group(1).strip())
        return _ok(local + char, domain, "Added '" + char + "' at end")

    # ── add / prepend X at start ─────────────────────────────────────────────
    m = re.search(
        r'(?:add|prepend|put|insert|place|type)\s+(.+?)\s+'
        r'(?:at the start|at start|to the start|at beginning|at the beginning|'
        r'in the beginning|at front|at the front|to the front)', command)
    if m:
        char = _vc_clean(m.group(1).strip())
        return _ok(char + local, domain, "Added '" + char + "' at start")

    # ── remove / delete last letter ──────────────────────────────────────────
    if re.search(r'(?:remove|delete|take out|erase|drop)\s+(?:the\s+)?last\s+(?:letter|char|character)?', command):
        if local:
            return _ok(local[:-1], domain, "Removed last character '" + local[-1] + "'")
        return _no_change("Local part is already empty")

    # ── remove / delete first letter ─────────────────────────────────────────
    if re.search(r'(?:remove|delete|take out|erase|drop)\s+(?:the\s+)?first\s+(?:letter|char|character)?', command):
        if local:
            return _ok(local[1:], domain, "Removed first character '" + local[0] + "'")
        return _no_change("Local part is already empty")

    # ── remove / delete X (general) ──────────────────────────────────────────
    m = re.search(r'(?:remove|delete|take out|drop|erase|eliminate|strip)\s+(.+)', command)
    if m:
        char = _vc_clean(m.group(1).strip())
        new_local, found = _find_and_replace(local, char, '')
        if found:
            return _ok(new_local, domain, "Removed '" + char + "'")
        new_domain, found = _find_and_replace(domain, char, '')
        if found:
            return _ok(local, new_domain, "Removed '" + char + "' from domain")
        return _no_change("Could not find '" + char + "' in the email address")

    # ── move X to end ────────────────────────────────────────────────────────
    m = re.search(r'move\s+(.+?)\s+to\s+(?:the\s+)?end', command)
    if m:
        char = _vc_clean(m.group(1).strip())
        if char in local:
            new_local = local.replace(char, '', 1) + char
            return _ok(new_local, domain, "Moved '" + char + "' to end")
        return _no_change("Could not find '" + char + "'")

    # ── move X to start ───────────────────────────────────────────────────────
    m = re.search(r'move\s+(.+?)\s+to\s+(?:the\s+)?(?:start|beginning|front)', command)
    if m:
        char = _vc_clean(m.group(1).strip())
        if char in local:
            new_local = char + local.replace(char, '', 1)
            return _ok(new_local, domain, "Moved '" + char + "' to start")
        return _no_change("Could not find '" + char + "'")

    # ── standalone "fix X" (no replacement given) → same as remove ───────────
    if m_fix:
        char = _vc_clean(m_fix.group(1).strip())
        new_local, found = _find_and_replace(local, char, '')
        if found:
            return _ok(new_local, domain, "Removed '" + char + "'")
        return _no_change("Could not find '" + char + "' to fix")

    return _no_change(
        "Could not understand the correction. "
        "Try: replace X with Y, add X before Y, remove X, or fix X to Y"
    )


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user=current_user)


# ── Voice API ─────────────────────────────────────────────────────────────────
@app.route("/voice/process", methods=["POST"])
@login_required
def voice_process():
    """
    Receives a WAV blob from the browser, runs STT → intent → TTS,
    and returns a JSON payload with transcription + audio URL.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    result = process_voice_command(audio_file, session)
    return jsonify(result)


@app.route("/voice/compose-text", methods=["POST"])
@login_required
def voice_compose_text():
    """
    Accepts a typed field value for the active voice compose step.
    Body: { "field": "to" | "subject" | "body", "value": "..." }
    Returns the same JSON shape as /voice/process.
    """
    data  = request.get_json(force=True) or {}
    field = data.get("field", "").strip()
    value = data.get("value", "").strip()
    if not field or not value:
        return jsonify({"error": "Missing field or value"}), 400
    result = process_text_compose_input(field, value, session)
    return jsonify(result)


# ── Email API ─────────────────────────────────────────────────────────────────
@app.route("/emails", methods=["GET"])
@login_required
def get_emails():
    emails = fetch_emails(session)
    return jsonify({"emails": emails})


@app.route("/send-email", methods=["POST"])
@login_required
def send_email_route():
    data = request.get_json()
    to_addr = data.get("to", "").strip()
    subject = data.get("subject", "").strip()
    body = data.get("body", "").strip()

    if not to_addr or not subject or not body:
        return jsonify({"error": "Missing required fields"}), 400

    success, message = send_email(session, to_addr, subject, body)
    status = 200 if success else 500
    return jsonify({"success": success, "message": message}), status


# ── Serve TTS audio ───────────────────────────────────────────────────────────
@app.route("/static/audio/<path:filename>")
def serve_audio(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ── Logout ────────────────────────────────────────────────────────────────────
@app.route("/logout")
@login_required
def logout():
    logout_user()
    session.clear()
    return jsonify({"message": "Logged out"}), 200


if __name__ == "__main__":
    app.run(debug=app.config["DEBUG"], port=5000)
