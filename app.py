import os
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
