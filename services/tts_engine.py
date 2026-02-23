"""
Offline Text-to-Speech using pyttsx3.

pyttsx3 engines are NOT thread-safe; we reinitialise per call to avoid
issues when Flask handles concurrent requests.
"""
import os
import uuid
import logging
import pyttsx3
from config import Config

logger = logging.getLogger(__name__)


def speak_to_file(text: str) -> str:
    """
    Convert text to speech, save to a WAV file, and return the file path.
    Returns an empty string on error.
    """
    if not text:
        return ""

    out_path = os.path.join(Config.UPLOAD_FOLDER, f"tts_{uuid.uuid4().hex}.wav")

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)     # words per minute
        engine.setProperty("volume", 0.95)

        # Prefer a female voice if available
        voices = engine.getProperty("voices")
        for voice in voices:
            if "female" in voice.name.lower() or "zira" in voice.name.lower():
                engine.setProperty("voice", voice.id)
                break

        engine.save_to_file(text, out_path)
        engine.runAndWait()
        engine.stop()

        logger.info("TTS saved to %s", out_path)
        return out_path

    except Exception as exc:
        logger.error("TTS error: %s", exc)
        return ""
