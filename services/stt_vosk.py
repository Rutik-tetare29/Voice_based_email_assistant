"""
Offline Speech-to-Text using Vosk.

Usage:
    text = transcribe("path/to/audio.wav")

Model is loaded once at import time to avoid repeated disk I/O.
Download a model from https://alphacephei.com/vosk/models (e.g. vosk-model-small-en-us-0.15)
and set VOSK_MODEL_PATH in your .env file.
"""
import os
import wave
import json
import logging
from vosk import Model, KaldiRecognizer
from config import Config

logger = logging.getLogger(__name__)

# ── Load model once (module-level singleton) ──────────────────────────────────
try:
    if not os.path.isdir(Config.VOSK_MODEL_PATH):
        raise FileNotFoundError(
            f"Vosk model folder not found: {Config.VOSK_MODEL_PATH}\n"
            "Download a model from https://alphacephei.com/vosk/models "
            "(e.g. vosk-model-small-en-us-0.15), unzip it, and set "
            "VOSK_MODEL_PATH in your .env file."
        )
    _model = Model(Config.VOSK_MODEL_PATH)
    logger.info("Vosk model loaded from %s", Config.VOSK_MODEL_PATH)
except Exception as exc:
    _model = None
    logger.error("Failed to load Vosk model: %s", exc)


def transcribe(wav_path: str) -> str:
    """
    Transcribe a mono 16 kHz WAV file and return the recognised text.
    Returns an empty string on failure.
    """
    if _model is None:
        logger.error("Vosk model not loaded — cannot transcribe")
        return ""

    try:
        with wave.open(wav_path, "rb") as wf:
            if wf.getnchannels() != 1:
                logger.warning(
                    "Audio has %d channels; Vosk requires mono.", wf.getnchannels()
                )
            sample_rate = wf.getframerate()
            recognizer = KaldiRecognizer(_model, sample_rate)
            recognizer.SetWords(True)

            results = []
            while True:
                data = wf.readframes(4000)
                if not data:
                    break
                if recognizer.AcceptWaveform(data):
                    part = json.loads(recognizer.Result())
                    results.append(part.get("text", ""))

            # Flush final partial result
            final = json.loads(recognizer.FinalResult())
            results.append(final.get("text", ""))

        return " ".join(r for r in results if r).strip()

    except Exception as exc:
        logger.error("Transcription error: %s", exc)
        return ""
