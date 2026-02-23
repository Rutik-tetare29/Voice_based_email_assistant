"""
Offline Speech-to-Text using OpenAI Whisper (local model, no API key).

The browser already sends a 16 kHz mono PCM WAV so we feed it straight
to Whisper without any resampling.

Model is downloaded automatically on first load to  ~/.cache/whisper/
(~145 MB for 'base') and reused on every subsequent run.

Model size guide
----------------
  tiny   ~75 MB   – fastest, lower accuracy
  base   ~145 MB  – recommended: good accuracy ~3-6 s on CPU
  small  ~465 MB  – better accuracy, ~8-15 s on CPU
  medium ~1.5 GB  – near-perfect, needs GPU for real-time use

Set WHISPER_MODEL=base in your .env file (default: base).
"""
import logging
import numpy as np
import soundfile as sf
import whisper

from config import Config

logger = logging.getLogger(__name__)

# ── Load model once (module-level singleton) ──────────────────────────────────
_model_name: str = Config.WHISPER_MODEL
_model = None

try:
    logger.info("Loading Whisper '%s' model …", _model_name)
    _model = whisper.load_model(_model_name)
    logger.info("Whisper '%s' model loaded successfully.", _model_name)
except Exception as exc:
    _model = None
    logger.error("Failed to load Whisper model '%s': %s", _model_name, exc)


# ── Prompt — guides Whisper toward email-command vocabulary ───────────────────
_PROMPT = (
    "Voice email assistant. Commands: read email, send email, logout, help, stop, cancel. "
    "Email addresses like user at gmail dot com, subject lines, and message bodies."
)


def transcribe(wav_path: str) -> str:
    """
    Transcribe a 16 kHz mono WAV file and return the recognised text.
    Returns an empty string on failure.
    """
    if _model is None:
        logger.error("Whisper model not loaded — cannot transcribe")
        return ""

    try:
        # Load audio — soundfile handles .wav without requiring ffmpeg
        audio, sr = sf.read(wav_path, dtype="float32")

        # Mix stereo → mono if needed
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Browser sends 16 kHz already.  Guard for edge cases with other rates.
        if sr != 16000:
            target_len = int(len(audio) * 16000 / sr)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, target_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)

        result = _model.transcribe(
            audio,
            language="en",
            fp16=False,                      # False for CPU; True speeds up GPU
            initial_prompt=_PROMPT,
            temperature=0.0,                 # greedy decoding — most deterministic
            condition_on_previous_text=False,
        )

        text = result.get("text", "").strip()
        logger.info("Whisper transcription: %r", text)
        return text

    except Exception as exc:
        logger.error("Whisper transcription error for %s: %s", wav_path, exc)
        return ""

