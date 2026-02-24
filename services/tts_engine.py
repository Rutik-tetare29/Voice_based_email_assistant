"""
Offline Text-to-Speech using pyttsx3.

pyttsx3 engines are NOT thread-safe; we reinitialise per call to avoid
issues when Flask handles concurrent requests.

pyttsx3 on Windows (SAPI5) silently truncates save_to_file output when the
text is long. Fix: split into sentence-level chunks, generate one WAV per
chunk, then stitch the raw PCM together into a single file.
"""
import os
import re
import struct
import uuid
import logging
import pyttsx3
from config import Config

logger = logging.getLogger(__name__)

# Characters per chunk sent to pyttsx3 — keeps each WAV well within SAPI limits
_TTS_CHUNK_CHARS = 180


def _split_sentences(text: str) -> list[str]:
    """
    Split text at sentence boundaries, keeping each chunk ≤ _TTS_CHUNK_CHARS.
    Falls back to splitting on commas / colons if a sentence is still too long.
    """
    # Split on .  !  ?  followed by space or end-of-string
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    buf = ""
    for sent in raw:
        # If a single sentence is already too long, split further on , ; :
        if len(sent) > _TTS_CHUNK_CHARS:
            sub_parts = re.split(r'(?<=[,;:])\s+', sent)
            for sub in sub_parts:
                if len(buf) + len(sub) + 1 > _TTS_CHUNK_CHARS and buf:
                    chunks.append(buf.strip())
                    buf = sub
                else:
                    buf = (buf + " " + sub).strip() if buf else sub
        else:
            if len(buf) + len(sent) + 1 > _TTS_CHUNK_CHARS and buf:
                chunks.append(buf.strip())
                buf = sent
            else:
                buf = (buf + " " + sent).strip() if buf else sent
    if buf:
        chunks.append(buf.strip())
    return [c for c in chunks if c]


def _make_engine():
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    engine.setProperty("volume", 0.95)
    voices = engine.getProperty("voices")
    for voice in voices:
        if "female" in voice.name.lower() or "zira" in voice.name.lower():
            engine.setProperty("voice", voice.id)
            break
    return engine


def _read_wav_data(path: str) -> tuple[dict, bytes]:
    """Read a WAV file and return (header_params, raw_pcm_bytes)."""
    with open(path, "rb") as f:
        data = f.read()
    # Find 'data' chunk
    idx = data.find(b'data')
    if idx == -1:
        return {}, b""
    pcm_size = struct.unpack_from("<I", data, idx + 4)[0]
    pcm = data[idx + 8 : idx + 8 + pcm_size]
    # Parse fmt chunk
    fmt_idx = data.find(b'fmt ')
    params = {}
    if fmt_idx != -1:
        params["channels"]    = struct.unpack_from("<H", data, fmt_idx + 10)[0]
        params["sample_rate"] = struct.unpack_from("<I", data, fmt_idx + 12)[0]
        params["bit_depth"]   = struct.unpack_from("<H", data, fmt_idx + 22)[0]
    return params, pcm


def _write_wav(path: str, pcm: bytes, channels: int, sample_rate: int, bit_depth: int):
    """Write a minimal PCM WAV file."""
    byte_rate   = sample_rate * channels * bit_depth // 8
    block_align = channels * bit_depth // 8
    with open(path, "wb") as f:
        f.write(b'RIFF')
        f.write(struct.pack("<I", 36 + len(pcm)))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                            byte_rate, block_align, bit_depth))
        f.write(b'data')
        f.write(struct.pack("<I", len(pcm)))
        f.write(pcm)


def speak_to_file(text: str) -> str:
    """
    Convert text to speech, save to a WAV file, and return the file path.
    Splits long text into chunks to avoid pyttsx3/SAPI5 truncation on Windows.
    Returns an empty string on error.
    """
    if not text:
        return ""

    # ── Safety: strip SSML/XML angle-bracket tags before they reach SAPI5 ────
    # SAPI5 treats < > as SSML markup; a malformed tag silently aborts all audio.
    text = re.sub(r'<[^>]*>', ' ', text)          # remove <tag> blocks
    text = text.replace('<', ' ').replace('>', ' ')  # kill any strays
    text = re.sub(r'\s+', ' ', text).strip()

    chunks   = _split_sentences(text)
    tmp_dir  = Config.UPLOAD_FOLDER
    out_path = os.path.join(tmp_dir, f"tts_{uuid.uuid4().hex}.wav")
    tmp_paths: list[str] = []

    try:
        for chunk in chunks:
            tmp = os.path.join(tmp_dir, f"_tts_chunk_{uuid.uuid4().hex}.wav")
            tmp_paths.append(tmp)
            engine = _make_engine()
            engine.save_to_file(chunk, tmp)
            engine.runAndWait()
            engine.stop()

        # ── Stitch all chunk WAVs into one file ───────────────────────────────
        all_pcm    = b""
        wav_params = {}
        for tp in tmp_paths:
            if not os.path.exists(tp) or os.path.getsize(tp) == 0:
                continue
            params, pcm = _read_wav_data(tp)
            if pcm:
                all_pcm += pcm
                if not wav_params:
                    wav_params = params

        if not all_pcm or not wav_params:
            logger.error("TTS produced no audio data")
            return ""

        _write_wav(
            out_path, all_pcm,
            wav_params.get("channels",    1),
            wav_params.get("sample_rate", 22050),
            wav_params.get("bit_depth",   16),
        )
        logger.info("TTS saved to %s (%d bytes, %d chunks)", out_path, len(all_pcm), len(tmp_paths))
        return out_path

    except Exception as exc:
        logger.error("TTS error: %s", exc)
        return ""

    finally:
        for tp in tmp_paths:
            try:
                os.remove(tp)
            except OSError:
                pass
