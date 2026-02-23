'use strict';

/**
 * voice.js — Reliable voice capture via AudioContext ScriptProcessor.
 *
 * WHY not MediaRecorder → decodeAudioData:
 *   MediaRecorder produces compressed webm/ogg. decodeAudioData on that
 *   blob fails silently on many Windows / Chrome / Edge setups.
 *
 * SOLUTION:
 *   ScriptProcessor captures raw Float32 PCM directly from the mic at the
 *   browser's native rate → we resample to 16 kHz mono → write PCM-16 WAV
 *   → POST to Flask → Vosk transcribes it reliably.
 */

// ── State ─────────────────────────────────────────────────────────────────────
let audioCtx        = null;   // created ONCE on first user gesture, never closed
let nativeSampleRate = 44100; // set when audioCtx is created
let scriptProcessor = null;
let micSource       = null;
let micStream       = null;
let pcmBuffers      = [];
let isRecording     = false;
let recordTimeout   = null;
let isSpeaking      = false;   // true while TTS audio is playing
let _ttsAudio       = null;    // reference to current <audio> element
let _emailStep      = null;    // current step of voice-guided email compose (or null)
let _wsRecog        = null;    // Web Speech API recognizer used during TTS playback

const TARGET_SAMPLE_RATE = 16000;   // Vosk requirement
const MAX_RECORD_SECONDS = 8;
const BUFFER_SIZE        = 4096;

// ── DOM shorthand ─────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ── Entry point (onclick) ─────────────────────────────────────────────────────
async function toggleRecording() {
  // If AI is speaking, stop it — stopSpeaking() will auto-restart recording
  if (isSpeaking) { stopSpeaking(); return; }
  isRecording ? stopRecording() : await startRecording();
}

// ── Initialise AudioContext (called once from first user gesture) ─────────────
async function _ensureAudioCtx() {
  if (!micStream || micStream.getTracks().some(t => t.readyState === 'ended')) {
    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount:     1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl:  true,
        },
        video: false,
      });
    } catch (err) {
      setStatus('Mic access denied: ' + err.message, 'error');
      return false;
    }
  }

  if (!audioCtx) {
    audioCtx         = new (window.AudioContext || window.webkitAudioContext)();
    nativeSampleRate = audioCtx.sampleRate;
  }

  // Browsers can suspend the ctx; resume only works inside a user-gesture call
  // (this path is always triggered by a click on first call)
  if (audioCtx.state === 'suspended') {
    try { await audioCtx.resume(); } catch (_) {}
  }

  return true;
}

// ── Start ─────────────────────────────────────────────────────────────────────
async function startRecording() {
  if (isRecording) return;                           // already recording

  const ready = await _ensureAudioCtx();
  if (!ready) return;

  // AudioContext might have been left suspended by the browser after TTS.
  // If so we can't resume it without a user gesture — just wait for next tap.
  if (audioCtx.state === 'suspended') {
    setStatus('Tap 🎤 to activate microphone', 'idle');
    return;
  }

  pcmBuffers = [];

  // Fresh ScriptProcessor each recording (they can't be reused across recordings)
  if (scriptProcessor) { scriptProcessor.disconnect(); scriptProcessor = null; }
  if (micSource)       { micSource.disconnect();       micSource = null; }

  micSource       = audioCtx.createMediaStreamSource(micStream);
  scriptProcessor = audioCtx.createScriptProcessor(BUFFER_SIZE, 1, 1);

  scriptProcessor.onaudioprocess = event => {
    if (!isRecording) return;
    const samples = event.inputBuffer.getChannelData(0);
    pcmBuffers.push(new Float32Array(samples));
  };

  micSource.connect(scriptProcessor);
  scriptProcessor.connect(audioCtx.destination);

  isRecording   = true;
  setRecordingUI(true);
  recordTimeout = setTimeout(stopRecording, MAX_RECORD_SECONDS * 1000);
}

// ── Stop ──────────────────────────────────────────────────────────────────────
function stopRecording() {
  if (!isRecording) return;
  clearTimeout(recordTimeout);
  isRecording = false;
  setRecordingUI(false);

  // Disconnect graph but keep audioCtx and micStream alive — the OS mic
  // indicator stays on and auto-restart after TTS works without a user gesture.
  if (scriptProcessor) { scriptProcessor.disconnect(); scriptProcessor = null; }
  if (micSource)       { micSource.disconnect();       micSource = null; }

  processAndSend(pcmBuffers);
}

// ── Merge → resample → WAV → POST ─────────────────────────────────────────────
async function processAndSend(buffers) {
  if (!buffers.length) {
    setStatus('No audio captured — try again', 'error');
    return;
  }
  setStatus('Processing…', 'processing');

  // Merge all chunks
  const totalLen = buffers.reduce((n, b) => n + b.length, 0);
  const merged   = new Float32Array(totalLen);
  let off = 0;
  for (const b of buffers) { merged.set(b, off); off += b.length; }

  // Resample to 16 kHz
  const resampled = resample(merged, nativeSampleRate, TARGET_SAMPLE_RATE);

  // Encode as PCM-16 WAV
  const wavBlob = toWavBlob(resampled, TARGET_SAMPLE_RATE);

  // Send
  const form = new FormData();
  form.append('audio', wavBlob, 'recording.wav');

  try {
    const res  = await fetch('/voice/process', { method: 'POST', body: form });
    const data = await res.json();

    if (!res.ok) {
      setStatus('Server error: ' + (data.error || 'Unknown'), 'error');
      return;
    }

    $('transcriptionText').textContent = data.transcription || '(nothing recognised)';
    $('responseText').textContent      = data.response_text || '';

    // ── Track email compose step ──────────────────────────────────────────────
    _emailStep = data.email_step || null;

    // ── Handle special intents ────────────────────────────────────────────────
    if (data.intent === 'stop_reading') {
      // Instant cut (Web Speech watcher may have already done this, but be safe)
      _stopSpeechWatcher();
      if (_ttsAudio) { _ttsAudio.pause(); _ttsAudio.src = ''; _ttsAudio = null; }
      _setSpeakingUI(false);
      setStatus('✋ Stopped · Tap 🎤 to continue', 'idle');
      return;
    }

    if (data.intent === 'cancel_email') {
      _emailStep = null;
      if (data.audio_url) { playTTS(data.audio_url); return; }  // TTS onended auto-restarts
      setStatus('Email cancelled · Ready to listen', 'idle');
      startRecording();
      return;
    }

    // ── Normal flow ───────────────────────────────────────────────────────────
    // If nothing was heard (silence), just go back to ready state silently
    if (!data.transcription && data.intent === 'unknown') {
      setStatus('Ready · Tap 🎤 to speak', 'idle');
      return;
    }
    // Show compose step as status hint
    const stepLabels = { to: '📧 Step 1/4 · Say recipient address', subject: '📝 Step 2/4 · Say subject', body: '💬 Step 3/4 · Say your message', confirm: '✅ Step 4/4 · Say "confirm" or "cancel"' };
    const statusMsg = _emailStep ? stepLabels[_emailStep] : ('Done • ' + (data.intent || '—'));
    setStatus(statusMsg, _emailStep ? 'recording' : 'done');

    if (data.audio_url) playTTS(data.audio_url);
    if (data.intent === 'logout') {
      _releaseMic();
      setTimeout(() => { window.location.href = '/'; }, 2500);
    }

  } catch (err) {
    setStatus('Network error: ' + err.message, 'error');
    console.error(err);
  }
}

// ── Resample (linear interpolation) ──────────────────────────────────────────
function resample(samples, fromRate, toRate) {
  if (fromRate === toRate) return samples;
  const outLen = Math.round(samples.length * toRate / fromRate);
  const out    = new Float32Array(outLen);
  const ratio  = fromRate / toRate;
  for (let i = 0; i < outLen; i++) {
    const pos  = i * ratio;
    const idx  = Math.floor(pos);
    const frac = pos - idx;
    out[i] = (samples[idx] ?? 0) + frac * ((samples[idx + 1] ?? 0) - (samples[idx] ?? 0));
  }
  return out;
}

// ── PCM-16 WAV encoder ─────────────────────────────────────────────────────────
function toWavBlob(float32, sampleRate) {
  const numCh    = 1;
  const dataSize = float32.length * 2;
  const buf      = new ArrayBuffer(44 + dataSize);
  const view     = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };

  str(0,  'RIFF');  view.setUint32(4,  36 + dataSize,          true);
  str(8,  'WAVE');  str(12, 'fmt ');
  view.setUint32(16, 16,                    true);   // PCM chunk
  view.setUint16(20, 1,                     true);   // PCM format
  view.setUint16(22, numCh,                 true);
  view.setUint32(24, sampleRate,            true);
  view.setUint32(28, sampleRate * numCh * 2, true);  // byte rate
  view.setUint16(32, numCh * 2,             true);   // block align
  view.setUint16(34, 16,                    true);   // bits per sample
  str(36, 'data');  view.setUint32(40, dataSize, true);

  let o = 44;
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    view.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    o += 2;
  }
  return new Blob([buf], { type: 'audio/wav' });
}

// ── Words that should cut TTS the instant they're spoken ──────────────────────
// Mirrors server-side _STOP_EXACT + _CANCEL_EXACT, kept broad for the small model
const _INTERRUPT_WORDS = new Set([
  'stop','top','stock','shop','stuff','stoop','stored','sport','stomp',
  'quiet','quite','silence','silent','pause','paws','halt',
  'enough','shut','cancel','cancelled','council','console','abort','no','nope',
]);

// ── Instant stop-watcher using Web Speech API (no server roundtrip) ───────────
function _startSpeechWatcher() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return;   // browser doesn't support it — user must tap ⏹ button

  _stopSpeechWatcher();   // kill any previous instance

  const recog = new SR();
  recog.lang            = 'en-US';
  recog.continuous      = true;
  recog.interimResults  = true;   // fire on partial results for fastest response
  recog.maxAlternatives = 3;

  recog.onresult = (event) => {
    if (!isSpeaking) { recog.stop(); return; }
    for (let i = event.resultIndex; i < event.results.length; i++) {
      for (let j = 0; j < event.results[i].length; j++) {
        const heard = event.results[i][j].transcript.toLowerCase().trim();
        for (const word of heard.split(/\s+/)) {
          if (_INTERRUPT_WORDS.has(word)) {
            _stopSpeechWatcher();
            // Instant cut — no server trip, no TTS response
            if (_ttsAudio) { _ttsAudio.pause(); _ttsAudio.src = ''; _ttsAudio = null; }
            _setSpeakingUI(false);
            // Go idle — do NOT auto-start recording (would immediately capture
            // silence / the tail of "stop" and trigger "I did not hear anything")
            setStatus('✋ Stopped · Tap 🎤 to continue', 'idle');
            return;
          }
        }
      }
    }
  };

  recog.onerror = (e) => {
    // 'no-speech' is normal — just restart
    if (isSpeaking && e.error === 'no-speech') {
      setTimeout(() => { if (isSpeaking) _startSpeechWatcher(); }, 50);
    }
  };

  recog.onend = () => {
    // Auto-restart while TTS is still playing (browser stops after a few secs)
    if (isSpeaking && _wsRecog === recog) {
      setTimeout(() => { if (isSpeaking) _startSpeechWatcher(); }, 50);
    }
  };

  try { recog.start(); _wsRecog = recog; }
  catch (e) { console.warn('SpeechRecognition start error:', e); }
}

function _stopSpeechWatcher() {
  if (_wsRecog) { try { _wsRecog.stop(); } catch (_) {} _wsRecog = null; }
}

// ── TTS playback ──────────────────────────────────────────────────────────────
function playTTS(url) {
  if (_ttsAudio) { _ttsAudio.pause(); _ttsAudio.src = ''; _ttsAudio = null; }

  const a = $('ttsAudio');
  a.src = url + '?t=' + Date.now();
  _ttsAudio = a;

  _setSpeakingUI(true);

  a.onended = () => {
    _ttsAudio = null;
    _stopSpeechWatcher();
    _setSpeakingUI(false);
    // Only auto-start if TTS completed naturally (not interrupted by stop)
    // A 400ms grace delay avoids immediately capturing end-of-speech noise
    setTimeout(() => { if (!isRecording && !isSpeaking) startRecording(); }, 400);
  };
  a.onerror = () => {
    _ttsAudio = null;
    _stopSpeechWatcher();
    _setSpeakingUI(false);
    setTimeout(() => { if (!isRecording && !isSpeaking) startRecording(); }, 400);
  };

  a.play()
    .then(() => {
      // Start instant stop-watcher the moment audio begins
      _startSpeechWatcher();
    })
    .catch(e => {
      console.warn('TTS play error:', e);
      _setSpeakingUI(false);
    });
}

// Called by ⏹ button or toggleRecording() while speaking
function stopSpeaking() {
  _stopSpeechWatcher();
  if (_ttsAudio) { _ttsAudio.pause(); _ttsAudio.src = ''; _ttsAudio = null; }
  _setSpeakingUI(false);
  // Go idle — user taps mic when ready for next command
  setStatus('✋ Stopped · Tap 🎤 to continue', 'idle');
}

// Call this only on explicit logout to release the OS mic indicator
function _releaseMic() {
  _stopSpeechWatcher();
  if (scriptProcessor) { scriptProcessor.disconnect(); scriptProcessor = null; }
  if (micSource)       { micSource.disconnect();       micSource = null; }
  if (micStream)       { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
  if (audioCtx)        { audioCtx.close();             audioCtx = null; }
}

// ── Speaking UI helper ────────────────────────────────────────────────────────
function _setSpeakingUI(speaking) {
  isSpeaking = speaking;
  const micBtn  = $('micBtn');
  const stopBtn = $('stopBtn');
  const sr1     = $('speakRing1');
  const sr2     = $('speakRing2');

  if (speaking) {
    // Mic stays visible but dimmed — user can click it to interrupt + record
    micBtn.classList.add('opacity-50', 'scale-90');
    stopBtn.classList.remove('hidden');
    sr1 && sr1.classList.remove('hidden');
    sr2 && sr2.classList.remove('hidden');
    setStatus('🔊 Speaking… say "stop" or tap 🎤', 'processing');
  } else {
    micBtn.classList.remove('opacity-50', 'scale-90');
    stopBtn.classList.add('hidden');
    sr1 && sr1.classList.add('hidden');
    sr2 && sr2.classList.add('hidden');
  }
}

// ── UI helpers ────────────────────────────────────────────────────────────────
function setRecordingUI(on) {
  const btn = $('micBtn');
  if (on) {
    btn.classList.add('recording');
    btn.innerHTML = '⏹';
    $('ring1').classList.remove('hidden');
    $('ring2').classList.remove('hidden');
    setStatus(`Recording… (auto-stops in ${MAX_RECORD_SECONDS}s)`, 'recording');
  } else {
    btn.classList.remove('recording');
    btn.innerHTML = '🎤';
    $('ring1').classList.add('hidden');
    $('ring2').classList.add('hidden');
  }
}

function setStatus(msg, type = 'idle') {
  const colors = { idle:'text-gray-400', recording:'text-red-400', processing:'text-yellow-400', done:'text-green-400', error:'text-red-500' };
  const el = $('statusText');
  el.className  = 'text-center text-sm mt-2 ' + (colors[type] || colors.idle);
  el.textContent = msg;
}
// Release mic tracks when page closes / navigates away
window.addEventListener('beforeunload', _releaseMic);