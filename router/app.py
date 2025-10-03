import os
import time
import threading
import requests
import webrtcvad
import alsaaudio

import numpy as np  # only used for simple sanity ops if needed

# ----- Riva client (2.19.0) -----
from riva.client import Auth, ASRService, SpeechSynthesisService
from riva.client.proto.riva_asr_pb2 import RecognitionConfig, AudioEncoding

# =========================
# ENV / CONFIG
# =========================
RIVA_ADDR = os.getenv("RIVA_ADDR", "127.0.0.1:50051")
MT_URL    = os.getenv("MT_URL", "http://127.0.0.1:7010")  # FastAPI MT service base

# Audio device names
A_IN  = os.getenv("A_IN", "plughw:0,0")   # Speaker A mic
A_OUT = os.getenv("A_OUT", "plughw:0,0")  # Speaker A earphone/speaker
B_IN  = os.getenv("B_IN", "plughw:1,0")   # Speaker B mic
B_OUT = os.getenv("B_OUT", "plughw:1,0")  # Speaker B earphone/speaker

# Language directions per side ("en" or "es" are enough; we'll map to en-US / es-US for Riva)
A_SRC = os.getenv("A_SRC", "es")
A_TGT = os.getenv("A_TGT", "en")
B_SRC = os.getenv("B_SRC", "en")
B_TGT = os.getenv("B_TGT", "es")

# Riva TTS voices (use riva_tts_client --list_voices to confirm)
VOICE_EN = os.getenv("VOICE_EN", "English-US.Female-1")
VOICE_ES = os.getenv("VOICE_ES", "Spanish-US.Female-1")

# Audio pipeline
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))  # 16 kHz mono PCM16
FRAME_MS    = int(os.getenv("FRAME_MS", "20"))        # VAD frame size ms (10/20/30 supported)
BURST_MIN   = int(os.getenv("BURST_MIN_MS", "300"))   # minimum voiced ms to emit a burst
BURST_MAX   = int(os.getenv("BURST_MAX_MS", "600"))   # max burst length before forced emit
VAD_LEVEL   = int(os.getenv("VAD_LEVEL", "2"))        # 0..3 (aggressiveness)

CHANNELS     = 1
SAMPLE_BYTES = 2
FRAME_SIZE   = int(SAMPLE_RATE * FRAME_MS / 1000)     # samples per frame

# =========================
# ALSA HELPERS
# =========================
def open_capture(device_name: str):
    # pyalsaaudio API flags are deprecated positional; keep for compatibility
    pcm = alsaaudio.PCM(type=alsaaudio.PCM_CAPTURE, mode=alsaaudio.PCM_NORMAL, device=device_name)
    pcm.setchannels(CHANNELS)
    pcm.setrate(SAMPLE_RATE)
    pcm.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    pcm.setperiodsize(FRAME_SIZE)
    return pcm

def open_playback(device_name: str):
    pcm = alsaaudio.PCM(type=alsaaudio.PCM_PLAYBACK, mode=alsaaudio.PCM_NORMAL, device=device_name)
    pcm.setchannels(CHANNELS)
    pcm.setrate(SAMPLE_RATE)
    pcm.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    pcm.setperiodsize(FRAME_SIZE)
    return pcm

# =========================
# RIVA CLIENTS
# =========================
_riva_auth = Auth(uri=RIVA_ADDR)
_riva_asr  = ASRService(_riva_auth)
_riva_tts  = SpeechSynthesisService(_riva_auth)

def _lang_code_2letter_to_riva(code2: str) -> str:
    """'en' -> 'en-US', 'es' -> 'es-US'. Default to en-US."""
    if not code2:
        return "en-US"
    code2 = code2.lower()
    if code2.startswith("es"):
        return "es-US"
    return "en-US"

def riva_asr_offline_recognize(pcm_bytes: bytes, lang_hint_2letter: str | None) -> str:
    """
    Recognize a single burst of PCM16 (16kHz mono) using Riva offline ASR.
    Returns best transcript or "".
    """
    lang_code = _lang_code_2letter_to_riva(lang_hint_2letter or "en")
    cfg = RecognitionConfig(
        encoding=AudioEncoding.LINEAR_PCM,
        sample_rate_hz=SAMPLE_RATE,
        language_code=lang_code,
        max_alternatives=1,
        enable_automatic_punctuation=True,
        verbatim_transcripts=False,
    )
    try:
        result = _riva_asr.offline_recognize(pcm_bytes, cfg)
    except Exception as e:
        print(f"[Riva ASR] recognize error: {e}")
        return ""

    if not result or not result.results:
        return ""

    for res in result.results:
        if res.alternatives:
            return res.alternatives[0].transcript.strip()

    return ""

def riva_tts_speak(text: str, tgt_lang_2letter: str) -> bytes:
    """
    Synthesize PCM16 mono via Riva TTS at SAMPLE_RATE.
    """
    text = (text or "").strip()
    if not text:
        return b""

    lang_code = _lang_code_2letter_to_riva(tgt_lang_2letter)
    voice = VOICE_EN if lang_code.startswith("en") else VOICE_ES

    try:
        resp = _riva_tts.synthesize(
            text=text,
            voice_name=voice,
            language_code=lang_code,
            sample_rate_hz=SAMPLE_RATE,
            encoding="LINEAR_PCM",
        )
        return resp.audio or b""
    except Exception as e:
        print(f"[Riva TTS] synth error: {e}")
        return b""

# =========================
# MT (local HTTP service)
# =========================
def mt_translate(text: str, src_2: str, tgt_2: str) -> str:
    """
    Calls your local MT FastAPI server: POST {MT_URL}/translate
      body: {"text": "...", "source": "en", "target": "es"}
      returns: {"text": "..."} or {"translation":"..."} depending on your implementation
    """
    text = (text or "").strip()
    if not text or src_2 == tgt_2:
        return text

    try:
        r = requests.post(
            f"{MT_URL}/translate",
            json={"text": text, "source": src_2, "target": tgt_2},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        # accept either {"text": "..."} or {"translation": "..."}
        return (data.get("text") or data.get("translation") or "").strip()
    except Exception as e:
        print(f"[MT] translate error: {e}")
        return text  # graceful fallback

# =========================
# VAD BURSTING
# =========================
def capture_burst(device_name: str, vad: webrtcvad.Vad) -> bytes:
    """
    Capture 20ms frames and emit a burst of ~BURST_MIN..BURST_MAX ms when speech ends.
    """
    cap = open_capture(device_name)
    buf = b""
    speaking = False
    max_bytes = int(SAMPLE_RATE * BURST_MAX / 1000) * SAMPLE_BYTES
    min_bytes = int(SAMPLE_RATE * BURST_MIN / 1000) * SAMPLE_BYTES

    while True:
        length, data = cap.read()
        if length <= 0:
            continue

        is_speech = False
        try:
            is_speech = vad.is_speech(data, SAMPLE_RATE)
        except Exception:
            # if frame size drifts, drop it
            is_speech = False

        if is_speech:
            speaking = True
            buf += data
            if len(buf) >= max_bytes:
                burst = buf
                buf = b""
                cap.close()
                return burst
        else:
            if speaking:
                # speech just ended
                if len(buf) >= min_bytes:
                    burst = buf
                    buf = b""
                    cap.close()
                    return burst
                # too short -> reset
                buf = b""
                speaking = False

# =========================
# PLAYBACK
# =========================
def play_pcm(device_name: str, pcm: bytes):
    if not pcm:
        return
    out = open_playback(device_name)
    chunk = FRAME_SIZE * SAMPLE_BYTES
    for i in range(0, len(pcm), chunk):
        out.write(pcm[i:i + chunk])
    out.close()

# =========================
# MAIN DIRECTION LOOP
# =========================
def direction_loop(name: str, in_dev: str, out_dev: str, src_lang_2: str, tgt_lang_2: str):
    """
    One direction:
      mic(in_dev) -> VAD burst -> Riva ASR (src) -> MT (src->tgt) -> Riva TTS (tgt) -> speaker(out_dev)
    src_lang_2/tgt_lang_2: 'en' or 'es'
    """
    print(f"[{name}] {in_dev} ({src_lang_2})->({tgt_lang_2}) {out_dev}")
    vad = webrtcvad.Vad(VAD_LEVEL)

    while True:
        burst = capture_burst(in_dev, vad)
        if not burst:
            continue

        # 1) ASR (Riva offline) — we pass a language hint per side
        text_src = riva_asr_offline_recognize(burst, src_lang_2)
        if not text_src:
            continue

        # 2) MT if needed
        out_text = text_src if src_lang_2 == tgt_lang_2 else mt_translate(text_src, src_lang_2, tgt_lang_2)

        # 3) TTS in target language
        pcm_tts = riva_tts_speak(out_text, tgt_lang_2)
        play_pcm(out_dev, pcm_tts)

# =========================
# ENTRY
# =========================
def main():
    print("[router] starting…")
    print(f" A: {A_IN}  {A_SRC}->{A_TGT}  -> {B_OUT} (voice {'VOICE_EN' if A_TGT.startswith('en') else 'VOICE_ES'})")
    print(f" B: {B_IN}  {B_SRC}->{B_TGT}  -> {A_OUT} (voice {'VOICE_ES' if B_TGT.startswith('es') else 'VOICE_EN'})")

    tA = threading.Thread(target=direction_loop, args=("A", A_IN, B_OUT, A_SRC, A_TGT), daemon=True)
    tB = threading.Thread(target=direction_loop, args=("B", B_IN, A_OUT, B_SRC, B_TGT), daemon=True)
    tA.start()
    tB.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting.")

if __name__ == "__main__":
    main()
