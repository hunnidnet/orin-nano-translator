import os
import time
import threading
import requests
import numpy as np
import webrtcvad
import alsaaudio

# ----- Riva client (2.19.0) -----
from riva.client import Auth, SpeechSynthesisService, NLPService

# ------------------ ENV / CONFIG ------------------
ASR_URL   = os.getenv("ASR_URL", "http://127.0.0.1:7010")   # your faster-whisper microservice
RIVA_ADDR = os.getenv("RIVA_ADDR", "127.0.0.1:50051")

A_IN  = os.getenv("A_IN", "plughw:0,0")   # speaker A's mic
A_OUT = os.getenv("A_OUT", "plughw:0,0")   # speaker A's ear
B_IN  = os.getenv("B_IN", "plughw:1,0")   # speaker B's mic
B_OUT = os.getenv("B_OUT", "plughw:1,0")   # speaker B's ear

# language directions you want for each side
A_SRC = os.getenv("A_SRC", "es")
A_TGT = os.getenv("A_TGT", "en")
B_SRC = os.getenv("B_SRC", "en")
B_TGT = os.getenv("B_TGT", "es")

# Riva voices (use `talk.py --list-voices` to confirm names)
VOICE_EN = os.getenv("VOICE_EN", "English-US.Male-Neutral")
VOICE_ES = os.getenv("VOICE_ES", "Spanish-US.Female-1")

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
FRAME_MS    = int(os.getenv("FRAME_MS", "20"))
BURST_MIN   = int(os.getenv("BURST_MIN_MS", "300"))
BURST_MAX   = int(os.getenv("BURST_MAX_MS", "600"))
VAD_LEVEL   = int(os.getenv("VAD_LEVEL", "2"))  # 0..3

CHANNELS     = 1
SAMPLE_BYTES = 2
FRAME_SIZE   = int(SAMPLE_RATE * FRAME_MS / 1000)

# ------------------ ALSA HELPERS ------------------
def open_capture(device_name):
    pcm = alsaaudio.PCM(type=alsaaudio.PCM_CAPTURE, mode=alsaaudio.PCM_NORMAL, device=device_name)
    pcm.setchannels(CHANNELS)
    pcm.setrate(SAMPLE_RATE)
    pcm.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    pcm.setperiodsize(FRAME_SIZE)
    return pcm

def open_playback(device_name):
    pcm = alsaaudio.PCM(type=alsaaudio.PCM_PLAYBACK, mode=alsaaudio.PCM_NORMAL, device=device_name)
    pcm.setchannels(CHANNELS)
    pcm.setrate(SAMPLE_RATE)
    pcm.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    pcm.setperiodsize(FRAME_SIZE)
    return pcm

# ------------------ RIVA CLIENTS ------------------
_riva_auth = Auth(uri=RIVA_ADDR)
_riva_tts  = SpeechSynthesisService(_riva_auth)
_riva_nlp  = NLPService(_riva_auth)  # has translate_text()

def riva_tts_speak(text: str, lang_code: str) -> bytes:
    """
    lang_code: 'en-US' or 'es-US'
    picks VOICE_EN or VOICE_ES accordingly
    """
    if not text.strip():
        return b""
    voice = VOICE_EN if lang_code.startswith("en") else VOICE_ES
    resp = _riva_tts.synthesize(
        text=text,
        voice_name=voice,
        language_code=lang_code,
        sample_rate_hz=SAMPLE_RATE,
        encoding="LINEAR_PCM",
    )
    return resp.audio  # PCM16 mono

def riva_translate(text: str, src_code: str, tgt_code: str) -> str:
    """src_code/tgt_code like 'en-US', 'es-US'"""
    if not text.strip() or src_code == tgt_code:
        return text
    return _riva_nlp.translate_text(
        text,
        source_language_code=src_code,
        target_language_code=tgt_code
    )

# ------------------ ASR CALL ------------------
def asr_chunk(pcm_bytes: bytes, lang_hint: str | None) -> dict:
    """
    POST to faster-whisper microservice.
    lang_hint: 'en' or 'es' or None (auto)
    Returns: {'text': str, 'lang': 'en'|'es'|...}
    """
    try:
        r = requests.post(
            f"{ASR_URL}/asr/chunk",
            json={"pcm16_hex": pcm_bytes.hex(), "sr": SAMPLE_RATE, "lang": lang_hint},
            timeout=5,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[ASR] error: {e}")
        return {"text": "", "lang": lang_hint or ""}

# ------------------ VAD BURSTING ------------------
def capture_burst(device_name, vad: webrtcvad.Vad) -> bytes:
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
        is_speech = vad.is_speech(data, SAMPLE_RATE)
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

# ------------------ PLAYBACK ------------------
def play_pcm(device_name, pcm: bytes):
    if not pcm:
        return
    out = open_playback(device_name)
    # write in chunks to avoid huge single write
    for i in range(0, len(pcm), FRAME_SIZE * SAMPLE_BYTES):
        out.write(pcm[i:i + FRAME_SIZE * SAMPLE_BYTES])
    out.close()

# ------------------ MAIN DIRECTION LOOP ------------------
def direction_loop(name: str, in_dev: str, out_dev: str, src_lang: str, tgt_lang: str):
    """
    One direction:
      mic(in_dev) -> VAD burst -> ASR (src_lang) -> NMT(src->tgt) -> TTS(tgt) -> speaker(out_dev)
    src_lang/tgt_lang: 'en' or 'es'
    """
    print(f"[{name}] {in_dev} ({src_lang})->({tgt_lang}) {out_dev}")
    vad = webrtcvad.Vad(VAD_LEVEL)

    src_code = "es-US" if src_lang.startswith("es") else "en-US"
    tgt_code = "es-US" if tgt_lang.startswith("es") else "en-US"

    while True:
        burst = capture_burst(in_dev, vad)
        if not burst:
            continue

        # 1) ASR (hint the expected language for this side to reduce latency)
        asr = asr_chunk(burst, src_lang)
        text_src = asr.get("text", "").strip()
        if not text_src:
            continue

        # 2) NMT if needed (src!=tgt)
        out_text = text_src if src_code == tgt_code else riva_translate(text_src, src_code, tgt_code)

        # 3) TTS in target language
        pcm_tts = riva_tts_speak(out_text, tgt_code)
        play_pcm(out_dev, pcm_tts)

# ------------------ ENTRY ------------------
def main():
    print("[router] starting…")
    print(f" A: {A_IN}  {A_SRC}->{A_TGT}  -> {B_OUT} (voice {VOICE_EN if A_TGT=='en' else VOICE_ES})")
    print(f" B: {B_IN}  {B_SRC}->{B_TGT}  -> {A_OUT} (voice {VOICE_ES if B_TGT=='es' else VOICE_EN})")

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
