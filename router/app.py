import os
import time
import threading
import requests
import numpy as np
import webrtcvad
import alsaaudio

# Riva client (2.19.0)
from riva.client import Auth, ASRService, SpeechSynthesisService

# ------------------ ENV / CONFIG ------------------
RIVA_ADDR = os.getenv("RIVA_ADDR", "127.0.0.1:50051")
MT_URL    = os.getenv("MT_URL", "http://127.0.0.1:7010")

A_IN  = os.getenv("A_IN", "plughw:0,0")
A_OUT = os.getenv("A_OUT", "plughw:0,0")
B_IN  = os.getenv("B_IN", "plughw:1,0")
B_OUT = os.getenv("B_OUT", "plughw:1,0")

A_SRC = os.getenv("A_SRC", "es")
A_TGT = os.getenv("A_TGT", "en")
B_SRC = os.getenv("B_SRC", "en")
B_TGT = os.getenv("B_TGT", "es")

VOICE_EN = os.getenv("VOICE_EN", "English-US.Female-1")
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
auth     = Auth(uri=RIVA_ADDR)     # insecure local
riva_asr = ASRService(auth)
riva_tts = SpeechSynthesisService(auth)

def language_code_short_to_riva(lang_short: str) -> str:
    """'en' -> 'en-US', 'es' -> 'es-US' (edit if you want other locales)"""
    if lang_short.startswith("es"):
        return "es-US"
    return "en-US"

def riva_asr_streaming_recognize(pcm_bytes: bytes, lang_short_hint: str) -> str:
    """
    Use Riva streaming ASR even though we already have the full burst.
    We send it in 20ms chunks to mimic live behavior and keep low latency.
    """
    lang_code = language_code_short_to_riva(lang_short_hint)

    # generator of 20ms chunks
    def audio_chunks():
        step = FRAME_SIZE * SAMPLE_BYTES
        for i in range(0, len(pcm_bytes), step):
            yield pcm_bytes[i:i+step]

    # Ask for one final text
    responses = riva_asr.streaming_response_generator(
        audio_chunks(),
        sample_rate_hz=SAMPLE_RATE,
        language_code=lang_code,
        enable_word_time_offsets=False,
        interim_results=False,   # we just want the final result for each burst
        automatic_punctuation=True,
    )

    final_text = ""
    for resp in responses:
        # pick the top alternative if present
        if resp.results:
            alt = resp.results[0].alternatives[0]
            if alt.transcript:
                final_text = alt.transcript

    return final_text.strip()

def riva_tts_speak(text: str, lang_short: str) -> bytes:
    if not text.strip():
        return b""
    lang_code = language_code_short_to_riva(lang_short)
    voice = VOICE_EN if lang_code.startswith("en") else VOICE_ES
    resp = riva_tts.synthesize(
        text=text,
        voice_name=voice,
        language_code=lang_code,
        sample_rate_hz=SAMPLE_RATE,
        encoding="LINEAR_PCM",
    )
    return resp.audio  # PCM16 mono

def http_mt_translate(text: str, src_short: str, tgt_short: str) -> str:
    """Call your local MT server (ctranslate2) over HTTP."""
    if not text.strip() or src_short == tgt_short:
        return text
    try:
        r = requests.post(
            f"{MT_URL}/translate",
            json={"text": text, "source": src_short, "target": tgt_short},
            timeout=5,
        )
        r.raise_for_status()
        # Expecting {"translation": "..."} (adjust if your API differs)
        data = r.json()
        return data.get("translation", "").strip() or text
    except Exception as e:
        print(f"[MT] error: {e}")
        return text

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
                if len(buf) >= min_bytes:
                    burst = buf
                    buf = b""
                    cap.close()
                    return burst
                buf = b""
                speaking = False

# ------------------ PLAYBACK ------------------
def play_pcm(device_name, pcm: bytes):
    if not pcm:
        return
    out = open_playback(device_name)
    for i in range(0, len(pcm), FRAME_SIZE * SAMPLE_BYTES):
        out.write(pcm[i:i + FRAME_SIZE * SAMPLE_BYTES])
    out.close()

# ------------------ MAIN DIRECTION LOOP ------------------
def direction_loop(name: str, in_dev: str, out_dev: str, src_lang: str, tgt_lang: str):
    """
    One direction:
      mic(in_dev) -> VAD burst -> Riva ASR (src_lang) -> MT(src->tgt) -> Riva TTS(tgt) -> speaker(out_dev)
    src_lang/tgt_lang: 'en' or 'es'
    """
    print(f"[{name}] {in_dev} ({src_lang})->({tgt_lang}) {out_dev}")
    vad = webrtcvad.Vad(VAD_LEVEL)

    while True:
        burst = capture_burst(in_dev, vad)
        if not burst:
            continue

        # 1) ASR on Riva (hint language for lower latency)
        text_src = riva_asr_streaming_recognize(burst, src_lang)
        if not text_src:
            continue

        # 2) MT (your local service)
        out_text = http_mt_translate(text_src, src_lang, tgt_lang)

        # 3) TTS (Riva) in target lang
        pcm_tts = riva_tts_speak(out_text, tgt_lang)
        play_pcm(out_dev, pcm_tts)

# ------------------ ENTRY ------------------
def main():
    print("[router] starting…")
    print(f" A: {A_IN}  {A_SRC}->{A_TGT}  -> {B_OUT} (voice {'VOICE_EN' if A_TGT=='en' else 'VOICE_ES'})")
    print(f" B: {B_IN}  {B_SRC}->{B_TGT}  -> {A_OUT} (voice {'VOICE_ES' if B_TGT=='es' else 'VOICE_EN'})")

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
