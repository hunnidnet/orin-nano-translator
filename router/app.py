import os, time, threading, queue, requests, struct
import numpy as np
import webrtcvad
import alsaaudio

# ---------- ENV / CONFIG ----------
CANARY_URL = os.getenv("CANARY_URL", "http://127.0.0.1:7000/ast")
RIVA_ADDR  = os.getenv("RIVA_ADDR", "127.0.0.1:50051")

A_IN  = os.getenv("A_IN", "hw:2,0")
A_OUT = os.getenv("A_OUT", "hw:2,0")
B_IN  = os.getenv("B_IN", "hw:3,0")
B_OUT = os.getenv("B_OUT", "hw:3,0")

A_SRC = os.getenv("A_SRC", "es")
A_TGT = os.getenv("A_TGT", "en")
B_SRC = os.getenv("B_SRC", "en")
B_TGT = os.getenv("B_TGT", "es")

RIVA_VOICE_A = os.getenv("RIVA_VOICE_A", "en-US-Polyglot-1")
RIVA_VOICE_B = os.getenv("RIVA_VOICE_B", "es-ES-Polyglot-1")

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
FRAME_MS    = int(os.getenv("FRAME_MS", "20"))
BURST_MIN   = int(os.getenv("BURST_MIN_MS", "300"))
BURST_MAX   = int(os.getenv("BURST_MAX_MS", "600"))
VAD_LEVEL   = int(os.getenv("VAD_LEVEL", "2")) # 0-3

CHANNELS = 1
SAMPLE_BYTES = 2  # S16_LE

# ---------- RIVA TTS CLIENT ----------
# Try to import either package name.
TTSService = None
try:
    from riva.client import TTSService as _TTSService
    TTSService = _TTSService
except Exception:
    try:
        from nvidia.riva.client import TTSService as _TTSService
        TTSService = _TTSService
    except Exception:
        TTSService = None

def riva_synthesize(text: str, voice_name: str, sample_rate=SAMPLE_RATE) -> bytes:
    """
    Returns PCM16 mono bytes from Riva TTS (blocking).
    """
    if TTSService is None:
        # As a fallback, synthesize silence with short beep so pipeline stays alive
        dur = max(0.2, min(3.0, len(text)/12.0))
        t = np.linspace(0, dur, int(dur*sample_rate), endpoint=False)
        beep = 0.05*np.sin(2*np.pi*440*t)
        return (beep*32767).astype(np.int16).tobytes()

    svc = TTSService(RIVA_ADDR)
    # Depending on the client, the call signature names may vary slightly.
    audio = svc.synthesize(
        text=text,
        language_code=None,  # voice implies language
        encoding="LINEAR_PCM",
        sample_rate_hz=sample_rate,
        voice_name=voice_name,
    )
    # audio is bytes PCM16 mono
    return audio

# ---------- ALSA HELPERS ----------
def open_capture(device_name):
    pcm = alsaaudio.PCM(type=alsaaudio.PCM_CAPTURE, mode=alsaaudio.PCM_NORMAL, device=device_name)
    pcm.setchannels(CHANNELS)
    pcm.setrate(SAMPLE_RATE)
    pcm.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    frame_size = int(SAMPLE_RATE * FRAME_MS / 1000)
    pcm.setperiodsize(frame_size)
    return pcm, frame_size

def open_playback(device_name):
    pcm = alsaaudio.PCM(type=alsaaudio.PCM_PLAYBACK, mode=alsaaudio.PCM_NORMAL, device=device_name)
    pcm.setchannels(CHANNELS)
    pcm.setrate(SAMPLE_RATE)
    pcm.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    frame_size = int(SAMPLE_RATE * FRAME_MS / 1000)
    pcm.setperiodsize(frame_size)
    return pcm

# ---------- PIPELINE ----------
def capture_loop(dev_name, out_q, frame_bytes):
    cap, _ = open_capture(dev_name)
    while True:
        length, data = cap.read()
        if length > 0:
            out_q.put(data)

def vad_burster(in_q, out_q, vad: webrtcvad.Vad):
    """
    Assemble 20ms PCM frames into 300-600ms bursts when speech is detected.
    """
    max_bytes = int(SAMPLE_RATE * BURST_MAX / 1000) * SAMPLE_BYTES
    min_bytes = int(SAMPLE_RATE * BURST_MIN / 1000) * SAMPLE_BYTES
    buf = b""
    speaking = False
    while True:
        frame = in_q.get()
        if len(frame) == 0:
            continue
        is_speech = vad.is_speech(frame, SAMPLE_RATE)
        if is_speech:
            speaking = True
            buf += frame
            if len(buf) >= max_bytes:
                out_q.put(buf)
                buf = b""
                speaking = False
        else:
            if speaking:
                if len(buf) >= min_bytes:
                    out_q.put(buf)
                buf = b""
            speaking = False

def canary_translate(pcm_bytes: bytes, src_lang: str, tgt_lang: str) -> str:
    payload = {
        "pcm16_hex": pcm_bytes.hex(),
        "sr": SAMPLE_RATE,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang
    }
    r = requests.post(CANARY_URL, json=payload, timeout=5)
    r.raise_for_status()
    return r.json().get("translation", "")

def playback_loop(dev_name, in_q):
    out = open_playback(dev_name)
    while True:
        pcm = in_q.get()
        out.write(pcm)

def direction_worker(name, in_dev, out_dev, src_lang, tgt_lang, riva_voice):
    """
    One direction: mic -> VAD -> Canary -> Riva TTS -> speaker
    """
    raw_q = queue.Queue(maxsize=100)
    burst_q = queue.Queue(maxsize=10)
    out_q = queue.Queue(maxsize=10)
    vad = webrtcvad.Vad(VAD_LEVEL)

    frame_bytes = int(SAMPLE_RATE * FRAME_MS / 1000) * SAMPLE_BYTES

    # Capture
    threading.Thread(target=capture_loop, args=(in_dev, raw_q, frame_bytes), daemon=True).start()
    # VAD -> bursts
    threading.Thread(target=vad_burster, args=(raw_q, burst_q, vad), daemon=True).start()
    # Playback
    threading.Thread(target=playback_loop, args=(out_dev, out_q), daemon=True).start()

    print(f"[{name}] running: {in_dev} -> {src_lang}->{tgt_lang} -> {out_dev}")

    while True:
        burst = burst_q.get()
        try:
            text = canary_translate(burst, src_lang, tgt_lang)
            if text.strip():
                pcm_tts = riva_synthesize(text, riva_voice, SAMPLE_RATE)
                out_q.put(pcm_tts)
        except Exception as e:
            print(f"[{name}] error: {e}")

def main():
    print("[router] starting…")
    print(f" A: {A_IN} -> {A_TGT} via {A_SRC}  out:{A_OUT}  voice:{RIVA_VOICE_B if A_TGT=='es' else RIVA_VOICE_A}")
    print(f" B: {B_IN} -> {B_TGT} via {B_SRC}  out:{B_OUT}  voice:{RIVA_VOICE_A if B_TGT=='en' else RIVA_VOICE_B}")

    # Direction A: typically Spanish -> English (to device B_OUT)
    voice_for_A = RIVA_VOICE_A if A_TGT == "en" else RIVA_VOICE_B
    tA = threading.Thread(target=direction_worker, args=("A", A_IN, B_OUT, A_SRC, A_TGT, voice_for_A), daemon=True)
    tA.start()

    # Direction B: typically English -> Spanish (to device A_OUT)
    voice_for_B = RIVA_VOICE_B if B_TGT == "es" else RIVA_VOICE_A
    tB = threading.Thread(target=direction_worker, args=("B", B_IN, A_OUT, B_SRC, B_TGT, voice_for_B), daemon=True)
    tB.start()

    # Keep alive
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
