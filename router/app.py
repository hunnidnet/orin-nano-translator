import os
import time
import threading
import requests
import webrtcvad
import alsaaudio

# ----- Riva client (2.19.0) -----
from riva.client import Auth, ASRService, SpeechSynthesisService
from riva.client.proto import riva_asr_pb2 as rasr  # use module, not direct names

# =========================
# ENUM/CONFIG COMPAT SHIM
# =========================
# Some Riva client versions export the encoding enum as AudioEncoding,
# others as RivaAudioEncoding. We normalize here.
_EncEnum = getattr(rasr, "AudioEncoding", None) or getattr(rasr, "RivaAudioEncoding", None)
if _EncEnum is None:
    # Fallback: LINEAR_PCM is 1 in current schemas; keep safe default
    LINEAR_PCM = 1
else:
    LINEAR_PCM = getattr(_EncEnum, "LINEAR_PCM", 1)

RecognitionConfig = rasr.RecognitionConfig

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

# Language per side (2-letter: 'en' / 'es')
A_SRC = os.getenv("A_SRC", "es")
A_TGT = os.getenv("A_TGT", "en")
B_SRC = os.getenv("B_SRC", "en")
B_TGT = os.getenv("B_TGT", "es")

# Riva TTS voices (check with riva_tts_client --list_voices)
VOICE_EN = os.getenv("VOICE_EN", "English-US.Female-1")
VOICE_ES = os.getenv("VOICE_ES", "Spanish-US.Female-1")

# Audio pipeline
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))  # 16 kHz mono PCM16
FRAME_MS    = int(os.getenv("FRAME_MS", "20"))        # 10/20/30ms supported by WebRTC VAD
BURST_MIN   = int(os.getenv("BURST_MIN_MS", "300"))   # min voiced duration before emit
BURST_MAX   = int(os.getenv("BURST_MAX_MS", "600"))   # max burst size before forced emit
VAD_LEVEL   = int(os.getenv("VAD_LEVEL", "2"))        # 0..3 (more = more aggressive)

CHANNELS     = 1
SAMPLE_BYTES = 2
FRAME_SIZE   = int(SAMPLE_RATE * FRAME_MS / 1000)     # samples per frame

# =========================
# ALSA HELPERS
# =========================
def open_capture(device_name: str):
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

def _lang2_to_riva(code2: str) -> str:
    """Map 'en'->'en-US', 'es'->'es-US'. Default 'en-US'."""
    if not code2:
        return "en-US"
    code2 = code2.lower()
    return "es-US" if code2.startswith("es") else "en-US"

def riva_asr_offline_recognize(pcm_bytes: bytes, lang_hint_2letter: str | None) -> str:
    """
    Recognize one burst of PCM16 (16kHz mono) using Riva OFFLINE ASR.
    Returns the top transcript (string) or "".
    """
    lang_code = _lang2_to_riva(lang_hint_2letter or "en")

    cfg = RecognitionConfig(
        encoding=LINEAR_PCM,
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
            return (res.alternatives[0].transcript or "").strip()
    return ""

def riva_tts_speak(text: str, tgt_lang2: str) -> bytes:
    """
    Synthesize PCM16 mono via Riva TTS at SAMPLE_RATE.
    """
    text = (text or "").strip()
    if not text:
        return b""

    lang_code = _lang2_to_riva(tgt_lang2)
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
def mt_translate(text: str, src2: str, tgt2: str) -> str:
    """
    POST {MT_URL}/translate  JSON: {"text": "...", "source": "en", "target": "es"}
    Accepts either {"text": "..."} or {"translation": "..."} in response.
    """
    text = (text or "").strip()
    if not text or src2 == tgt2:
        return text

    try:
        r = requests.post(
            f"{MT_URL}/translate",
            json={"text": text, "source": src2, "target": tgt2},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
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

        try:
            is_speech = vad.is_speech(data, SAMPLE_RATE)
        except Exception:
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
                if len(buf) >= min_bytes:
                    burst = buf
                    buf = b""
                    cap.close()
                    return burst
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
def direction_loop(name: str, in_dev: str, out_dev: str, src2: str, tgt2: str):
    """
    Direction:
      mic(in_dev) -> VAD burst -> Riva ASR (src) -> MT (src->tgt) -> Riva TTS (tgt) -> speaker(out_dev)
    """
    print(f"[{name}] {in_dev} ({src2})->({tgt2}) {out_dev}")
    vad = webrtcvad.Vad(VAD_LEVEL)

    while True:
        burst = capture_burst(in_dev, vad)
        if not burst:
            continue

        # 1) ASR (Riva offline)
        text_src = riva_asr_offline_recognize(burst, src2)
        if not text_src:
            continue

        # 2) MT if needed
        out_text = text_src if src2 == tgt2 else mt_translate(text_src, src2, tgt2)

        # 3) TTS in target language
        pcm_tts = riva_tts_speak(out_text, tgt2)
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
