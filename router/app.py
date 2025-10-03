# app.py  — Router using Riva streaming ASR + local MT + Riva TTS

import os
import time
import threading
import requests
import webrtcvad
import alsaaudio

# ----- Riva client (>=2.19) -----
from riva.client import Auth, ASRService, SpeechSynthesisService
from riva.client.proto import riva_asr_pb2 as rasr


# =========================
# ENUM / COMPAT SHIMS
# =========================
# Audio encoding enum changed names across releases. Normalize to an int.
_EncEnum = getattr(rasr, "AudioEncoding", None) or getattr(rasr, "RivaAudioEncoding", None)
LINEAR_PCM = getattr(_EncEnum, "LINEAR_PCM", 1) if _EncEnum else 1

# Some releases put sample rate as sample_rate_hz, others expect it omitted for streaming.
# We’ll try with and without; see riva_asr_streaming_recognize().


# =========================
# ENV / CONFIG
# =========================
RIVA_ADDR   = os.getenv("RIVA_ADDR", "127.0.0.1:50051")
MT_ENDPOINT = os.getenv("MT_ENDPOINT", "http://127.0.0.1:7010/mt")

# Audio device names (ALSA)
A_IN  = os.getenv("A_IN", "plughw:0,0")
A_OUT = os.getenv("A_OUT", "plughw:0,0")
B_IN  = os.getenv("B_IN", "plughw:1,0")
B_OUT = os.getenv("B_OUT", "plughw:1,0")

# Language per side (2-letter: 'en' / 'es')
A_SRC = os.getenv("A_SRC", "es")
A_TGT = os.getenv("A_TGT", "en")
B_SRC = os.getenv("B_SRC", "en")
B_TGT = os.getenv("B_TGT", "es")

# Riva TTS voices (check with: docker exec riva-speech /opt/riva/clients/riva_tts_client --list_voices --riva_uri=localhost:50051)
VOICE_EN = os.getenv("VOICE_EN", "English-US.Female-1")
VOICE_ES = os.getenv("VOICE_ES", "Spanish-US.Female-1")

# Audio pipeline
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))  # 16 kHz mono PCM16
FRAME_MS    = int(os.getenv("FRAME_MS", "20"))        # 10/20/30ms supported by WebRTC VAD
BURST_MIN   = int(os.getenv("BURST_MIN_MS", "300"))
BURST_MAX   = int(os.getenv("BURST_MAX_MS", "600"))
VAD_LEVEL   = int(os.getenv("VAD_LEVEL", "2"))        # 0..3 (higher = more aggressive)

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
    return "es-US" if code2.lower().startswith("es") else "en-US"


def _make_streaming_config(lang_code: str):
    """
    Build a StreamingRecognitionConfig that works across Riva minor versions.
    We try to set sample_rate_hz in the inner RecognitionConfig; if that
    field is not present in your installed proto, we fall back without it.
    """
    # base config (no sample rate yet)
    base_cfg = rasr.RecognitionConfig(
        encoding=LINEAR_PCM,
        language_code=lang_code,
        max_alternatives=1,
        enable_automatic_punctuation=True,
        audio_channel_count=1,
        verbatim_transcripts=False,
    )

    # Try adding sample_rate_hz if the field exists in this build
    try:
        # hasattr on protobufs returns False for unknown fields; set in try/except
        cfg_with_sr = rasr.RecognitionConfig()
        cfg_with_sr.CopyFrom(base_cfg)
        setattr(cfg_with_sr, "sample_rate_hz", SAMPLE_RATE)
        # Touch to force serialization (will raise if unknown)
        _ = cfg_with_sr.SerializeToString()
        use_cfg = cfg_with_sr
    except Exception:
        use_cfg = base_cfg  # fall back

    return rasr.StreamingRecognitionConfig(
        config=use_cfg,
        interim_results=False,
        enable_word_time_offsets=False,
        max_alternatives=1,
        single_utterance=False,  # we send one burst, but keep API generic
    )


def riva_asr_streaming_recognize(pcm_bytes: bytes, lang_hint_2letter: str | None) -> str:
    """
    Send one burst of PCM16 (16kHz mono) via Riva STREAMING ASR and
    return the final transcript (or "").
    """
    lang_code = _lang2_to_riva(lang_hint_2letter or "en")
    stream_cfg = _make_streaming_config(lang_code)

    # Riva client expects an iterator of bytes chunks (20ms frames are fine)
    chunk = FRAME_SIZE * SAMPLE_BYTES
    audio_chunks = [pcm_bytes[i:i + chunk] for i in range(0, len(pcm_bytes), chunk)]

    # Newer clients accept (audio_chunks, streaming_config=...)
    # Older ones take positional args only. Try both signatures.
    responses = None
    try:
        responses = _riva_asr.streaming_response_generator(audio_chunks, streaming_config=stream_cfg)
    except TypeError:
        responses = _riva_asr.streaming_response_generator(audio_chunks, stream_cfg)

    final_text = ""
    try:
        for resp in responses:
            # Collect only finalized results
            if not resp or not resp.results:
                continue
            for res in resp.results:
                if res.is_final and res.alternatives:
                    final_text = (res.alternatives[0].transcript or "").strip()
        return final_text
    except Exception as e:
        print(f"[Riva ASR] streaming error: {e}")
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
# Your MT endpoint expects: {"text": "...", "source_lang": "en", "target_lang": "es"}

def mt_translate(text: str, src2: str, tgt2: str) -> str:
    text = (text or "").strip()
    if not text or src2 == tgt2:
        return text

    payload = {
        "text": text,
        "source_lang": src2,
        "target_lang": tgt2,
    }
    try:
        r = requests.post(MT_ENDPOINT, json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        return (data.get("text") or data.get("translation") or "").strip()
    except Exception as e:
        print(f"[MT] translate error: {e} | payload={payload}")
        return text  # graceful fallback


# =========================
# VAD BURSTING
# =========================
def capture_burst(device_name: str, vad: webrtcvad.Vad) -> bytes:
    """
    Capture FRAME_MS frames and emit a burst of ~BURST_MIN..BURST_MAX ms when speech ends.
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
      mic(in_dev) -> VAD burst -> Riva streaming ASR (src) -> MT (src->tgt) -> Riva TTS (tgt) -> speaker(out_dev)
    """
    print(f"[{name}] {in_dev} ({src2})->({tgt2}) {out_dev}")
    vad = webrtcvad.Vad(VAD_LEVEL)

    while True:
        burst = capture_burst(in_dev, vad)
        if not burst:
            continue

        # 1) ASR (Riva streaming)
        text_src = riva_asr_streaming_recognize(burst, src2)
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
