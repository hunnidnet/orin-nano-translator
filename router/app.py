# app.py — Router using Riva STREAMING ASR + local MT + Riva TTS (with robust VAD + DEBUG)

import os
import time
import subprocess
import threading
import requests
import webrtcvad
import alsaaudio

from riva.client import Auth, ASRService, SpeechSynthesisService
from riva.client.proto import riva_asr_pb2 as rasr

# ---------- Compat ----------
_EncEnum = getattr(rasr, "AudioEncoding", None) or getattr(rasr, "RivaAudioEncoding", None)
LINEAR_PCM = getattr(_EncEnum, "LINEAR_PCM", 1) if _EncEnum else 1

# ---------- Env / Config ----------
RIVA_ADDR   = os.getenv("RIVA_ADDR", "127.0.0.1:50051")
MT_ENDPOINT = os.getenv("MT_ENDPOINT", "http://127.0.0.1:7010/mt")

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

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))   # 16kHz mono
FRAME_MS    = int(os.getenv("FRAME_MS", "20"))         # 10/20/30 allowed by WebRTC VAD
BURST_MIN   = int(os.getenv("BURST_MIN_MS", "200"))    # min voiced ms to emit
BURST_MAX   = int(os.getenv("BURST_MAX_MS", "1200"))   # max ms before forced emit
VAD_LEVEL   = int(os.getenv("VAD_LEVEL", "1"))         # 0..3 (more = more aggressive)
HANGOVER_MS = int(os.getenv("HANGOVER_MS", "150"))     # silence needed to end utterance
HARD_TIMEOUT_MS = int(os.getenv("HARD_TIMEOUT_MS", "2500"))  # force emit after this many ms

DEBUG = os.getenv("DEBUG", "0") == "1"

CHANNELS     = 1
SAMPLE_BYTES = 2
FRAME_SIZE   = int(SAMPLE_RATE * FRAME_MS / 1000)      # samples per frame
FRAME_BYTES  = FRAME_SIZE * SAMPLE_BYTES

# ---------- ALSA helpers ----------
def open_capture(device_name: str):
    if DEBUG: print(f"[audio] open CAPTURE {device_name} @ {SAMPLE_RATE}Hz, 1ch, S16_LE, {FRAME_MS}ms")
    pcm = alsaaudio.PCM(type=alsaaudio.PCM_CAPTURE, mode=alsaaudio.PCM_NORMAL, device=device_name)
    pcm.setchannels(CHANNELS)
    pcm.setrate(SAMPLE_RATE)
    pcm.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    pcm.setperiodsize(FRAME_SIZE)
    return pcm

def open_playback(device_name: str):
    if DEBUG: print(f"[audio] open PLAYBACK {device_name} @ {SAMPLE_RATE}Hz, 1ch, S16_LE, {FRAME_MS}ms")
    pcm = alsaaudio.PCM(type=alsaaudio.PCM_PLAYBACK, mode=alsaaudio.PCM_NORMAL, device=device_name)
    pcm.setchannels(CHANNELS)
    pcm.setrate(SAMPLE_RATE)
    pcm.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    pcm.setperiodsize(FRAME_SIZE)
    return pcm

# ---------- Riva clients ----------
_riva_auth = Auth(uri=RIVA_ADDR)
_riva_asr  = ASRService(_riva_auth)
_riva_tts  = SpeechSynthesisService(_riva_auth)

def _lang2_to_riva(code2: str) -> str:
    return "es-US" if (code2 or "").lower().startswith("es") else "en-US"

def _make_streaming_config(lang_code: str):
    base_cfg = rasr.RecognitionConfig(
        encoding=LINEAR_PCM,
        language_code=lang_code,
        max_alternatives=1,
        audio_channel_count=1,
        enable_automatic_punctuation=True,
        verbatim_transcripts=False,
    )
    # add sample_rate_hz if the field exists in this build
    try:
        cfg = rasr.RecognitionConfig()
        cfg.CopyFrom(base_cfg)
        setattr(cfg, "sample_rate_hz", SAMPLE_RATE)
        _ = cfg.SerializeToString()
        use_cfg = cfg
    except Exception:
        use_cfg = base_cfg

    # build StreamingRecognitionConfig with only widely supported fields
    for kwargs in (
        dict(config=use_cfg, interim_results=False, max_alternatives=1, single_utterance=False),
        dict(config=use_cfg, interim_results=False, max_alternatives=1),
        dict(config=use_cfg, interim_results=False),
        dict(config=use_cfg),
    ):
        try:
            s = rasr.StreamingRecognitionConfig(**kwargs)
            _ = s.SerializeToString()
            return s
        except Exception:
            continue
    return rasr.StreamingRecognitionConfig(config=use_cfg)

def riva_asr_streaming_recognize(pcm_bytes: bytes, lang_hint_2letter: str | None) -> str:
    lang_code = _lang2_to_riva(lang_hint_2letter or "en")
    stream_cfg = _make_streaming_config(lang_code)

    chunks = [pcm_bytes[i:i + FRAME_BYTES] for i in range(0, len(pcm_bytes), FRAME_BYTES)]
    if DEBUG:
        ms = (len(pcm_bytes) / SAMPLE_BYTES) / SAMPLE_RATE * 1000.0
        print(f"[ASR] sending {len(chunks)} frames ({ms:.0f} ms) lang={lang_code}")

    try:
        try:
            responses = _riva_asr.streaming_response_generator(chunks, streaming_config=stream_cfg)
        except TypeError:
            responses = _riva_asr.streaming_response_generator(chunks, stream_cfg)
        final_text = ""
        for resp in responses:
            if not resp or not resp.results:
                continue
            for res in resp.results:
                if getattr(res, "is_final", False) and res.alternatives:
                    final_text = (res.alternatives[0].transcript or "").strip()
        if DEBUG and final_text:
            print(f"[ASR] final: {final_text}")
        return final_text
    except Exception as e:
        print(f"[Riva ASR] streaming error: {e}")
        return ""

def riva_tts_speak(text: str, tgt_lang2: str) -> bytes:
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
        if DEBUG:
            print(f"[TTS] {len(resp.audio)} bytes ({len(resp.audio)/SAMPLE_BYTES/SAMPLE_RATE*1000:.0f} ms)")
        return resp.audio or b""
    except Exception as e:
        print(f"[Riva TTS] synth error: {e}")
        return b""

# ---------- MT ----------
def mt_translate(text: str, src2: str, tgt2: str) -> str:
    text = (text or "").strip()
    if not text or src2 == tgt2:
        return text
    payload = {"text": text, "source_lang": src2, "target_lang": tgt2}
    try:
        r = requests.post(MT_ENDPOINT, json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        out = (data.get("text") or data.get("translation") or "").strip()
        if DEBUG and out:
            print(f"[MT] {src2}->{tgt2}: {text}  =>  {out}")
        return out
    except Exception as e:
        print(f"[MT] translate error: {e} | payload={payload}")
        return text

# ---------- VAD bursting (robust) ----------
def capture_burst(device_name: str, vad: webrtcvad.Vad) -> bytes:
    """
    Emit a burst when: speech started, then either:
      - 150ms of silence (hangover), or
      - hard timeout reached (2.5s), or
      - buffer > BURST_MAX
    """
    cap = open_capture(device_name)
    buf = bytearray()
    speaking = False
    silence_ms = 0
    voiced_ms = 0
    hard_ms = 0

    while True:
        length, data = cap.read()
        if length <= 0:
            continue

        try:
            is_speech = vad.is_speech(data, SAMPLE_RATE)
        except Exception:
            is_speech = False

        if is_speech:
            buf += data
            voiced_ms += FRAME_MS
            hard_ms += FRAME_MS
            silence_ms = 0
            if not speaking:
                speaking = True
                if DEBUG: print("[VAD] speech START")
        else:
            if speaking:
                silence_ms += FRAME_MS
            # while not speaking, we ignore

        # Hard stop if too long
        if speaking and hard_ms >= HARD_TIMEOUT_MS:
            if DEBUG:
                print(f"[VAD] HARD TIMEOUT -> emit ({len(buf)/SAMPLE_BYTES/SAMPLE_RATE*1000:.0f} ms)")
            out = bytes(buf)
            cap.close()
            return out

        # Normal stop on hangover silence
        if speaking and silence_ms >= HANGOVER_MS and voiced_ms >= BURST_MIN:
            if DEBUG:
                print(f"[VAD] speech END (silence {silence_ms} ms) -> emit ({len(buf)/SAMPLE_BYTES/SAMPLE_RATE*1000:.0f} ms)")
            out = bytes(buf)
            cap.close()
            return out

        # Force emit on very large burst
        if speaking and (len(buf) >= int(SAMPLE_RATE * BURST_MAX / 1000) * SAMPLE_BYTES):
            if DEBUG:
                print(f"[VAD] FORCE EMIT (max {BURST_MAX} ms) -> emit ({len(buf)/SAMPLE_BYTES/SAMPLE_RATE*1000:.0f} ms)")
            out = bytes(buf)
            cap.close()
            return out

# ---------- Playback ----------
def play_pcm(device_name: str, pcm: bytes):
    if not pcm:
        return
    out = open_playback(device_name)
    for i in range(0, len(pcm), FRAME_BYTES):
        out.write(pcm[i:i + FRAME_BYTES])
    out.close()

# ---------- Direction loop ----------
def direction_loop(name: str, in_dev: str, out_dev: str, src2: str, tgt2: str):
    print(f"[{name}] {in_dev} ({src2})->({tgt2}) {out_dev}")
    vad = webrtcvad.Vad(VAD_LEVEL)

    while True:
        burst = capture_burst(in_dev, vad)
        if not burst:
            continue

        # ASR (streaming)
        text_src = riva_asr_streaming_recognize(burst, src2)
        if not text_src:
            if DEBUG: print("[ASR] empty result")
            continue

        # NMT
        out_text = text_src if src2 == tgt2 else mt_translate(text_src, src2, tgt2)

        # TTS
        pcm_tts = riva_tts_speak(out_text, tgt2)
        play_pcm(out_dev, pcm_tts)

# ---------- Entry ----------
def _print_alsa_inventory():
    # Helpful when device names are wrong in container
    for cmd in (["arecord", "-l"], ["arecord", "-L"]):
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
            print(f"$ {' '.join(cmd)}\n{out.strip()}\n")
        except Exception as e:
            print(f"(could not run {' '.join(cmd)}: {e})")

def main():
    print("[router] starting…")
    _print_alsa_inventory()
    print(f" A: {A_IN}  {A_SRC}->{A_TGT}  -> {B_OUT} (voice {VOICE_EN if A_TGT.startswith('en') else VOICE_ES})")
    print(f" B: {B_IN}  {B_SRC}->{B_TGT}  -> {A_OUT} (voice {VOICE_ES if B_TGT.startswith('es') else VOICE_EN})")
    if DEBUG:
        print(f"[cfg] SR={SAMPLE_RATE}Hz, FRAME={FRAME_MS}ms, VAD={VAD_LEVEL}, "
              f"MIN={BURST_MIN}ms, MAX={BURST_MAX}ms, HANGOVER={HANGOVER_MS}ms, TIMEOUT={HARD_TIMEOUT_MS}ms")

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
