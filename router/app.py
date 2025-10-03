#!/usr/bin/env python3
# Low-latency router: short VAD bursts + overlapped ASR/MT/TTS + timed logs.

import os, time, threading, queue, requests, subprocess
import alsaaudio, webrtcvad

from riva.client import Auth, ASRService, SpeechSynthesisService
from riva.client.proto import riva_asr_pb2 as rasr

# ---------- Compat ----------
_EncEnum = getattr(rasr, "AudioEncoding", None) or getattr(rasr, "RivaAudioEncoding", None)
LINEAR_PCM = getattr(_EncEnum, "LINEAR_PCM", 1) if _EncEnum else 1

# ---------- Env / Config ----------
RIVA_ADDR   = os.getenv("RIVA_ADDR", "127.0.0.1:50051")
MT_ENDPOINT = os.getenv("MT_ENDPOINT", os.getenv("MT_URL", "http://127.0.0.1:7010/mt"))

A_IN  = os.getenv("A_IN",  "plug_cap0")   # capture side A
A_OUT = os.getenv("A_OUT", "plug_play0")  # playback to side B
B_IN  = os.getenv("B_IN",  "plug_cap1")
B_OUT = os.getenv("B_OUT", "plug_play1")

A_SRC = os.getenv("A_SRC", "es")  # A speaks this; will be translated into A_TGT
A_TGT = os.getenv("A_TGT", "en")
B_SRC = os.getenv("B_SRC", "en")
B_TGT = os.getenv("B_TGT", "es")

VOICE_EN = os.getenv("VOICE_EN", "English-US.Female-1")
VOICE_ES = os.getenv("VOICE_ES", "Spanish-US.Female-1")

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
FRAME_MS    = int(os.getenv("FRAME_MS", "20"))          # 10/20/30
VAD_LEVEL   = int(os.getenv("VAD_LEVEL", "0"))          # 0..3
BURST_MIN   = int(os.getenv("BURST_MIN_MS", "300"))     # min voiced ms to emit
HANGOVER_MS = int(os.getenv("HANGOVER_MS", "80"))       # silence to end chunk
BURST_MAX   = int(os.getenv("BURST_MAX_MS", "1200"))    # cap chunk size
HARD_TIMEOUT_MS = int(os.getenv("HARD_TIMEOUT_MS", "2500"))

DEBUG = os.getenv("DEBUG", "1") == "1"

CHANNELS     = 1
SAMPLE_BYTES = 2
FRAME_SIZE   = int(SAMPLE_RATE * FRAME_MS / 1000)
FRAME_BYTES  = FRAME_SIZE * SAMPLE_BYTES

# ---------- Riva ----------
_auth = Auth(uri=RIVA_ADDR)
_asr  = ASRService(_auth)
_tts  = SpeechSynthesisService(_auth)

def _lang2_to_riva(code2: str) -> str:
    return "es-US" if (code2 or "").lower().startswith("es") else "en-US"

def _make_stream_cfg(lang_code: str):
    # Minimal RecognitionConfig (some fields vary across releases)
    base = rasr.RecognitionConfig(
        encoding=LINEAR_PCM,
        language_code=lang_code,
        max_alternatives=1,
        audio_channel_count=1,
        enable_automatic_punctuation=True,
        verbatim_transcripts=False,
    )
    # Try to set sample_rate_hz if supported
    try:
        cfg = rasr.RecognitionConfig()
        cfg.CopyFrom(base)
        setattr(cfg, "sample_rate_hz", SAMPLE_RATE)
        _ = cfg.SerializeToString()
        use_cfg = cfg
    except Exception:
        use_cfg = base

    # Build StreamingRecognitionConfig with only stable fields
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

def riva_asr_stream(pcm_bytes: bytes, lang2: str) -> str:
    """Send one short voiced chunk to Riva streaming ASR. Return final best text."""
    lang_code = _lang2_to_riva(lang2)
    cfg = _make_stream_cfg(lang_code)
    frames = [pcm_bytes[i:i+FRAME_BYTES] for i in range(0, len(pcm_bytes), FRAME_BYTES)]

    t0 = time.perf_counter()
    try:
        # Some client builds want named arg, others positional
        try:
            responses = _asr.streaming_response_generator(frames, streaming_config=cfg)
        except TypeError:
            responses = _asr.streaming_response_generator(frames, cfg)
        final = ""
        for resp in responses:
            for res in getattr(resp, "results", []):
                if getattr(res, "is_final", False) and res.alternatives:
                    final = (res.alternatives[0].transcript or "").strip()
        if DEBUG:
            ms = (time.perf_counter() - t0) * 1000
            dur_ms = len(pcm_bytes) / SAMPLE_BYTES / SAMPLE_RATE * 1000
            print(f"[ASR] {lang_code} chunk {dur_ms:.0f} ms -> '{final}' [{ms:.0f} ms]")
        return final
    except Exception as e:
        print(f"[ASR] error: {e}")
        return ""

def riva_tts(text: str, tgt2: str) -> bytes:
    if not text.strip():
        return b""
    lang_code = _lang2_to_riva(tgt2)
    voice = VOICE_EN if lang_code.startswith("en") else VOICE_ES
    t0 = time.perf_counter()
    try:
        resp = _tts.synthesize(
            text=text, voice_name=voice, language_code=lang_code,
            sample_rate_hz=SAMPLE_RATE, encoding="LINEAR_PCM",
        )
        audio = resp.audio or b""
        if DEBUG:
            print(f"[TTS] '{text}' -> {len(audio)} bytes [{(time.perf_counter()-t0)*1000:.0f} ms]")
        return audio
    except Exception as e:
        print(f"[TTS] error: {e}")
        return b""

# ---------- MT ----------
def mt(text: str, src2: str, tgt2: str) -> str:
    text = (text or "").strip()
    if not text or src2 == tgt2:
        return text
    t0 = time.perf_counter()
    try:
        r = requests.post(MT_ENDPOINT, json={"text": text, "source_lang": src2, "target_lang": tgt2}, timeout=10)
        r.raise_for_status()
        data = r.json()
        out = (data.get("text") or data.get("translation") or "").strip()
        if DEBUG:
            print(f"[MT ] {src2}->{tgt2}: '{text}' => '{out}' [{(time.perf_counter()-t0)*1000:.0f} ms]")
        return out
    except Exception as e:
        print(f"[MT ] error: {e}")
        return text

# ---------- ALSA ----------
def open_capture(dev: str):
    if DEBUG: print(f"[audio] CAPTURE {dev} @ {SAMPLE_RATE}Hz mono S16_LE frame={FRAME_MS}ms")
    p = alsaaudio.PCM(type=alsaaudio.PCM_CAPTURE, mode=alsaaudio.PCM_NORMAL, device=dev)
    p.setchannels(1); p.setrate(SAMPLE_RATE); p.setformat(alsaaudio.PCM_FORMAT_S16_LE); p.setperiodsize(FRAME_SIZE)
    return p

def open_playback(dev: str):
    if DEBUG: print(f"[audio] PLAYBACK {dev} @ {SAMPLE_RATE}Hz mono S16_LE frame={FRAME_MS}ms")
    p = alsaaudio.PCM(type=alsaaudio.PCM_PLAYBACK, mode=alsaaudio.PCM_NORMAL, device=dev)
    p.setchannels(1); p.setrate(SAMPLE_RATE); p.setformat(alsaaudio.PCM_FORMAT_S16_LE); p.setperiodsize(FRAME_SIZE)
    return p

# ---------- Queues & Workers ----------
class PlaybackThread(threading.Thread):
    def __init__(self, device: str):
        super().__init__(daemon=True)
        self.device = device
        self.q = queue.Queue()
        self._stop = threading.Event()

    def enqueue(self, pcm: bytes):
        if pcm: self.q.put(pcm)

    def run(self):
        out = open_playback(self.device)
        while not self._stop.is_set():
            try:
                pcm = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            for i in range(0, len(pcm), FRAME_BYTES):
                out.write(pcm[i:i+FRAME_BYTES])

    def stop(self):
        self._stop.set()

def vad_capture_loop(name: str, in_dev: str, out_dev: str, src2: str, tgt2: str):
    print(f"[{name}] {in_dev} ({src2})->({tgt2}) {out_dev}")
    vad = webrtcvad.Vad(VAD_LEVEL)
    cap = open_capture(in_dev)
    pb  = PlaybackThread(out_dev); pb.start()

    def process_chunk(pcm_chunk: bytes):
        t0 = time.perf_counter()
        # ASR
        text_src = riva_asr_stream(pcm_chunk, src2)
        if not text_src:
            return
        # MT
        text_tgt = mt(text_src, src2, tgt2)
        # TTS
        audio = riva_tts(text_tgt, tgt2)
        if audio:
            pb.enqueue(audio)
        if DEBUG:
            print(f"[PIPE] end2end chunk [{(time.perf_counter()-t0)*1000:.0f} ms]")

    # worker pool (2 threads is plenty for Orin Nano)
    workers = [threading.Thread(target=lambda: None)]
    work_q  = queue.Queue()

    def worker():
        while True:
            chunk = work_q.get()
            if chunk is None: break
            process_chunk(chunk)

    workers = [threading.Thread(target=worker, daemon=True) for _ in range(2)]
    for w in workers: w.start()

    buf = bytearray()
    speaking = False
    silence_ms = 0
    voiced_ms  = 0
    hard_ms    = 0

    try:
        while True:
            length, data = cap.read()
            if length <= 0:
                continue
            try:
                speech = vad.is_speech(data, SAMPLE_RATE)
            except Exception:
                speech = False

            if speech:
                if not speaking:
                    speaking = True
                    silence_ms = 0; voiced_ms = 0; hard_ms = 0
                    if DEBUG: print(f"[VAD] START")
                buf += data
                voiced_ms += FRAME_MS
                hard_ms   += FRAME_MS
            else:
                if speaking:
                    silence_ms += FRAME_MS

            # emit conditions
            max_bytes = int(SAMPLE_RATE * BURST_MAX / 1000) * SAMPLE_BYTES
            min_bytes = int(SAMPLE_RATE * BURST_MIN / 1000) * SAMPLE_BYTES

            def emit(reason: str):
                nonlocal buf, speaking, silence_ms, voiced_ms, hard_ms
                chunk = bytes(buf)
                dur_ms = len(chunk) / SAMPLE_BYTES / SAMPLE_RATE * 1000
                if DEBUG: print(f"[VAD] {reason} -> emit {dur_ms:.0f} ms, queue")
                if chunk:
                    work_q.put(chunk)
                buf.clear()
                speaking = False
                silence_ms = voiced_ms = hard_ms = 0

            if speaking and hard_ms >= HARD_TIMEOUT_MS:
                emit("HARD_TIMEOUT")

            elif speaking and len(buf) >= max_bytes:
                emit("MAX_LEN")

            elif speaking and (silence_ms >= HANGOVER_MS and len(buf) >= min_bytes):
                emit("END")

    except KeyboardInterrupt:
        pass
    finally:
        for _ in workers: work_q.put(None)
        pb.stop()

# ---------- Entry ----------
def _print_cards():
    for cmd in (["arecord","-l"], ["arecord","-L"]):
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
            print(f"$ {' '.join(cmd)}\n{out.strip()}\n")
        except Exception as e:
            print(f"(could not run {' '.join(cmd)}: {e})")

def main():
    print("[router] starting…")
    _print_cards()
    print(f" A: {A_IN}  {A_SRC}->{A_TGT} -> {B_OUT} (voice {'EN' if A_TGT.startswith('en') else 'ES'})")
    print(f" B: {B_IN}  {B_SRC}->{B_TGT} -> {A_OUT} (voice {'ES' if B_TGT.startswith('es') else 'EN'})")
    if DEBUG:
        print(f"[cfg] SR={SAMPLE_RATE}Hz FRAME={FRAME_MS}ms VAD={VAD_LEVEL} "
              f"MIN={BURST_MIN}ms HANGOVER={HANGOVER_MS}ms MAX={BURST_MAX}ms TIMEOUT={HARD_TIMEOUT_MS}ms")

    tA = threading.Thread(target=vad_capture_loop, args=("A", A_IN, B_OUT, A_SRC, A_TGT), daemon=True)
    tB = threading.Thread(target=vad_capture_loop, args=("B", B_IN, A_OUT, B_SRC, B_TGT), daemon=True)
    tA.start(); tB.start()

    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting.")

if __name__ == "__main__":
    main()
