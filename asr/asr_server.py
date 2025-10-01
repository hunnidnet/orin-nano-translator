import os, io
import numpy as np
import soundfile as sf
from fastapi import FastAPI
from pydantic import BaseModel
from faster_whisper import WhisperModel

FW_MODEL = os.getenv("FW_MODEL", "small")           # tiny, base, small, medium, large-v3
FW_DEVICE = os.getenv("FW_DEVICE", "cuda")          # "cuda" on Jetson
FW_COMPUTE = os.getenv("FW_COMPUTE_TYPE", "float16")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))

app = FastAPI()
model = None

@app.on_event("startup")
def _load():
    global model
    model = WhisperModel(FW_MODEL, device=FW_DEVICE, compute_type=FW_COMPUTE)

class ASRIn(BaseModel):
    pcm16_hex: str
    sr: int = 16000
    lang: str | None = None  # e.g. "en", "es"; if None, auto-detect

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": FW_MODEL,
        "device": FW_DEVICE,
        "compute": FW_COMPUTE,
        "sr": SAMPLE_RATE,
        "loaded": model is not None,
    }

@app.post("/asr/chunk")
def asr_chunk(req: ASRIn):
    # Convert PCM16 mono bytes → float32
    pcm = np.frombuffer(bytes.fromhex(req.pcm16_hex), dtype=np.int16).astype(np.float32) / 32768.0
    if pcm.size == 0:
        return {"text": "", "lang": req.lang or ""}

    # Resample to 16k if needed (simple WAV roundtrip to keep deps light)
    if req.sr != SAMPLE_RATE:
        with io.BytesIO() as b:
            sf.write(b, pcm, req.sr, format="WAV", subtype="PCM_16")
            b.seek(0)
            audio, sr = sf.read(b)
        with io.BytesIO() as b2:
            sf.write(b2, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
            b2.seek(0)
            pcm, _ = sf.read(b2)

    # faster-whisper accepts np.ndarray at 16k
    language = req.lang  # None = auto
    segments, info = model.transcribe(
        pcm,
        beam_size=1,
        vad_filter=True,
        language=language,
        task="transcribe",    # text in the same language; you translate later in CT2
        no_speech_threshold=0.6,
        condition_on_previous_text=False,
    )

    text = "".join(seg.text for seg in segments).strip()
    lang = language or (info.language if info is not None else "")
    # normalize lang to 2 letters when possible
    lang = (lang or "")[:2]
    return {"text": text, "lang": lang}
