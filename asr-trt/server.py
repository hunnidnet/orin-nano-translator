import os
import io
import json
import numpy as np
import soundfile as sf
from fastapi import FastAPI
from pydantic import BaseModel
from whisper_trt.model import WhisperTRT

TRT_MODEL_SIZE = os.getenv("TRT_MODEL_SIZE", "small")   # tiny/base/small/medium
TRT_COMPUTE    = os.getenv("TRT_COMPUTE", "int8")       # int8 or fp16
SAMPLE_RATE    = int(os.getenv("SAMPLE_RATE", "16000"))
CACHE_DIR      = os.getenv("CACHE_DIR", "/cache")       # persisted engine cache

app = FastAPI()
asr = None

class AsrChunk(BaseModel):
    pcm16_hex: str
    sr: int = SAMPLE_RATE
    lang: str | None = None    # e.g. "en" or "es"; None = auto

@app.on_event("startup")
def _startup():
    global asr
    # Construct WhisperTRT => builds or loads TensorRT engine in CACHE_DIR
    asr = WhisperTRT(
        model_size=TRT_MODEL_SIZE,
        compute_type=TRT_COMPUTE,
        cache_dir=CACHE_DIR
    )

@app.get("/health")
def health():
    return {"status": "ok", "model": TRT_MODEL_SIZE, "compute": TRT_COMPUTE}

@app.post("/asr/chunk")
def asr_chunk(req: AsrChunk):
    # bytes -> float32 mono
    pcm = np.frombuffer(bytes.fromhex(req.pcm16_hex), dtype=np.int16).astype(np.float32) / 32768.0

    # resample (quick and simple) if needed
    if req.sr != SAMPLE_RATE and len(pcm) > 0:
        with io.BytesIO() as b:
            sf.write(b, pcm, req.sr, format="WAV", subtype="PCM_16")
            b.seek(0)
            data, _ = sf.read(b)
        with io.BytesIO() as b2:
            sf.write(b2, data, SAMPLE_RATE, format="WAV", subtype="PCM_16")
            b2.seek(0)
            pcm, _ = sf.read(b2)

    # transcribe
    # language=None lets Whisper auto-detect (covers both English & Spanish)
    text, lang, meta = asr.transcribe(
        pcm,
        sample_rate=SAMPLE_RATE,
        language=req.lang  # pass "en" or "es" to override auto
    )
    # meta may contain avg_logprob, no_speech_prob, etc.
    out = {
        "text": text or "",
        "lang": lang or "",
    }
    if isinstance(meta, dict):
        out.update({k: meta[k] for k in ("avg_logprob","no_speech_prob") if k in meta})
    return out
