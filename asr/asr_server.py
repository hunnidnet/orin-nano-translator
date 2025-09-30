import os, io, time
import numpy as np
import soundfile as sf
from fastapi import FastAPI
from pydantic import BaseModel
from faster_whisper import WhisperModel

ASR_MODEL = os.getenv("ASR_MODEL", "small")  # tiny/base/small/medium
DEVICE = os.getenv("ASR_DEVICE", "cuda")
CTYPE  = os.getenv("ASR_COMPUTE_TYPE", "int8_float16")  # good on Jetson

model = None
app = FastAPI()

@app.on_event("startup")
def _startup():
    global model
    model = WhisperModel(ASR_MODEL, device=DEVICE, compute_type=CTYPE,
                         cpu_threads=2, num_workers=1)

class Chunk(BaseModel):
    pcm16_hex: str
    sr: int = 16000
    lang: str|None = None  # "en" or "es" or None for auto

@app.post("/asr/chunk")
def asr_chunk(req: Chunk):
    pcm = np.frombuffer(bytes.fromhex(req.pcm16_hex), dtype=np.int16).astype(np.float32)/32768.0
    # faster-whisper likes files; we’ll give it a small wav
    with io.BytesIO() as b:
        sf.write(b, pcm, req.sr, format="WAV", subtype="PCM_16")
        b.seek(0)
        segments, info = model.transcribe(b, language=req.lang, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=150))
    text = "".join([s.text for s in segments]).strip()
    lang = info.language if info and info.language else (req.lang or "auto")
    return {"text": text, "lang": lang}
