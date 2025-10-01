import os, io, numpy as np, soundfile as sf
from fastapi import FastAPI
from pydantic import BaseModel

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
DEFAULT_LANG = os.getenv("DEFAULT_LANG", "")  # "", "en", "es"
ENGINE_DIR = os.getenv("ENGINE_DIR", "/engines")   # mount your TRT engines here
MODEL_NAME = os.getenv("MODEL_NAME", "small")      # small/base/…

# ---------- LOAD whisper_trt ----------
# TODO: replace this block with the correct import/constructor from your repo
# Example (pseudo):
# from whisper_trt import TRTWhisper
# asr = TRTWhisper(engine_dir=ENGINE_DIR, model_name=MODEL_NAME, fp16=True)

asr = None
try:
    # Raise if you haven't wired the class yet:
    raise NotImplementedError("Wire whisper_trt class here")
except Exception as e:
    # We still start, but reply with a clear error on first call
    asr = e

# ---------- API ----------
app = FastAPI()

class ASRIn(BaseModel):
    pcm16_hex: str
    sr: int = SAMPLE_RATE
    lang: str | None = None

def _bytes_to_pcm(b, sr):
    pcm = np.frombuffer(bytes.fromhex(b), dtype=np.int16).astype(np.float32)/32768.0
    if sr != SAMPLE_RATE and len(pcm) > 0:
        with io.BytesIO() as buf:
            sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
            buf.seek(0); data, _sr = sf.read(buf)
        with io.BytesIO() as buf2:
            sf.write(buf2, data, SAMPLE_RATE, format="WAV", subtype="PCM_16")
            buf2.seek(0); pcm, _ = sf.read(buf2)
    return pcm

@app.get("/health")
def health():
    return {
        "status": "ok" if not isinstance(asr, Exception) else "error",
        "engine_dir": ENGINE_DIR,
        "model": MODEL_NAME,
        "note": "Replace the TODO in server_trt_shim.py to initialize whisper_trt"
                if isinstance(asr, Exception) else "ready"
    }

@app.post("/asr/chunk")
def asr_chunk(req: ASRIn):
    if isinstance(asr, Exception):
        return {"text": "", "lang": "", "error": str(asr)}

    pcm = _bytes_to_pcm(req.pcm16_hex, req.sr)
    lang = req.lang or DEFAULT_LANG or None

    # TODO: call your whisper_trt inference here.
    # Pseudo:
    # text, detected = asr.transcribe_pcm(pcm, sr=SAMPLE_RATE, language=lang)
    # return {"text": text, "lang": detected or (lang or "")}

    return {"text": "", "lang": "", "error": "Not wired to whisper_trt yet"}
