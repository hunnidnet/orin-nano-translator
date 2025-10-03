# asr-fastwhisper/server.py
from fastapi import FastAPI
from pydantic import BaseModel
from faster_whisper import WhisperModel
import base64, io, soundfile as sf, os

FW_MODEL = os.getenv("FW_MODEL", "small")
FW_DEVICE = os.getenv("FW_DEVICE", "cuda")
FW_COMPUTE = os.getenv("FW_COMPUTE_TYPE", "float16")

app = FastAPI()
model = None

@app.on_event("startup")
def _load():
    global model
    model = WhisperModel(FW_MODEL, device=FW_DEVICE, compute_type=FW_COMPUTE)

@app.get("/health")
def health():
    import ctranslate2 as ct2
    return {
        "status": "ok",
        "fw_model": FW_MODEL,
        "device": FW_DEVICE,
        "compute": FW_COMPUTE,
        "ct2": ct2.__version__,
        "ct2_path": ct2.__file__,
        "ct2_supported_cuda_types": list(ct2.get_supported_compute_types("cuda")),
    }

class TranscribeIn(BaseModel):
    audio_b64: str            # base64 WAV/PCM16 @ 16k
    language: str | None = None

@app.post("/asr")
def asr(req: TranscribeIn):
    audio_bytes = base64.b64decode(req.audio_b64)
    buf = io.BytesIO(audio_bytes)
    # autodetect wav params
    data, sr = sf.read(buf, dtype="int16")
    if sr != 16000:
        return {"error": f"expected 16kHz, got {sr}"}
    segments, info = model.transcribe(data, language=req.language, vad_filter=True)
    text = "".join([seg.text for seg in segments])
    return {"text": text.strip(), "lang": info.language}
