import os, io, time
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Body
from pydantic import BaseModel
import torch
import json

app = FastAPI()

MODEL_ID = os.getenv("CANARY_MODEL_ID")  # e.g., "nvidia/canary-1b-v2"
asr_translate = None
fw_model = None  # faster-whisper fallback

def load_model():
    global asr_translate, fw_model
    if MODEL_ID:
        # Canary/HF pipeline (no torchvision)
        from transformers import pipeline
        asr_translate = pipeline(
            task="automatic-speech-recognition",
            model=MODEL_ID,
            torch_dtype="float32",  # Jetson-friendly; can change to "float16" later
        )
        return

    # Fallback: faster-whisper translate (no transformers / no torchvision)
    from faster_whisper import WhisperModel
    # small-medium balances speed/accuracy; adjust per your taste
    fw_model = WhisperModel("small", device="cpu", compute_type="int8")
    # or try: WhisperModel("medium", device="cuda", compute_type="float16") once CUDA works

@app.on_event("startup")
def _startup():
    load_model()

class AstIn(BaseModel):
    pcm16_hex: str
    sr: int
    src_lang: str
    tgt_lang: str

@app.post("/ast")
def ast(inb: AstIn):
    import numpy as np
    pcm = bytes.fromhex(inb.pcm16_hex) if inb.pcm16_hex else b""
    if not pcm:
        return {"translation": ""}

    # int16 mono -> float32
    import soundfile as sf
    import io
    # Write a fake WAV header in-memory so libs accept it easily
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(inb.sr)
        w.writeframes(pcm)
    buf.seek(0)

    if asr_translate:
        # HF pipeline (e.g., Canary multilingual)
        out = asr_translate(buf, generate_kwargs={"task": "translate"})
        text = out.get("text", "")
        return {"translation": text}

    # faster-whisper fallback with task="translate"
    segments, info = fw_model.transcribe(buf, task="translate", language=None)  # auto-detect
    text = " ".join(seg.text.strip() for seg in segments)
    return {"translation": text}
