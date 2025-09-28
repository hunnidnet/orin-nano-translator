import os, io, time
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Body
from pydantic import BaseModel
import torch
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

# Try to load Canary AST; fallback to Whisper translate pipeline for first-run sanity.
USE_FALLBACK = False
MODEL_ID = os.getenv("CANARY_MODEL_ID", "").strip()

processor = None
model = None
pipe = None

app = FastAPI(title="Canary AST Server", version="0.1")

@app.on_event("startup")
def load_model():
    global processor, model, pipe, USE_FALLBACK
    try:
        if MODEL_ID:
            from transformers import AutoProcessor, AutoModelForSeq2SeqLM
            processor = AutoProcessor.from_pretrained(MODEL_ID)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16
            ).cuda().eval()
            print(f"Loaded Canary model: {MODEL_ID}")
            USE_FALLBACK = False
        else:
            raise RuntimeError("No CANARY_MODEL_ID set")
    except Exception as e:
        print(f"[WARN] Canary model not set/available: {e}")
        print("[INFO] Falling back to Whisper translate pipeline for smoke test.")
        from transformers import pipeline
        # This fallback translates speech to English; we will naïvely switch langs via 'task'
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device=0 if torch.cuda.is_available() else -1,
            generate_kwargs={"task": "translate"},
        )
        USE_FALLBACK = True

class AstRequest(BaseModel):
    # 16-bit little-endian PCM mono at 16k; send as hex string to avoid JSON b64 hassle
    pcm16_hex: str
    sr: int = 16000
    src_lang: str  # "es" or "en"
    tgt_lang: str  # "en" or "es"

@app.post("/ast")
def ast(req: AstRequest):
    # decode PCM
    pcm_bytes = bytes.fromhex(req.pcm16_hex)
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if len(audio) == 0:
        return {"translation": "", "timestamp": time.time()}

    if USE_FALLBACK:
        # Whisper translate always tends to English by default; for EN->ES we’ll first transcribe, then mark for client-side TTS in ES.
        # Note: This is only for smoke tests until you set CANARY_MODEL_ID.
        text = pipe({"raw": audio, "sampling_rate": req.sr})["text"]
        # crude: if src==en and tgt==es, we just return the EN text; downstream will TTS in ES (not true translation). Replace with real Canary ASAP.
        return {"translation": text, "timestamp": time.time(), "note": "fallback-whisper"}
    else:
        from transformers import AutoProcessor
        inputs = processor(audio, sampling_rate=req.sr, src_lang=req.src_lang, tgt_lang=req.tgt_lang, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=128)
        text = processor.batch_decode(out, skip_special_tokens=True)[0]
        return {"translation": text, "timestamp": time.time()}
