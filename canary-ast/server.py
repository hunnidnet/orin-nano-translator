import os, io
import numpy as np
import soundfile as sf
from fastapi import FastAPI
from pydantic import BaseModel
from nemo.collections.asr.models import EncDecMultiTaskModel

MODEL_ID = os.getenv("CANARY_MODEL_ID", "nvidia/canary-1b-flash")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))

app = FastAPI()
canary = None

@app.on_event("startup")
def _startup():
    global canary
    # Load NeMo Canary 1B Flash (supports en/es/de/fr; ASR & AST)
    canary = EncDecMultiTaskModel.from_pretrained(MODEL_ID)
    # Greedy decoding for lowest latency
    dec_cfg = canary.cfg.decoding
    dec_cfg.beam.beam_size = 1
    canary.change_decoding_strategy(dec_cfg)

class ASTIn(BaseModel):
    pcm16_hex: str
    sr: int = 16000
    src_lang: str = "es"  # 'en','es','de','fr'
    tgt_lang: str = "en"
    pnc: str = "yes"      # punctuation/casing: "yes" or "no"

@app.post("/ast")
def ast(req: ASTIn):
    # bytes -> float32 mono
    pcm = np.frombuffer(bytes.fromhex(req.pcm16_hex), dtype=np.int16).astype(np.float32) / 32768.0
    if req.sr != SAMPLE_RATE and len(pcm) > 0:
        # quick resample via round-trip WAV in-memory
        with io.BytesIO() as b:
            sf.write(b, pcm, req.sr, format="WAV", subtype="PCM_16")
            b.seek(0)
            data, sr = sf.read(b)
        with io.BytesIO() as b2:
            sf.write(b2, data, SAMPLE_RATE, format="WAV", subtype="PCM_16")
            b2.seek(0)
            pcm, _ = sf.read(b2)

    # write tiny temp wav (NeMo transcribe prefers file paths)
    tmp = "/tmp/snippet.wav"
    with open(tmp, "wb") as f:
        sf.write(f, pcm, SAMPLE_RATE, format="WAV", subtype="PCM_16")

    src = req.src_lang.lower()
    tgt = req.tgt_lang.lower()
    pnc = req.pnc.lower() if req.pnc.lower() in ("yes", "no") else "yes"

    # ASR if src==tgt; AST if src!=tgt
    out = canary.transcribe(
        [tmp],
        batch_size=1,
        pnc=pnc,
        timestamps="no",
        source_lang=src,
        target_lang=tgt,
    )

    text = out[0].text if out and hasattr(out[0], "text") else ""
    if src != tgt:
        return {"transcript": "", "translation": text}
    else:
        return {"transcript": text, "translation": ""}

@app.get("/health")
def health():
    return {"status": "ok", "loaded": canary is not None, "model": MODEL_ID}
