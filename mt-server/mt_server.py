from fastapi import FastAPI
from pydantic import BaseModel
from mt_runtime import MarianCT2

app = FastAPI()
mt = MarianCT2(
    en2es_dir="/models/ct2-en-es",
    es2en_dir="/models/ct2-es-en",
    en2es_sp_dir="/spm/opus-mt-en-es",
    es2en_sp_dir="/spm/opus-mt-es-en",
)

class MTIn(BaseModel):
    text: str
    source_lang: str
    target_lang: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/mt")
def mt_endpoint(req: MTIn):
    out = mt.translate(req.text, req.source_lang, req.target_lang)
    return {"text": out}
