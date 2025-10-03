Orin Nano Translator

A Jetson Orin Nano–optimized speech-to-speech translation system.
Pipeline:
	1.	ASR (Automatic Speech Recognition) → NVIDIA Riva (English + Spanish, streaming ASR)
	2.	MT (Machine Translation) → Local service powered by CTranslate2
	3.	TTS (Text-to-Speech) → NVIDIA Riva (multilingual voices, English + Spanish)

Audio is captured from ALSA devices, passed through VAD, then transcribed → translated → synthesized in real time.

⸻

Requirements
	•	Jetson Orin Nano with JetPack 6 / L4T R36.2+
	•	Docker with NVIDIA runtime enabled
	•	NVIDIA Riva Quickstart for Jetson (riva_quickstart_arm64_vX.Y.Z)
	•	Built ctranslate2 wheel for aarch64 (for the MT server)
	•	ALSA configured with your capture/playback devices (.asoundrc)

⸻

Setup

1. NVIDIA Riva (ASR + TTS)
	1.	Download Riva Quickstart for ARM64 and extract:

tar -xzf riva_quickstart_arm64_v2.19.0.tar.gz
cd riva_quickstart_arm64_v2.19.0


	2.	Edit config.sh to enable ASR + TTS in English and Spanish:

service_enabled_asr=true
service_enabled_tts=true
service_enabled_nlp=false
service_enabled_nmt=false

asr_acoustic_model=("conformer")
asr_language_code=("en-US" "es-US")

tts_model=("fastpitch_hifigan")
tts_language_code=("en-US" "es-US")


	3.	Initialize and start Riva:

bash riva_init.sh
bash riva_start.sh

The server will expose gRPC on 127.0.0.1:50051.

	4.	Verify ASR:

docker exec riva-speech /opt/riva/clients/riva_streaming_asr_client \
  --riva_uri=localhost:50051 \
  --audio_file=/opt/riva/wav/en-US_sample.wav \
  --language_code=en-US



⸻

2. Machine Translation (CTranslate2)

We use MarianMT/Opus-MT models converted to CTranslate2 for efficient GPU inference.

Install CTranslate2 (Jetson wheel)

cd CTranslate2/python
pip wheel . -w dist
pip install dist/ctranslate2-*.whl

Download & Convert Models

Use Hugging Face MarianMT models. Examples:
	•	English → Spanish
	•	Spanish → English

# Install converters
pip install ctranslate2 transformers sentencepiece

# Create a models folder
mkdir -p models && cd models

# Download Hugging Face MarianMT EN→ES
python3 -m ctranslate2.converters.transformers \
  --model Helsinki-NLP/opus-mt-en-es \
  --output-dir en-es-ctranslate2 \
  --quantization float16

# Download Hugging Face MarianMT ES→EN
python3 -m ctranslate2.converters.transformers \
  --model Helsinki-NLP/opus-mt-es-en \
  --output-dir es-en-ctranslate2 \
  --quantization float16

You now have two optimized CTranslate2 model directories:

models/en-es-ctranslate2/
models/es-en-ctranslate2/

Minimal MT Server

mt_server.py:

from fastapi import FastAPI
import ctranslate2, sentencepiece as spm

app = FastAPI()

# Load models + tokenizers
sp_en = spm.SentencePieceProcessor(model_file="Helsinki-NLP/opus-mt-en-es/source.spm")
sp_es = spm.SentencePieceProcessor(model_file="Helsinki-NLP/opus-mt-es-en/source.spm")

translator_en_es = ctranslate2.Translator("models/en-es-ctranslate2", device="cuda", compute_type="float16")
translator_es_en = ctranslate2.Translator("models/es-en-ctranslate2", device="cuda", compute_type="float16")

@app.post("/translate")
def translate(payload: dict):
    src = payload["text"]
    src_lang = payload["source"]
    tgt_lang = payload["target"]

    if src_lang.startswith("en"):
        tokens = sp_en.encode(src, out_type=str)
        result = translator_en_es.translate_batch([tokens])
        out = sp_en.decode(result[0].hypotheses[0])
    else:
        tokens = sp_es.encode(src, out_type=str)
        result = translator_es_en.translate_batch([tokens])
        out = sp_es.decode(result[0].hypotheses[0])

    return {"translation": out}

Run:

uvicorn mt_server:app --host 0.0.0.0 --port 7010


⸻

3. Router Service

The router orchestrates:
	•	ASR: Riva gRPC service
	•	MT: local CTranslate2 FastAPI server
	•	TTS: Riva gRPC service

docker compose build router
docker compose up -d router
docker logs -f translator-router


⸻

docker-compose.yml

services:
  router:
    build: ./router
    container_name: translator-router
    network_mode: "host"
    devices:
      - "/dev/snd:/dev/snd"
    privileged: true
    volumes:
      - "${HOME}/.asoundrc:/root/.asoundrc:ro"
    environment:
      - RIVA_ADDR=127.0.0.1:50051
      - MT_URL=http://127.0.0.1:7010

      - A_IN=plughw:0,0
      - A_OUT=plughw:0,0
      - B_IN=plughw:1,0
      - B_OUT=plughw:1,0

      - A_SRC=es
      - A_TGT=en
      - B_SRC=en
      - B_TGT=es

      - VOICE_EN=English-US.Female-1
      - VOICE_ES=Spanish-US.Female-1

      - SAMPLE_RATE=16000
      - FRAME_MS=20
      - BURST_MIN_MS=300
      - BURST_MAX_MS=600
      - VAD_LEVEL=2


⸻

Usage
	•	Plug in two microphones/speakers (map via .asoundrc).
	•	Start Riva, MT, and the router.
	•	Speak Spanish into side A → hear English TTS on side B.
	•	Speak English into side B → hear Spanish TTS on side A.

⸻

Notes
	•	Riva streaming ASR is optimized for Jetson; offline ASR models are not available on Tegra.
	•	CTranslate2 + MarianMT float16 runs entirely on Jetson GPU.
	•	For additional language pairs, replace Hugging Face models in step 2.

⸻

Do you want me to also add prebuilt Dockerfiles for the MT server, so you don’t need to run uvicorn manually and it integrates cleanly into docker-compose?
