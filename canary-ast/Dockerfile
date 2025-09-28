FROM nvcr.io/nvidia/pytorch:24.05-py3

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Server deps
RUN pip install --no-cache-dir fastapi uvicorn[standard] numpy soundfile webrtcvad pydantic requests

# ASR/AST deps (Canary path uses HF transformers; we include a fallback to Whisper to smoke test)
RUN pip install --no-cache-dir torch torchaudio transformers accelerate librosa

WORKDIR /srv
COPY server.py .
EXPOSE 7000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7000"]
