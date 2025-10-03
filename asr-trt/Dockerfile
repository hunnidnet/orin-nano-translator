FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

# OS deps: git for pulling whisper_trt, ffmpeg/libsndfile for audio IO
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsndfile1 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Python base deps
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
      openai-whisper==20231117 \
      fastapi==0.111.0 \
      uvicorn==0.30.0 \
      numpy==1.26.4 \
      soundfile==0.12.1

# --- IMPORTANT: install torch2trt before whisper_trt ---
# Try PyPI first (works on recent JP6.x); if it fails, fall back to source
RUN (python3 -m pip install --no-cache-dir torch2trt==0.5.0) || \
    (git clone --depth 1 https://github.com/NVIDIA-AI-IOT/torch2trt /tmp/torch2trt && \
     cd /tmp/torch2trt && python3 setup.py install && cd / && rm -rf /tmp/torch2trt)

# Now install whisper_trt from GitHub (not on PyPI)
RUN python3 -m pip install --no-cache-dir \
      "git+https://github.com/NVIDIA-AI-IOT/whisper_trt.git@main"

# App
WORKDIR /srv
COPY server.py /srv/server.py

# Runtime config (override in compose if needed)
ENV TRT_MODEL_SIZE=small \
    TRT_COMPUTE=int8 \
    SAMPLE_RATE=16000 \
    CACHE_DIR=/cache

EXPOSE 7001
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7001"]
