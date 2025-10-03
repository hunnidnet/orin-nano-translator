# Jetson-friendly ML base with CUDA/TensorRT + PyTorch preinstalled
FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Python deps:
# - openai-whisper provides tokenizer/audio utils whisper_trt imports
# - whisper_trt builds/uses TensorRT engines (no PyTorch at runtime)
# - FastAPI/uvicorn for a tiny HTTP shim
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
      openai-whisper==20231117 \
      whisper_trt \
      fastapi==0.111.0 \
      uvicorn==0.30.0 \
      numpy==1.26.4 \
      soundfile==0.12.1

# App
WORKDIR /srv
COPY server.py /srv/server.py

# Engine/model config (override via compose if you want)
ENV TRT_MODEL_SIZE=small \
    TRT_COMPUTE=int8 \
    SAMPLE_RATE=16000 \
    CACHE_DIR=/cache

EXPOSE 7001
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7001"]
