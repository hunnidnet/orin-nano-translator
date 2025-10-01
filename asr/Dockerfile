# Jetson-friendly base with CUDA and Python
FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

# System deps (soundfile needs libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 sox ffmpeg python3-pip python3-setuptools \
 && rm -rf /var/lib/apt/lists/*

# Python deps
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir \
      faster-whisper==1.0.3 \
      fastapi==0.111.0 \
      uvicorn==0.30.0 \
      numpy==1.26.4 \
      soundfile==0.12.1

WORKDIR /srv
COPY server.py /srv/server.py

# Tunables via env
ENV FW_MODEL=small \
    FW_DEVICE=cuda \
    FW_COMPUTE_TYPE=float16 \
    SAMPLE_RATE=16000

EXPOSE 7001
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7001", "--workers", "1"]
