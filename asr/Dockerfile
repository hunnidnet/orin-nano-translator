# JetPack r36.x base with CUDA/cuDNN & Python
FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3
# If your rootfs is 36.4.x, you can use r36.4.0-py3

ENV PYTHONUNBUFFERED=1 \
    CTRANSLATE2_NUM_THREADS=2 \
    ASR_MODEL=small \
    ASR_DEVICE=cuda \
    ASR_COMPUTE_TYPE=int8_float16

RUN apt-get update && apt-get install -y --no-install-recommends \
      libsndfile1 ffmpeg sox python3-pip python3-setuptools python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir \
      faster-whisper==1.0.3 uvicorn==0.30.0 fastapi==0.111.0 soundfile==0.12.1 numpy==1.26.4

WORKDIR /srv
COPY asr_server.py /srv/asr_server.py
EXPOSE 7010
CMD ["uvicorn", "asr_server:app", "--host", "0.0.0.0", "--port", "7010", "--workers", "1"]
