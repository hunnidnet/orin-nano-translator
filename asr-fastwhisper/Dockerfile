FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

# system deps (ffmpeg for audio decode)
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libsndfile1 python3-pip \
  && rm -rf /var/lib/apt/lists/*

# copy your locally built CUDA-enabled CTranslate2 wheel
# IMPORTANT: this wheel must be in the build context (project root)
COPY ctranslate2-4.6.0-cp310-cp310-linux_aarch64.whl /tmp/ctranslate2-4.6.0-cp310-cp310-linux_aarch64.whl

# install in one shot (no uninstall/reinstall needed)
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir /tmp/ctranslate2-4.6.0-cp310-cp310-linux_aarch64.whl && \
    python3 -m pip install --no-cache-dir \
        faster-whisper==1.0.0 \
        fastapi==0.111.0 \
        uvicorn==0.30.0 \
        soundfile==0.12.1

WORKDIR /srv
COPY server.py /srv/server.py

# environment (small model, 16 kHz, GPU fp16)
ENV FW_MODEL=small \
    FW_DEVICE=cuda \
    FW_COMPUTE_TYPE=float16 \
    SAMPLE_RATE=16000

EXPOSE 7001
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7001"]
