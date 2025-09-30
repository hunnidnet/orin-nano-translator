# canary-ast/Dockerfile  (JetPack r36.x)
FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf \
    NEMO_LOG_LEVEL=WARNING

RUN apt-get update && apt-get install -y --no-install-recommends \
      libsndfile1 ffmpeg sox python3-pip python3-setuptools python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Important: use NeMo >= 2.3 for canary2 prompt format
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir \
      nemo_toolkit[asr]==2.3.2 \
      soundfile==0.12.1 \
      librosa==0.10.2.post1 \
      numba==0.59.1 \
      numpy==1.26.4 \
      fastapi==0.111.0 \
      uvicorn==0.30.0 \
      requests==2.32.3 \
 && python3 -m pip uninstall -y torchvision || true

WORKDIR /srv
COPY canary-ast/server.py /srv/server.py
EXPOSE 7000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7000", "--workers", "1"]
