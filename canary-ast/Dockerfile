FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

ENV DEBIAN_FRONTEND=noninteractive
# Minimal runtime deps (no torchvision/transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg sox python3-pip python3-setuptools python3-wheel \
 && rm -rf /var/lib/apt/lists/*

# Jetson-compatible pins
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
# We don't install transformers here; NeMo will handle Canary Flash.
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir \
    nemo_toolkit[asr]==2.0.0 \
    huggingface_hub==0.23.4 \
    soundfile==0.12.1 \
    librosa==0.10.2.post1 \
    numba==0.59.1 \
    numpy==1.26.4 \
    fastapi==0.111.0 \
    uvicorn==0.30.0 \
    requests==2.32.3

WORKDIR /srv
COPY canary-ast/server.py /srv/server.py

EXPOSE 7000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7000"]
