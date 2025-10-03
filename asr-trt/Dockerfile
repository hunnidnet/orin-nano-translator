FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

# System deps: TRT python + dev + DLA compiler are REQUIRED so torch2trt can import tensorrt
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsndfile1 python3-pip \
    python3-libnvinfer python3-libnvinfer-dev \
    libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers-dev \
    nvdla-compiler \
 && rm -rf /var/lib/apt/lists/*

# Base Python deps
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
      openai-whisper==20231117 \
      fastapi==0.111.0 \
      uvicorn==0.30.0 \
      numpy==1.26.4 \
      soundfile==0.12.1

# Install torch2trt from source (PyPI wheels are not published for JP6)
RUN git clone --depth 1 https://github.com/NVIDIA-AI-IOT/torch2trt /tmp/torch2trt && \
    cd /tmp/torch2trt && \
    python3 setup.py install && \
    cd / && rm -rf /tmp/torch2trt

# Now whisper_trt (it imports torch2trt in its __init__)
RUN python3 -m pip install --no-cache-dir \
      "git+https://github.com/NVIDIA-AI-IOT/whisper_trt.git@main"

# App
WORKDIR /srv
COPY server.py /srv/server.py

# Runtime config (override via compose)
ENV TRT_MODEL_SIZE=small \
    TRT_COMPUTE=int8 \
    SAMPLE_RATE=16000 \
    CACHE_DIR=/cache

EXPOSE 7001
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7001"]
