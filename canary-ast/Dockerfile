# canary-ast/Dockerfile
# Jetson-compatible PyTorch image (matches JetPack r36.x)
FROM nvcr.io/nvidia/l4t-ml:r36.0.0-py3

# Avoid interactive tzdata prompts
ENV DEBIAN_FRONTEND=noninteractive
# Disable torchvision use inside transformers (must be set BEFORE python starts)
ENV TRANSFORMERS_NO_TORCHVISION=1

# System deps for audio I/O and ffmpeg (for whisper-like pipelines)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg python3-pip python3-dev build-essential portaudio19-dev \
 && rm -rf /var/lib/apt/lists/*

# Python deps — keep lightweight; avoid installing torchvision
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip uninstall -y torchvision || true \
 && python3 -m pip install --no-cache-dir \
      "transformers[audio]==4.42.4" \
      "accelerate==0.33.0" \
      "fastapi==0.110.0" "uvicorn[standard]==0.29.0" \
      "soundfile==0.12.1" "webrtcvad==2.0.10" \
      "numpy==1.26.4" \
      "faster-whisper==1.0.3"

# App
WORKDIR /srv
COPY server.py /srv/server.py

EXPOSE 7000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7000"]
