# Jetson-friendly base
FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 sox ffmpeg python3-pip python3-setuptools && \
    rm -rf /var/lib/apt/lists/*

# Faster-Whisper + API
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir \
    faster-whisper==1.0.3 \
    fastapi==0.111.0 \
    uvicorn==0.30.0 \
    numpy==1.26.4 \
    soundfile==0.12.1

WORKDIR /srv
COPY asr_server.py /srv/asr_server.py

EXPOSE 7001
CMD ["uvicorn", "asr_server:app", "--host", "0.0.0.0", "--port", "7001"]
