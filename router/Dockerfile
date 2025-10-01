FROM python:3.11-slim

# Build deps for pyalsaaudio/webrtcvad and ALSA headers
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc python3-dev \
      libasound2-dev alsa-utils \
    && rm -rf /var/lib/apt/lists/*

# Pin setuptools for Riva client compatibility, then install deps
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
      "setuptools==70.0.0" \
      numpy==2.3.3 \
      requests \
      webrtcvad \
      pyalsaaudio \
      grpcio \
      nvidia-riva-client==2.19.0

WORKDIR /app
COPY app.py /app/app.py

CMD ["python", "/app/app.py"]
