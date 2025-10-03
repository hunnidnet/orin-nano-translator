FROM python:3.11-slim

# ALSA + build tools for webrtcvad/pyalsaaudio wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc python3-dev \
    libasound2-dev alsa-utils \
    && rm -rf /var/lib/apt/lists/*

# Python deps (include Riva client!)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      numpy==2.3.3 \
      requests \
      webrtcvad \
      pyalsaaudio \
      nvidia-riva-client==2.19.0

WORKDIR /app
COPY app.py /app/app.py

CMD ["python", "/app/app.py"]
