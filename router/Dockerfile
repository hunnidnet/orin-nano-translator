FROM python:3.11-slim
# Build deps for native wheels (gcc/make), headers for ALSA and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc python3-dev \
      libasound2-dev alsa-utils && \
    rm -rf /var/lib/apt/lists/*
# Install Python deps (numpy first speeds things up a bit)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy==2.3.3 requests webrtcvad pyalsaaudio

WORKDIR /app
COPY app.py /app/app.py
CMD ["python", "/app/app.py"]
