FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libasound2-dev alsa-utils ffmpeg build-essential \
 && rm -rf /var/lib/apt/lists/*

# pyalsaaudio -> ALSA capture/playback by "hw:card,device"
RUN pip install --no-cache-dir numpy webrtcvad requests pyalsaaudio pyyaml grpcio

# Riva client (pick the one you use; nvidia may provide riva-client wheel)
# Try both names—if one fails you can install it on host and mount; or pip install from wheel.
RUN pip install --no-cache-dir nvidia-riva-client || true
RUN pip install --no-cache-dir riva-client || true

WORKDIR /app
COPY app.py .
CMD ["python", "app.py"]
