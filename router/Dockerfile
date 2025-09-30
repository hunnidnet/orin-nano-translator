FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      libasound2-dev alsa-utils && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir requests webrtcvad pyalsaaudio numpy

WORKDIR /app
COPY app.py /app/app.py
CMD ["python", "/app/app.py"]
