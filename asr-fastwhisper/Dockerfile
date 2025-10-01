FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libsndfile1 build-essential python3-dev && \
    rm -rf /var/lib/apt/lists/*
# torch is in l4t-ml; install the server deps
RUN pip install --no-cache-dir fastapi uvicorn faster-whisper==1.0.0
WORKDIR /srv
COPY server.py /srv/server.py
EXPOSE 7001
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7001"]
