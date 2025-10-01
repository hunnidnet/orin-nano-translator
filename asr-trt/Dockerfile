# Jetson-friendly base (CUDA/cuDNN/TRT come from JetPack host)
FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev build-essential libsndfile1 ffmpeg sox \
 && rm -rf /var/lib/apt/lists/*

# Python deps for the shim server
RUN python3 -m pip install --no-cache-dir \
    fastapi==0.111.0 uvicorn==0.30.0 numpy==1.26.4 soundfile==0.12.1

# --- whisper_trt itself ---
# You already cloned it earlier; if not, mount it or pip install your fork.
# Here we assume you will mount the engines and the repo into the container.
# If your repo has a setup.py/pyproject, you can pip install it instead.
python3 -m pip install /opt/whisper_trt

WORKDIR /srv
COPY server_trt_shim.py /srv/server.py

# Expose the shim API on 7001
EXPOSE 7001
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7001"]
