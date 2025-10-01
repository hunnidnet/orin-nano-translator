# Jetson (ARM64) ML base with CUDA/cuDNN from JetPack
FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

# ---------- System deps ----------
# - Build toolchain for CTranslate2 (cmake, ninja, gcc, pybind11, python dev)
# - Minimal audio libs for the server (libsndfile, ffmpeg)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git build-essential cmake ninja-build \
      python3-dev pybind11-dev \
      libsndfile1 ffmpeg sox \
    && rm -rf /var/lib/apt/lists/*

# ---------- Python base ----------
RUN python3 -m pip install --no-cache-dir --upgrade pip wheel setuptools

# ---------- Build & install CTranslate2 (CUDA) ----------
# This mirrors your host steps:
#   - WITH_CUDA=ON, WITH_CUDNN=ON (as in your guide)
#   - OPENMP_RUNTIME=NONE to avoid libiomp issues on Jetson
#   - CUDA arch 8.7 for Orin Nano
WORKDIR /opt
RUN git clone --recursive https://github.com/OpenNMT/CTranslate2.git && \
    cd CTranslate2 && \
    cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DWITH_CUDA=ON \
      -DCUDA_ARCH_LIST=8.7 \
      -DWITH_CUDNN=ON \
      -DWITH_DNNL=OFF \
      -DWITH_MKL=OFF \
      -DWITH_RUY=OFF \
      -DOPENMP_RUNTIME=NONE \
      -DBUILD_CLI=ON \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_INSTALL_PREFIX=/usr/local && \
    ninja -C build -j"$(nproc)" && \
    ninja -C build install && \
    ldconfig && \
    cd python && \
    python3 -m pip install --no-cache-dir -r install_requirements.txt && \
    python3 -m pip wheel . -w /tmp/wheels && \
    python3 -m pip install --no-cache-dir /tmp/wheels/ctranslate2-*.whl && \
    rm -rf /opt/CTranslate2 /tmp/wheels

# ---------- ASR server deps ----------
# Install the server stack after CT2 so faster-whisper links to the CUDA build
RUN python3 -m pip install --no-cache-dir \
      fastapi==0.111.0 \
      uvicorn==0.30.0 \
      faster-whisper==1.0.0 \
      soundfile==0.12.1

# ---------- App ----------
WORKDIR /srv
COPY server.py /srv/server.py

# Defaults (override via compose env:)
ENV FW_MODEL=small \
    FW_DEVICE=cuda \
    FW_COMPUTE_TYPE=float16 \
    SAMPLE_RATE=16000

EXPOSE 7001
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7001"]
