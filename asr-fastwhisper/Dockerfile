FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

# system deps (ffmpeg for audio decode)
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libsndfile1 python3-pip \
  && rm -rf /var/lib/apt/lists/*

# copy your locally built CUDA-enabled CTranslate2 wheel
# IMPORTANT: this wheel must be in the build context (project root)
COPY ctranslate2-4.6.0-cp310-cp310-linux_aarch64.whl /tmp/ctranslate2-4.6.0-cp310-cp310-linux_aarch64.whl

# install in one shot (no uninstall/reinstall needed)
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir /tmp/ctranslate2-4.6.0-cp310-cp310-linux_aarch64.whl && \
    python3 -m pip install --no-cache-dir \
        faster-whisper==1.0.0 \
        fastapi==0.111.0 \
        uvicorn==0.30.0 \
        soundfile==0.12.1
        
# Make the CTranslate2 shared library discoverable by the dynamic linker
# We detect the actual site-packages path and register CT2's libs/ with ldconfig
RUN set -eux; \
  CT2_LIBDIR="$(python3 - <<'PY'
import ctranslate2, pathlib
print(str(pathlib.Path(ctranslate2.__file__).parent / "libs"))
PY
  )"; \
  echo "CT2 libs at: ${CT2_LIBDIR}"; \
  echo "${CT2_LIBDIR}" > /etc/ld.so.conf.d/ctranslate2.conf; \
  ldconfig; \
  # quick sanity: list and try to resolve dependencies (non-fatal if ldd not perfect)
  ls -l "${CT2_LIBDIR}"; \
  ldd "${CT2_LIBDIR}"/libctranslate2.so* || true
    
WORKDIR /srv
COPY server.py /srv/server.py

# environment (small model, 16 kHz, GPU fp16)
ENV FW_MODEL=small \
    FW_DEVICE=cuda \
    FW_COMPUTE_TYPE=float16 \
    SAMPLE_RATE=16000

EXPOSE 7001
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7001"]
