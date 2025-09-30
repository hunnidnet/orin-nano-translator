FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-setuptools && \
    rm -rf /var/lib/apt/lists/*

# Install a recent CT2 wheel (CPU+CUDA). If you built from source on host,
# you can also COPY the wheel in instead and pip install it here.
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir \
    ctranslate2==4.6.0 \
    sentencepiece==0.1.99 \
    sacremoses==0.1.1 \
    fastapi==0.111.0 \
    uvicorn==0.30.0 \
    numpy==1.26.4

WORKDIR /srv
COPY mt_server.py /srv/mt_server.py
COPY mt_runtime.py /srv/mt_runtime.py

EXPOSE 7010
CMD ["uvicorn", "mt_server:app", "--host", "0.0.0.0", "--port", "7010"]
