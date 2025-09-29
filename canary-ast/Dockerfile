FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TRANSFORMERS_CACHE=/cache
ENV HF_HOME=/cache

RUN apt-get update && apt-get install -y \
    ffmpeg git \
 && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    transformers>=4.39.0 \
    accelerate \
    fastapi uvicorn \
    soundfile librosa \
    sentencepiece

# Create cache directory
RUN mkdir -p /cache && chmod 777 /cache

WORKDIR /app
COPY server.py /app/

EXPOSE 7000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7000"]
