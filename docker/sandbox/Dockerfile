FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchvision==0.17.2 \
    datasets==2.18.0 \
    huggingface_hub==0.21.4 \
    numpy==1.26.4 \
    pillow==10.2.0

WORKDIR /workspace
