FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# torch/torchvision/datasets/huggingface_hub cover the "torchvision" and
# "huggingface" registry sources from arpa/tools/dataset_tools.py.
#
# tensorflow/tensorflow-datasets cover the "tfds" registry source. This
# matters for real papers in this project's own benchmark set: EMNIST
# (easy2) and CUB-200 (a likely resolution for hard2/Bilinear CNN) aren't in
# TORCHVISION_CLASS_MAP, so DatasetResolver resolves them via TFDS instead,
# and the generated dataset_loader.py does `import tensorflow_datasets as
# tfds` -- which fails without this. tensorflow-cpu keeps the image smaller
# since the sandbox only ever needs data loading, never GPU training.
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchvision==0.17.2 \
    datasets==2.18.0 \
    huggingface_hub==0.21.4 \
    numpy==1.26.4 \
    pillow==10.2.0 \
    tensorflow-cpu==2.16.1 \
    tensorflow-datasets==4.9.4

WORKDIR /workspace
