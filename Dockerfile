# syntax = docker/dockerfile:experimental
ARG UBUNTU_VERSION=20.04

ARG ARCH=
ARG CUDA=11.2
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}.1-base-ubuntu${UBUNTU_VERSION} as base
ARG CUDA
ARG CUDNN=8.1.0.77-1
ARG CUDNN_MAJOR_VERSION=8
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=8.0.0-1
ARG LIBNVINFER_MAJOR_VERSION=8
# Let us install tzdata painlessly
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Avoid confirmation dialogs
ENV DEBIAN_FRONTEND=noninteractive
# Makes Poetry behave more like npm, with deps installed inside a .venv folder
# See https://python-poetry.org/docs/configuration/#virtualenvsin-project
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

# CUDA drivers
SHELL ["/bin/bash", "-c"]
COPY ./dependencies/install_cuda.sh ./install_cuda.sh
RUN ./install_cuda.sh && \
    rm install_cuda.sh

# System dependencies
RUN apt update && apt install -y wget git python3 python3-pip zip protobuf-compiler vim

# Install CMake (required for onnx 1.8.1)
COPY dependencies/install_cmake.sh install_cmake.sh
RUN /bin/bash install_cmake.sh && \
    rm install_cmake.sh

# YOLOv5 (v5.0-with-freeze branch)
RUN git clone https://github.com/edgeimpulse/yolov5-training yolov5 && \
    cd yolov5 && \
    git checkout f8f31f6956c0feea98239ab323c2759f6eb5f282
RUN cd yolov5 && pip3 install -r requirements.txt

# Install TensorFlow
COPY dependencies/install_tensorflow.sh install_tensorflow.sh
RUN /bin/bash install_tensorflow.sh && \
    rm install_tensorflow.sh

# Install TensorFlow addons
COPY dependencies/install_tensorflow_addons.sh install_tensorflow_addons.sh
RUN --mount=type=cache,target=/root/.cache/pip --mount=type=cache,target=/app/wheels \
    /bin/bash install_tensorflow_addons.sh && \
    rm install_tensorflow_addons.sh

# Install onnx-tensorflow
RUN git clone https://github.com/onnx/onnx-tensorflow.git && \
    cd onnx-tensorflow && \
    git checkout 3f87e6235c96f2f66e523d95dc35ff4802862231 && \
    pip3 install -e .

# Local dependencies
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

COPY requirements-nvidia.txt ./
RUN pip3 install -r requirements-nvidia.txt

# Grab yolov5s.pt pretrained weights
RUN wget -O yolov5s.pt https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt

# Download some files that are pulled in, so we can run w/o network access
RUN mkdir -p /root/.config/Ultralytics/ && wget -O /root/.config/Ultralytics/Arial.ttf https://ultralytics.com/assets/Arial.ttf

WORKDIR /scripts

# Copy the normal files (e.g. run.sh and the extract_dataset scripts in)
COPY . ./

ENTRYPOINT ["/bin/bash", "run.sh"]
