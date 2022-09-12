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
COPY ./install_cuda.sh ./install_cuda.sh
RUN ./install_cuda.sh && \
    rm install_cuda.sh

# System dependencies
RUN apt update && apt install -y wget git python3 python3-pip

RUN git clone https://github.com/ultralytics/yolov5 && \
    cd yolov5 && \
    git checkout 23701ea
RUN cd yolov5 && pip3 install -r requirements.txt

# Install TensorFlow
COPY install_tensorflow.sh install_tensorflow.sh
RUN /bin/bash install_tensorflow.sh && \
    rm install_tensorflow.sh

# Local dependencies
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

# Patch up torch to disable cuda warnings
RUN sed -i -e "s/warnings.warn/\# warnings.warn/" /usr/local/lib/python3.8/dist-packages/torch/amp/autocast_mode.py && \
    sed -i -e "s/warnings.warn/\# warnings.warn/" /usr/local/lib/python3.8/dist-packages/torch/cpu/amp/autocast_mode.py && \
    sed -i -e "s/warnings.warn/\# warnings.warn/" /usr/local/lib/python3.8/dist-packages/torch/cuda/amp/autocast_mode.py

# Grab yolov5n.pt pretrained weights
RUN wget -O yolov5n.pt https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt

# Download some files that are pulled in, so we can run w/o network access
RUN mkdir -p /root/.config/Ultralytics/ && wget -O /root/.config/Ultralytics/Arial.ttf https://ultralytics.com/assets/Arial.ttf

WORKDIR /scripts

# Copy the normal files (e.g. run.sh and the extract_dataset scripts in)
COPY . ./

ENTRYPOINT ["/bin/bash", "run.sh"]
