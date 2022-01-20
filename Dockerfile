# syntax = docker/dockerfile:experimental
FROM ubuntu:20.04
WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y curl zip mercurial xxd lsb-release software-properties-common apt-transport-https vim wget git protobuf-compiler python3 python3-pip libssl-dev rustc libhdf5-dev llvm-9
RUN python3 -m pip install --upgrade pip==20.3.4

RUN apt update && apt install -y ffmpeg libsm6 libxext6 libgl1 moreutils

# Install cuda
COPY install_cuda.sh install_cuda.sh
RUN /bin/bash install_cuda.sh && \
    rm install_cuda.sh

RUN git clone https://github.com/ultralytics/yolov5 && \
    cd yolov5 && \
    git checkout 436ffc417ac2312de18287ddc4f87bdc2f7f5734
RUN cd yolov5 && pip3 install -r requirements.txt

# Install TensorFlow
COPY install_tensorflow.sh install_tensorflow.sh
RUN /bin/bash install_tensorflow.sh && \
    rm install_tensorflow.sh
# Local dependencies
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

# Patch up torch to disable cuda warnings
RUN sed -i -e "s/warnings.warn/\# warnings.warn/" /usr/local/lib/python3.8/dist-packages/torch/autocast_mode.py

# Grab yolov5s6_384_ti_lite.pt pretrained weights
RUN wget -O yolov5s6_384_ti.pt http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_00_00_05/edgeai-yolov5/pretrained_models/models/yolov5s6_384_ti_lite/weights/best.pt

# Download some files that are pulled in, so we can run w/o network access
RUN mkdir -p /root/.config/Ultralytics/ && wget -O /root/.config/Ultralytics/Arial.ttf https://ultralytics.com/assets/Arial.ttf

# Copy the normal files (e.g. run.sh and the extract_dataset scripts in)
COPY . ./

ENTRYPOINT ["/bin/bash", "run.sh"]
