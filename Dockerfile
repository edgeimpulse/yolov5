# syntax = docker/dockerfile:experimental
FROM ubuntu:20.04
WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y curl zip mercurial xxd lsb-release software-properties-common apt-transport-https vim wget git protobuf-compiler python3 python3-pip libssl-dev rustc libhdf5-dev llvm-9
RUN python3 -m pip install --upgrade pip==20.3.4

RUN apt update && apt install -y ffmpeg libsm6 libxext6 libgl1

RUN git clone https://github.com/ultralytics/yolov5 && \
    cd yolov5 && \
    git checkout 436ffc417ac2312de18287ddc4f87bdc2f7f5734
RUN cd yolov5 && pip3 install -r requirements.txt

# Install TensorFlow
COPY install_tensorflow.sh install_tensorflow.sh
RUN /bin/bash install_tensorflow.sh && \
    rm install_tensorflow.sh
# Apparently needed for TFLite export
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

# grab chronic (yes, a bit late in the Dockerfile, but don't want to rebuild full container ;-) )
RUN apt update && apt install -y moreutils

# Patch up torch to disable cuda warnings
RUN sed -i -e "s/warnings.warn/\# warnings.warn/" /usr/local/lib/python3.8/dist-packages/torch/autocast_mode.py

COPY . ./

ENTRYPOINT ["/bin/bash", "run.sh"]
