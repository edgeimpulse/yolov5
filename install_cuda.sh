#!/bin/bash
set -e

UNAME=`uname -m`

if [ "$UNAME" == "aarch64" ]; then
    echo "On aarch64, skipping cuda..."
else
    apt update && apt install -y nvidia-cuda-toolkit
fi
