#!/bin/bash
set -e

PACKAGE=tensorflow-addons
VERSION=0.15.0
UNAME=`uname -m`
TARGET_DIR=/app/wheels/$PACKAGE/$VERSION/$UNAME/

if compgen -G "$TARGET_DIR/*.whl" > /dev/null; then
    echo "Already has $PACKAGE $VERSION for $UNAME, skipping download"
else
    mkdir -p $TARGET_DIR

    if [ "$UNAME" == "aarch64" ]; then
        wget -P $TARGET_DIR/ https://cdn.edgeimpulse.com/build-system/wheels/aarch64/tensorflow_addons-0.15.0-cp38-cp38-linux_aarch64.whl
    else
        cd $TARGET_DIR
        pip3 download tensorflow-addons==0.15.0 --no-deps
    fi
fi

cd $TARGET_DIR
pip3 install *.whl
