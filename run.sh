#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd $SCRIPTPATH

IMAGE_SIZE=$(python3 get_image_size.py --x-file /home/X_train_features_beer.npy)

python3 -u extract_dataset.py --x-file /home/X_train_features_beer.npy --y-file /home/y_train_beer.npy --out-directory /tmp/data

cd yolov5
# --freeze 23 to do transfer learning?
python3 -u train.py --img $IMAGE_SIZE --batch 16 --epochs 200 --freeze 24 --data /tmp/data/data.yaml --weights ../yolov5s.pt --name yolov5s_results --cache
python3 -u export.py --weights ./runs/train/yolov5s_results/weights/last.pt --img $IMAGE_SIZE --include tflite
python3 -u export.py --weights ./runs/train/yolov5s_results/weights/last.pt --img $IMAGE_SIZE --include tflite --int8
cp runs/train/yolov5s_results/weights/last-fp16.tflite /home/model.tflite
cp runs/train/yolov5s_results/weights/last-int8.tflite /home/model_quantized_int8_io.tflite
