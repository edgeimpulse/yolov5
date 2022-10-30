#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd $SCRIPTPATH

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --epochs) # e.g. 50
      EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
    --learning-rate) # e.g. 0.01
      LEARNING_RATE="$2"
      shift # past argument
      shift # past value
      ;;
    --data-directory) # e.g. 0.2
      DATA_DIRECTORY="$2"
      shift # past argument
      shift # past value
      ;;
    --out-directory) # e.g. (96,96,3)
      OUT_DIRECTORY="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

if [ -z "$EPOCHS" ]; then
    echo "Missing --epochs"
    exit 1
fi
if [ -z "$LEARNING_RATE" ]; then
    echo "Missing --learning-rate"
    exit 1
fi
if [ -z "$DATA_DIRECTORY" ]; then
    echo "Missing --data-directory"
    exit 1
fi
if [ -z "$OUT_DIRECTORY" ]; then
    echo "Missing --out-directory"
    exit 1
fi

OUT_DIRECTORY=$(realpath $OUT_DIRECTORY)
DATA_DIRECTORY=$(realpath $DATA_DIRECTORY)

IMAGE_SIZE=$(python3 get_image_size.py --data-directory "$DATA_DIRECTORY")

# surpress OpenBLAS warnings
export OMP_NUM_THREADS=1

# convert Edge Impulse dataset (in Numpy format, with JSON for labels into something YOLOv5 understands)
python3 -u extract_dataset.py --data-directory $DATA_DIRECTORY --out-directory /tmp/data

cd /app/yolov5
rm -rf ./runs/train/yolov5_results/

# train:
#     --freeze 10 - freeze the bottom layers of the network
#     --workers 0 - as this otherwise requires a larger /dev/shm than we have on Edge Impulse prod,
#                   there's probably a workaround for this, but we need to check with infra.
python3 -u train.py --img $IMAGE_SIZE \
    --freeze 10 \
    --epochs $EPOCHS \
    --data /tmp/data/data.yaml \
    --weights /app/yolov5s.pt \
    --name yolov5_results \
    --cache \
    --workers 0
echo "Training complete"
echo ""

mkdir -p $OUT_DIRECTORY

# export as onnx
echo "Converting to ONNX..."
python3 -u models/export.py  --weights ./runs/train/yolov5_results/weights/last.pt --img-size $IMAGE_SIZE --batch-size 1 --grid
# add shape info
python3 /scripts/add_shape_info.py --onnx-file $OUT_DIRECTORY/model.onnx
echo "Converting to ONNX OK"
echo ""

# export as f32
echo "Converting to TensorFlow Lite model (fp16)..."
cp $OUT_DIRECTORY/model.onnx /tmp/model.onnx
# strip off all but one output layers
snd4onnx -rn 397 672 947 -if /tmp/model.onnx -of /tmp/model.onnx --non_verbose
# Convert to NHWC
python3 /scripts/convert-to-nhwc.py --onnx-file /tmp/model.onnx --out-file /tmp/model.onnx
# Convert to TFLite
python3 -u /scripts/onnx_to_tflite.py --onnx-file /tmp/model.onnx --out-file $OUT_DIRECTORY/model.tflite
echo "Converting to TensorFlow Lite model (fp16) OK"
echo ""
