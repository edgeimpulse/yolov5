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
    --model-size) # e.g. one of: n, s, m, l
      MODEL_SIZE="$2"
      shift # past argument
      shift # past value
      ;;
    --batch-size) # e.g. 16
      BATCH_SIZE="$2"
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
if [ -z "$MODEL_SIZE" ]; then
    echo "Missing --model-size"
    exit 1
fi
if [ -z "$BATCH_SIZE" ]; then
    BATCH_SIZE=16
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

# convert Edge Impulse dataset (in Numpy format, with JSON for labels into something YOLOv5 understands)
# and write out a specs file
python3 -u extract_dataset.py \
  --data-directory $DATA_DIRECTORY  \
  --out-directory /tmp/data \
  --model-size $MODEL_SIZE

cd /app/yolov5
# train:
#     --freeze 10 - freeze the bottom layers of the network
python3 -u train.py --img $IMAGE_SIZE \
    --freeze 10 \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --data /tmp/data/data.yaml \
    --weights /app/yolov5$MODEL_SIZE.pt \
    --name yolov5_results \
    --cache
echo "Training complete"
echo ""

mkdir -p $OUT_DIRECTORY

# export as onnx
echo "Converting to ONNX..."
python3 -u export.py --weights ./runs/train/yolov5_results/weights/last.pt --img $IMAGE_SIZE --include onnx
cp runs/train/yolov5_results/weights/last.onnx $OUT_DIRECTORY/model.onnx
echo "Converting to ONNX OK"
echo ""


# export as f32
echo "Converting to TensorFlow Lite model (fp16)..."
python3 -u export.py --weights ./runs/train/yolov5_results/weights/last.pt --img $IMAGE_SIZE --include saved_model tflite --keras
cp runs/train/yolov5_results/weights/last-fp16.tflite $OUT_DIRECTORY/model.tflite
# ZIP up and copy the saved model too
cd runs/train/yolov5_results/weights/last_saved_model
zip -r -X ./saved_model.zip . > /dev/null
cp ./saved_model.zip $OUT_DIRECTORY/saved_model.zip
cd /app/yolov5
echo "Converting to TensorFlow Lite model (fp16) OK"
echo ""

# export as i8 (skipping for now as it outputs a uint8 input, not an int8 - which the Studio won't handle)
echo "Converting to TensorFlow Lite model (int8)..."
python3 -u export.py --weights ./runs/train/yolov5_results/weights/last.pt --data /tmp/data/data.yaml --img $IMAGE_SIZE --include tflite --int8
cp runs/train/yolov5_results/weights/last-int8.tflite $OUT_DIRECTORY/model_quantized_int8_io.tflite
echo "Converting to TensorFlow Lite model (int8) OK"
echo ""
