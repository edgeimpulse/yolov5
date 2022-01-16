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
    --validation-set-size) # e.g. 0.2
      VALIDATION_SET_SIZE="$2"
      shift # past argument
      shift # past value
      ;;
    --input-shape) # e.g. (96,96,3)
      INPUT_SHAPE="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
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
if [ -z "$VALIDATION_SET_SIZE" ]; then
    echo "Missing --validation-set-size"
    exit 1
fi
if [ -z "$INPUT_SHAPE" ]; then
    echo "Missing --input-shape"
    exit 1
fi

IMAGE_SIZE=$(python3 get_image_size.py --input-shape "$INPUT_SHAPE")

# fix learning rate in Yolov5, probably a better way to do it but this works ;-)
sed -i -e "s/lr0: 0.0032/lr0: $LEARNING_RATE/" yolov5/data/hyps/hyp.finetune.yaml
sed -i -e "s/lr0: 0.01/lr0: $LEARNING_RATE/" yolov5/data/hyps/hyp.scratch.yaml

# convert Edge Impulse dataset (in Numpy format, with JSON for labels into something Yolov5 understands)
python3 -u extract_dataset.py --x-file /home/X_train_features.npy --y-file /home/y_train.npy --out-directory /tmp/data --input-shape "$INPUT_SHAPE"

# --freeze 24 should freeze all layers
cd yolov5
# train
python3 -u train.py --img $IMAGE_SIZE --batch 16 --epochs $EPOCHS --freeze 24 --data /tmp/data/data.yaml --weights ../yolov5s.pt --name yolov5s_results --cache # --hyp data/hyps/hyp.finetune.yaml
echo "Training complete"
echo ""


# export as f32
echo "Converting to TensorFlow Lite model (fp16)..."
chronic python3 -u export.py --weights ./runs/train/yolov5s_results/weights/last.pt --img $IMAGE_SIZE --include tflite
cp runs/train/yolov5s_results/weights/last-fp16.tflite /home/model.tflite
echo "Converting to TensorFlow Lite model (fp16) OK"
echo ""

# export as i8 (skipping for now for speed)
# echo "Converting to TensorFlow Lite model (int8)..."
# chronic python3 -u export.py --weights ./runs/train/yolov5s_results/weights/last.pt --img $IMAGE_SIZE --include tflite --int8
# cp runs/train/yolov5s_results/weights/last-int8.tflite /home/model_quantized_int8_io.tflite
# echo "Converting to TensorFlow Lite model (int8) OK"
# echo ""
