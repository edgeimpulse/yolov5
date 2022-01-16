# YOLOv5 transfer learning model for Edge Impulse

This repository is an example on how to bring a [custom transfer learning model](https://docs.edgeimpulse.com/docs/adding-custom-transfer-learning-models) into Edge Impulse. This repository is using YOLOv5 (an object detection model), but the same principles apply to other transfer learning models.

As a primer, read the [Adding custom transfer learning models](https://docs.edgeimpulse.com/docs/adding-custom-transfer-learning-models) page in the Edge Impulse docs.

What this repository does (see [run.sh](run.sh)):

1. Convert the training data / training labels into YOLOv5 format using [extract_dataset.py](extract_dataset.py).
1. Train YOLOv5 model (using https://github.com/ultralytics/yolov5).
1. Convert the YOLOv5 model into TFLite format.
1. Done!

To test this locally:

1. Create a new Edge Impulse project, and make sure the labeling method is set to 'Bounding boxes'.
1. Add and label some data.
1. Under **Create impulse** set the image size to 160x160, add an 'Image' DSP block and an 'Object Detection' learn block.
1. Generate features for the DSP block.
1. Then go to **Dashboard** and download the 'Image training data' and 'Image training labels' files.
1. Create a new folder in this repository named `home` and copy the downloaded files in under the names: `X_train_features.npy` and `y_train.npy`.
1. Build the container:

    ```
    $ docker build -t yolov5 .
    ```

1. Run the container to test:

    ```
    $ docker run --rm -v $PWD/home:/home yolov5 --epochs 1 --learning-rate 0.01 --validation-set-size 0.2 --input-shape "(160, 160, 3)"
    ```

1. This should have created a .tflite file in the 'home' directory.

Now you can initialize the block to Edge Impulse:

```
$ edge-impulse-blocks init
# Answer the questions, and select "yolov5" as the 'object detection output layer'
```

And push the block:

```
$ edge-impulse-blocks push
```

The block is now available under any project that's owned by your organization.
