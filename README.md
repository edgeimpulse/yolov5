# YOLOv5 transfer learning model for Edge Impulse

This repository is an example on how to [bring your own model](https://docs.edgeimpulse.com/docs/adding-custom-transfer-learning-models) into Edge Impulse. This repository is using YOLOv5 (an object detection model), but the same principles apply to other transfer learning models.

As a primer, read the [Bring your own model](https://docs.edgeimpulse.com/docs/adding-custom-transfer-learning-models) page in the Edge Impulse docs.

What this repository does (see [run.sh](run.sh)):

1. Convert the training data / training labels into YOLOv5 format using [extract_dataset.py](extract_dataset.py).
1. Train YOLOv5 model (using https://github.com/ultralytics/yolov5).
1. Convert the YOLOv5 model into TFLite format.
1. Done!

> **Note on epoch count:** YOLOv5 might take a while to converge, especially on large images. Play around with epoch count, or lower the resolution of your input until you have something that works.

## Running the pipeline

You run this pipeline via Docker. This encapsulates all dependencies and packages for you.

### Running via Docker

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) v1.16.0 or higher.
3. Create a new Edge Impulse project, and make sure the labeling method is set to 'Bounding boxes'.
4. Add and label some data.
5. Under **Create impulse** set the image size to e.g. 160x160, 320x320 or 640x640, add an 'Image' DSP block and an 'Object Detection' learn block.
6. Open a command prompt or terminal window.
7. Initialize the block:

    ```
    $ edge-impulse-blocks init
    # Answer the questions, select "Object Detection" for 'What type of data does this model operate on?' and "YOLOv5" for 'What's the last layer...'
    ```

8. Fetch new data via:

    ```
    $ edge-impulse-blocks runner --download-data data/
    ```

9. Build the container:

    ```
    $ docker build -t yolov5 .
    ```

10. Run the container to test the script (you don't need to rebuild the container if you make changes):

    ```
    $ docker run --shm-size=1024m --rm -v $PWD:/scripts yolov5 --data-directory data/ --epochs 30 --learning-rate 0.01 --out-directory out/
    ```
    Add the `--gpus all` option to the `docker run` command to enable usage of GPU while training. This will speed up the process a lot.

11. This creates a .tflite file in the 'out' directory.

#### Adding extra dependencies

If you have extra packages that you want to install within the container, add them to `requirements.txt` and rebuild the container.

## Fetching new data

To get up-to-date data from your project:

1. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) v1.16 or higher.
2. Open a command prompt or terminal window.
3. Fetch new data via:

    ```
    $ edge-impulse-blocks runner --download-data data/
    ```

> **Note: to fetch data from another project** run the following:
> ```
> $ edge-impulse-blocks --clean
> # Then follow the steps provided
> $ edge-impulse-blocks runner --download-data data/
> # Then choose your project
> ```

## Pushing the block back to Edge Impulse

You can also push this block back to Edge Impulse, that makes it available like any other ML block so you can retrain your model when new data comes in, or deploy the model to device. See [Docs > Adding custom learning blocks](https://docs.edgeimpulse.com/docs/edge-impulse-studio/organizations/adding-custom-transfer-learning-models) for more information.

1. Push the block:

    ```
    $ edge-impulse-blocks push
    ```

2. The block is now available under any of your projects, via  **Create impulse > Add learning block > Object Detection (Images)**.
