# YOLOv5 transfer learning model for Edge Impulse

This repository contains the code to bring YOLOv5 models into Edge Impulse.

## Built-in block no longer available

As of February 28, 2025, YOLOv5 is no longer available as a built-in block in Edge Impulse - due to a recent policy change on how we deal with hosting third party model training pipelines. No worries though, because this repository contains instructions on how to get YOLOv5 back in a few minutes. After following the instructions below you can select *Choose a different model*, re-pick YOLOv5, and you can retrain your model.

## Adding YOLOv5 to your Edge Impulse account

To add this model to your Edge Impulse account (personal or enterprise organization):

1. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation)
2. Download this repository.
3. Open a command prompt or terminal, navigate to the folder where you downloaded this repository, and run:

    ```
    edge-impulse-blocks init
    edge-impulse-blocks push
    ```

4. In your Edge Impulse project go to **Create impulse > Add learning block** and select **YOLOv5**. 🎉

## Modifying the model code

You can use this repository as a basis to bring other types of ML base models into Edge Impulse. As a primer, read the [Custom learning blocks](https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/adding-custom-learning-blocks) page in the Edge Impulse docs.

The entrypoint for this ML block is [run.sh](run.sh). It does:

1. Convert the training data / training labels into YOLOv5 format using [extract_dataset.py](extract_dataset.py).
1. Train YOLOv5 model (using https://github.com/ultralytics/yolov5).
1. Convert the YOLOv5 model into TFLite format.
1. Done!

> **Note on epoch count:** YOLOv5 might take a while to converge, especially on large images. Play around with epoch count, or lower the resolution of your input until you have something that works.

## Running the pipeline locally

You run this pipeline locally via Docker. This encapsulates all dependencies and packages for you.

### Running via Docker

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation).
3. Create a new Edge Impulse project, and make sure the labeling method is set to 'Bounding boxes'.
4. Add and label some data.
5. Under **Create impulse** set the image size to e.g. 160x160, 320x320 or 640x640, add an 'Image' DSP block and an 'Object Detection' learn block.
6. Open a command prompt or terminal window.
7. Initialize the block:

    ```
    $ edge-impulse-blocks init
    ```

8. Fetch new data from an object detection project via:

    ```
    $ edge-impulse-blocks runner --download-data data/
    ```

9. Build the container:

    ```
    $ docker build -t yolov5 .
    ```

10. Run the container to test the script (you don't need to rebuild the container if you make changes):

    ```
    $ docker run --shm-size=1024m --rm -v $PWD:/scripts yolov5 --data-directory data/ --epochs 30 --model-size n --out-directory out/
    ```

    > If you have an NVIDIA GPU, pass `--gpus all` to train on GPU.

11. This creates an .onnx file and two .tflite files (both quantized and unquantized) in the 'out' directory.

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

    > **Note:** To fetch data from another project, run ` edge-impulse-blocks runner --download-data data/ --clean`

## Pushing the block back to Edge Impulse

You can also push this block back to Edge Impulse, that makes it available like any other ML block so you can retrain your model when new data comes in, or deploy the model to device. See [Docs > Adding custom learning blocks](https://docs.edgeimpulse.com/docs/edge-impulse-studio/organizations/adding-custom-transfer-learning-models) for more information.

1. Push the block:

    ```
    $ edge-impulse-blocks push
    ```

2. The block is now available under any of your projects, via  **Create impulse > Add learning block**.

## License notice

This repository is licensed under [The Clear BSD License](LICENSE), but is utilizing the GPLv3 licensed [ultralytics/yolov5](https://github.com/ultralytics/yolov5) repository (from a commit before this repository was changed to AGPL) at arm's length.
