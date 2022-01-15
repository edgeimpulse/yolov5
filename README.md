# Yolov5 test

```
docker build -t yolov5 .
docker run --rm -v $PWD/home:/home yolov5
```

In the home directory you can drop an X and Y file from an Edge Impulse object detection project and it'll autoconvert into Yolov5 format, and then train the model. I don't think I'm doing it right yet as this refuses to converge.
