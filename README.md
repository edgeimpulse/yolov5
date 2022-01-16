# Yolov5 model for Edge Impulse

First build the container:

```
docker build -t yolov5 .
```

Then, checkout the `yolov5` branch for `edgeimpulse` and insert the custom transfer learning model:

```
insert into studio.organization_transfer_learning_blocks
	(organization_id, created, name, description, operates_on, model_path, init_code, docker_container, object_detection_last_layer)
values (1, NOW(), 'YoloV5', 'Hello world', 'object_detection', '', '', 'yolov5:latest', 'yolov5')
```

Now you'll have YoloV5 listed as a new transfer learning model under 'Object detection' (project needs to be owned by an organization).
