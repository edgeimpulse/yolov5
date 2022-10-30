from onnxruntime import InferenceSession
import numpy as np
import cv2
import os
import math
import onnx
import argparse
import time

np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description='Image inferencing (ONNX)')
parser.add_argument('--onnx-file', type=str, help='ONNX file', required=True)
parser.add_argument('--image', type=str, help='Image file', required=True)

args, unknown = parser.parse_known_args()

if not os.path.exists(args.onnx_file):
    print(args.onnx_file + ' does not exist (via --onnx-file)')
    exit(1)
if not os.path.exists(args.image):
    print(args.image + ' does not exist (via --image)')
    exit(1)

def get_features_from_img(input_shape, img):
    count, width, height, channels = input_shape

    # if channels == width of the image, then we are dealing with channel/width/height
    # instead of height/width/channel
    is_nchw = channels == img.shape[1]
    if (is_nchw):
        count, channels, width, height = input_shape

    print('is_nchw', is_nchw, 'input_shape', input_shape)

    if (channels == 3):
        ret = np.array([ x / 255 for x in list(img.flatten()) ]).astype(np.float32).reshape((1, width, height, channels))
    elif (channels == 1):
        rgb_weights = [0.2989, 0.5870, 0.1140]
        img_grayscale = np.dot(img[...,:3], rgb_weights)
        ret = np.array([ x / 255 for x in list(img_grayscale.flatten()) ]).astype(np.float32).reshape((1, width, height, channels))
    else:
        raise ValueError('Unknown depth for image')

    # transpose the image if required
    if (is_nchw):
        ret = np.transpose(ret, (0, 3, 1, 2))

    return ret

model = onnx.load(args.onnx_file)
if (len(model.graph.input) != 1):
    print('More than 1 input tensor, not supported')
    exit(1)

input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
input_name = model.graph.input[0].name

img = cv2.imread(os.path.join(args.image))
print('img shape', img.shape)
features = get_features_from_img(input_shape, img)

sess = InferenceSession(args.onnx_file)
in_args = {}
in_args[input_name] = features
result = sess.run(None, in_args)
print('result', result[0].shape)

print('first 20 bytes', result[0].flatten()[0:20])

def yolov5_class_filter(classdata):
    classes = []  # create a list
    for i in range(classdata.shape[0]):         # loop through all predictions
        classes.append(classdata[i].argmax())   # get the best classification location
    return classes  # return classes (int)

def yolov5_detect(output_data):  # input = interpreter, output is boxes(xyxy), classes, scores
    output_data = output_data[0]                # x(1, 25200, 7) to x(25200, 7)
    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
    scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
    classes = yolov5_class_filter(output_data[..., 5:]) # get classes
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]

    return xyxy, classes, scores  # output is boxes(x,y,x,y), classes(int), scores(float) [predictions length]

def render_out_img(xyxy, classes, scores):
    rects = []
    labels = []
    score_res = []

    minimum_confidence_rating = 0.3

    for i in range(len(scores)):
        if ((scores[i] >= minimum_confidence_rating) and (scores[i] <= 1.0)):
            xmin = float(xyxy[0][i])
            ymin = float(xyxy[1][i])
            xmax = float(xyxy[2][i])
            ymax = float(xyxy[3][i])

            print('index', i, 'has detection', scores[i],
                'xmin', xmin, 'xmax', xmax, 'ymin', ymin, 'ymax', ymax)

            # Who in their right min has decided to do ymin,xmin,ymax,xmax ?
            bbox = [ymin, xmin, ymax, xmax]

            rects.append(bbox)
            labels.append(int(classes[i]))
            score_res.append(float(scores[i]))

    for i in range(0, len(labels)):
        # if i != 0: continue

        bb = rects[i]
        [ymin, xmin, ymax, xmax] = bb

        xmin = int(xmin)
        xmax = int(xmax)
        ymin = int(ymin)
        ymax = int(ymax)

        color = (255, 0, 0)
        if (labels[i] == 1):
            color = (0, 255, 0)

        print('label', labels[i], 'confidence', score_res[i], 'x', xmin, 'y', ymin, 'w', xmax-xmin, 'h', ymax-ymin,
            'color', color)

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

    cv2.imwrite('out.png', img)

xyxy, classes, scores = yolov5_detect(result[0])
# print('xyxy', xyxy, 'classes', classes, 'scores', scores)
render_out_img(xyxy, classes, scores)
