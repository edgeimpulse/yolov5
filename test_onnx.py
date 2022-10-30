from onnxruntime import InferenceSession
import numpy as np
import cv2
import os
import math
import onnx
import argparse

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
features = get_features_from_img(input_shape, img)

sess = InferenceSession(args.onnx_file)
in_args = {}
in_args[input_name] = features
result = sess.run(None, in_args)
print('result', result[0].shape, result)
