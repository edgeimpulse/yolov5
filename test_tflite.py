import tensorflow as tf
import numpy as np
import cv2
import os
import math
import argparse

parser = argparse.ArgumentParser(description='Image inferencing (TFLite)')
parser.add_argument('--tflite-file', type=str, help='TFLite file', required=True)
parser.add_argument('--image', type=str, help='Image file', required=True)

args, unknown = parser.parse_known_args()

if not os.path.exists(args.tflite_file):
    print(args.tflite_file + ' does not exist (via --tflite-file)')
    exit(1)
if not os.path.exists(args.image):
    print(args.image + ' does not exist (via --image)')
    exit(1)

def process_input(input_details, data):
    """Prepares an input for inference, quantizing if necessary.

    Args:
        input_details: The result of calling interpreter.get_input_details()
        data (numpy array): The raw input data

    Returns:
        A tensor object representing the input, quantized if necessary
    """
    if input_details[0]['dtype'] is np.int8:
        scale = input_details[0]['quantization'][0]
        zero_point = input_details[0]['quantization'][1]
        data = (data / scale) + zero_point
        data = np.around(data)
        data = data.astype(np.int8)
    return tf.convert_to_tensor(data)

def get_features_from_img(interpreter, img):
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']

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

def invoke(interpreter, item, specific_input_shape):
    """Invokes the Python TF Lite interpreter with a given input
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    item_as_tensor = process_input(input_details, item)
    if specific_input_shape:
        item_as_tensor = tf.reshape(item_as_tensor, specific_input_shape)
    # Add batch dimension
    item_as_tensor = tf.expand_dims(item_as_tensor, 0)
    interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output, output_details

interpreter = tf.lite.Interpreter(model_path=args.tflite_file)
interpreter.allocate_tensors()

img = cv2.imread(os.path.join(args.image))

input_data = get_features_from_img(interpreter, img)

output, output_details = invoke(interpreter, input_data, list(input_data.shape[1:]))
output0 = interpreter.get_tensor(output_details[0]['index'])
print('result', output0.shape, output0)
