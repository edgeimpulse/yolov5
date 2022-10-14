import tensorflow as tf
import numpy as np
import cv2
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

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

    if (channels == 3):
        ret = np.array([ x for x in list(img.flatten()) ]).astype(np.float32).reshape((1, width, height, channels))
    elif (channels == 1):
        rgb_weights = [0.2989, 0.5870, 0.1140]
        img_grayscale = np.dot(img[...,:3], rgb_weights)
        ret = np.array([ x for x in list(img_grayscale.flatten()) ]).astype(np.float32).reshape((1, width, height, channels))
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

interpreter = tf.lite.Interpreter(model_path=os.path.join(dir_path, "out/model.tflite"))
interpreter.allocate_tensors()

# Extract your dataset via: python3 extract_dataset.py --data-directory data --out-directory data-out
img = cv2.imread(os.path.join(dir_path, 'data-out/train/images/image00009.jpg'))
print('img shape', img.shape)

input_data = get_features_from_img(interpreter, img)
print('input_data', input_data.shape)

output, output_details = invoke(interpreter, input_data, list(input_data.shape[1:]))
print('output_details', output_details)

output0 = interpreter.get_tensor(output_details[0]['index'])
print('output tensor is', output0.shape)

# now what I need is some way to go from output tensor to bounding boxes
# (confidence, label, bb x/y/w/h)
# e.g. https://stackoverflow.com/questions/65824714/process-output-data-from-yolov5-tflite
# but this is for v6...
