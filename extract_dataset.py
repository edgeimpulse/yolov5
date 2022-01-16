import numpy as np
import argparse, math, shutil, os, json
from PIL import Image
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Edge Impulse => Yolov5')
parser.add_argument('--x-file', type=str, required=True)
parser.add_argument('--y-file', type=str, required=True)
parser.add_argument('--out-directory', type=str, required=True)
parser.add_argument('--input-shape', type=str, required=True)

args = parser.parse_args()

X = np.load(args.x_file, mmap_mode='r')
with open(args.y_file, 'r') as f:
    Y = json.loads(f.read())['samples']

image_width, image_height, image_channels = [ int(x) for x in args.input_shape.replace('(', '').replace(')', '').split(',') ]

out_dir = args.out_directory
if os.path.exists(out_dir) and os.path.isdir(out_dir):
    shutil.rmtree(out_dir)

class_count = 0

print('Transforming Edge Impulse data format into something compatible with Yolov5')

for ix in range(0, len(X)):
    print('[' + str(ix + 1).zfill(3) + ' / ' + str(len(X)).zfill(3) + '] Creating image...')

    raw_img_data = (np.reshape(X[ix], (image_width, image_height, image_channels)) * 255).astype(np.uint8)
    labels = Y[ix]['boundingBoxes']

    category = 'valid' if ix % 5 == 0 else 'train'

    images_dir = os.path.join(out_dir, category, 'images')
    labels_dir = os.path.join(out_dir, category, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    im = Image.fromarray(raw_img_data)
    im.save(os.path.join(images_dir, 'image' + str(ix).zfill(5) + '.jpg'))

    labels_text = []
    for l in labels:
        if (l['label'] > class_count):
            class_count = l['label']

        # class x_center y_center width height
        x_center = (l['x'] + l['w']) / 2 / image_width
        y_center = (l['y'] + l['h']) / 2 / image_height
        width = l['w'] / image_width
        height = l['h'] / image_height

        labels_text.append(str(l['label'] - 1) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height))

    with open(os.path.join(labels_dir, 'image' + str(ix).zfill(5) + '.txt'), 'w') as f:
        f.write('\n'.join(labels_text))

print('Transforming Edge Impulse data format into something compatible with Yolov5 OK')
print('')

class_names = []
for c in range(0, class_count):
    class_names.append("'class" + str(c) + "'")
class_names = ', '.join(class_names)

data_yaml = """
train: """ + os.path.join(os.path.abspath(out_dir), 'train', 'images') + """
val: """ + os.path.join(os.path.abspath(out_dir), 'valid', 'images') + """

nc: """ + str(class_count) + """
names: [""" + class_names + """]
"""

with open(os.path.join(out_dir, 'data.yaml'), 'w') as f:
    f.write(data_yaml)

yolo_spec = """# parameters
nc: """ + str(class_count) + """  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple

# anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, BottleneckCSP, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 9, BottleneckCSP, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, BottleneckCSP, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 3, BottleneckCSP, [1024, False]],  # 9
  ]

# YOLOv5 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, BottleneckCSP, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, BottleneckCSP, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, BottleneckCSP, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, BottleneckCSP, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
"""

with open(os.path.join(out_dir, 'yolov5s.yaml'), 'w') as f:
    f.write(yolo_spec)
