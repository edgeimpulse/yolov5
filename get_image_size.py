import numpy as np
import argparse, math, shutil, os, json

parser = argparse.ArgumentParser(description='Edge Impulse get image size')
parser.add_argument('--x-file', type=str, required=True)

args = parser.parse_args()

X = np.load(args.x_file, mmap_mode='r')

image_size = int(math.sqrt(X.shape[1] / 3))
print(image_size)
