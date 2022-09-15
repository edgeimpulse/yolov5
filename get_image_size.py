import argparse, math, shutil, os, json
import numpy as np

parser = argparse.ArgumentParser(description='Edge Impulse get image size')
parser.add_argument('--data-directory', type=str, required=True)

args = parser.parse_args()

# Load data (images are in X_*.npy, labels are in JSON in Y_*.npy)
X_train = np.load(os.path.join(args.data_directory, 'X_split_train.npy'), mmap_mode='r')
image_width, image_height, image_channels = list(X_train.shape[1:])

print(image_width)
