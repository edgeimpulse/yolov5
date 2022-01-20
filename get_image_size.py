import argparse, math, shutil, os, json

parser = argparse.ArgumentParser(description='Edge Impulse get image size')
parser.add_argument('--input-shape', type=str, required=True)

args = parser.parse_args()

image_size = int(args.input_shape.replace('(', '').replace(')', '').split(',')[0])
print(image_size)
