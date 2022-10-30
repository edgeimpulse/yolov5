import numpy as np
import argparse, math, shutil, os, json, time
import onnx
from onnx_tf.backend import prepare # https://github.com/onnx/onnx-tensorflow
import tensorflow as tf

parser = argparse.ArgumentParser(description='ONNX to TFLite')
parser.add_argument('--onnx-file', type=str, required=True)
parser.add_argument('--out-file', type=str, required=True)

args = parser.parse_args()

# Load the ONNX model
onnx_model = onnx.load(args.onnx_file)

# Check that the IR is well formed
onnx.checker.check_model(onnx_model)

# Now do ONNX => TF
tf_model_path = '/tmp/savedmodel'
tf_rep = prepare(onnx_model, device='cpu')
tf_rep.export_graph(tf_model_path)

# TF => TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

# Save the model
with open(args.out_file, 'wb') as f:
    f.write(tflite_model)
