import numpy as np
from scc4onnx import order_conversion
import onnx
import onnx_graphsurgeon as gs
import argparse
from pathlib import Path
import os

parser = argparse.ArgumentParser(description='Convert ONNX model from NCHW to NHWC')
parser.add_argument('--onnx-file', type=str, required=True)
parser.add_argument('--out-file', type=str, required=True)

args, unknown = parser.parse_known_args()

onnx_graph = onnx.load(args.onnx_file)

onnx_graph = order_conversion(
    onnx_graph=onnx_graph,
    input_op_names_and_order_dims={"images": [0,2,3,1]},
    non_verbose=True
)

# create out directory
Path(os.path.dirname(args.out_file)).mkdir(parents=True, exist_ok=True)

onnx.save(onnx_graph, args.out_file)
