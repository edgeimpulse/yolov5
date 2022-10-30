import onnx
import argparse

parser = argparse.ArgumentParser(description='Add shape info to an ONNX file')
parser.add_argument('--onnx-file', type=str, help='ONNX file', required=True)

args, unknown = parser.parse_known_args()

path = args.onnx_file
onnx.save(onnx.shape_inference.infer_shapes(onnx.load(path)), path)
