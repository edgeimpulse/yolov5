import onnx
import argparse

parser = argparse.ArgumentParser(description='Cut ONNX graph for DRPAI')
parser.add_argument('--onnx-file', type=str, help='ONNX file', required=True)
parser.add_argument('--out-file', type=str, help='ONNX output file', required=True)
parser.add_argument('--output-names', type=str, nargs='+', help='output names', required=True)

args, unknown = parser.parse_known_args()

input_names  = ['images']
output_names = args.output_names

path = args.onnx_file
output_path = args.out_file
onnx.utils.extract_model(path, output_path, input_names, output_names)
