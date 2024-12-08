import sys
import onnx
import argparse
import numpy as np
from ONMAOperators import ONMAOperators

global args

operator_list = \
{
    "Abs": {"function": "ONMAOperator_1_Input_1_Output", "graph_name": "Abs_sample", "inputs": ["X"], "outputs": ["Y"], "datatype": onnx.TensorProto.FLOAT},
    "Acos": {"function": "ONMAOperator_1_Input_1_Output", "graph_name": "Acos_sample", "inputs": ["X"], "outputs": ["Y"], "datatype": onnx.TensorProto.FLOAT},
    "Acosh": {"function": "ONMAOperator_1_Input_1_Output", "graph_name": "Acosh_sample", "inputs": ["X"], "outputs": ["Y"], "datatype": onnx.TensorProto.FLOAT},
    "Asin": {"function": "ONMAOperator_1_Input_1_Output", "graph_name": "Asin_sample", "inputs": ["X"], "outputs": ["Y"], "datatype": onnx.TensorProto.FLOAT},
    "Asinh": {"function": "ONMAOperator_1_Input_1_Output", "graph_name": "Asinh_sample", "inputs": ["X"], "outputs": ["Y"], "datatype": onnx.TensorProto.FLOAT},
    "Atan": {"function": "ONMAOperator_1_Input_1_Output", "graph_name": "Atan_sample", "inputs": ["X"], "outputs": ["Y"], "datatype": onnx.TensorProto.FLOAT},
    "Atanh": {"function": "ONMAOperator_1_Input_1_Output", "graph_name": "Atanh_sample", "inputs": ["X"], "outputs": ["Y"], "datatype": onnx.TensorProto.FLOAT},
    "Add": {"function": "ONMAOperator_2_Inputs_1_Output", "graph_name": "Add_sample", "inputs": ["X1", "X2"], "outputs": ["Y"], "datatype": onnx.TensorProto.FLOAT},
    "And": {"function": "ONMAOperator_2_Inputs_1_Output", "graph_name": "And_sample", "inputs": ["X1", "X2"], "outputs": ["Y"], "datatype": onnx.TensorProto.BOOL},
    "BitwiseNot": {"function": "ONMAOperator_1_Input_1_Output", "graph_name": "BitwiseNot_sample", "inputs": ["X"], "outputs": ["Y"], "datatype": onnx.TensorProto.UINT16},
    "BitwiseAnd": {"function": "ONMAOperator_2_Inputs_1_Output", "graph_name": "BitwiseAnd_sample", "inputs": ["X1", "X2"], "outputs": ["Y"], "datatype": onnx.TensorProto.INT32},
    "BitwiseOr": {"function": "ONMAOperator_2_Inputs_1_Output", "graph_name": "BitwiseOr_sample", "inputs": ["X1", "X2"], "outputs": ["Y"], "datatype": onnx.TensorProto.INT32},
    "BitwiseXor": {"function": "ONMAOperator_2_Inputs_1_Output", "graph_name": "BitwiseXor_sample", "inputs": ["X1", "X2"], "outputs": ["Y"], "datatype": onnx.TensorProto.INT32},
}

def main():
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", "-op", help="Operator name", default="")
    args = parser.parse_args()

    function_name = operator_list[args.operator]["function"]
    operator_processing = getattr(ONMAOperators, function_name)
    operator_processing(args.operator, operator_list[args.operator]["graph_name"], inputs=operator_list[args.operator]["inputs"], outputs=operator_list[args.operator]["outputs"], datatype=operator_list[args.operator]["datatype"])

if main() == False:
    sys.exit(-1)
