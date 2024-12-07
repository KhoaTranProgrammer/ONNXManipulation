import sys
import onnx
import argparse
import numpy as np
from ONMAOperators import ONMAOperators

global args

def main():
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", "-op", help="Operator name", default="")
    args = parser.parse_args()

    if args.operator == "Abs" or args.operator == "Acos" or args.operator == "Acosh" or args.operator == "Asin" or args.operator == "Asinh" or args.operator == "Atan" or args.operator == "Atanh":
        ONMAOperators.ONMAOperator_1_Input_1_Output(args.operator, 'newname', inputs=["X"], outputs=["Y"])

    if args.operator == "Add":
        ONMAOperators.ONMAOperator_2_Inputs_1_Output(args.operator, 'newname', inputs=["X1", "X2"], outputs=["Y"])

    if args.operator == "And":
        ONMAOperators.ONMAOperator_2_Inputs_1_Output(args.operator, 'newname', inputs=["X1", "X2"], outputs=["Y"], datatype=onnx.TensorProto.BOOL)

    if args.operator == "BitwiseNot":
        ONMAOperators.ONMAOperator_1_Input_1_Output(args.operator, 'newname', inputs=["X"], outputs=["Y"], datatype=onnx.TensorProto.UINT16)

    if args.operator == "BitwiseAnd" or args.operator == "BitwiseOr" or args.operator == "BitwiseXor":
        ONMAOperators.ONMAOperator_2_Inputs_1_Output(args.operator, 'newname', inputs=["X1", "X2"], outputs=["Y"], datatype=onnx.TensorProto.INT32)

if main() == False:
    sys.exit(-1)
