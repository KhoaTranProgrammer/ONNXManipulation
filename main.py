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

    if args.operator == "Abs" or args.operator == "Acos" or args.operator == "Acosh":
        ONMAOperators.ONMAOperator_1_Input_1_Output(args.operator, 'newname', inputs=["X"], outputs=["Y"], datatype=onnx.TensorProto.FLOAT)

    if args.operator == "Add":
        ONMAOperators.ONMAOperator_2_Inputs_1_Output(args.operator, 'newname', inputs=["X1", "X2"], outputs=["Y"], datatype=onnx.TensorProto.FLOAT)

if main() == False:
    sys.exit(-1)
