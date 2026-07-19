import sys
import onnx
import argparse
import numpy as np
import json
import os
import onnxruntime as ort
from pathlib import Path

file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = Path(file_path).as_posix()
sys.path.append(file_path)
from ONMA.ONMAModel import ONMAModel

"""
Purpose: this script will modify network using description

Usage:
Run this script by command: python Tools/ONMAModifyModel.py
                            --input yolop-640-640.onnx
                            --output yolop-640-640_modify.onnx
                            --modify Sample/Modify_Sample.json
"""

def load_model(model_path):
    return ort.InferenceSession(model_path)

def generate_sample_input(session):
    input_data = {}
    for input_tensor in session.get_inputs():
        input_name = input_tensor.name
        input_shape = input_tensor.shape

        input_shape = [1 if dim is None else dim for dim in input_shape]
        input_data[input_name] = np.random.rand(*input_shape).astype(np.float32)
    return input_data

def run_inference(session, input_data):
    return session.run(None, input_data)

def compare_outputs(output1, output2, atol=1e-5, rtol=1e-5):
    if len(output1) != len(output2):
        print("Mismatch in the number of outputs")
        return False
    return all(np.allclose(out1, out2, atol=atol, rtol=rtol) for out1, out2 in zip(output1, output2))

def main():
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument("--modify", "-md", help="Setting to modify network", default="")
    parser.add_argument("--input", "-in", help="Input ONNX file path")
    parser.add_argument("--output", "-ot", help="Output ONNX file path")
    args = parser.parse_args()

    if args.modify != "":
        with open(args.modify) as user_file:
            file_contents = user_file.read()
        json_contents = json.loads(file_contents)

        model = ONMAModel()
        model.ONMAModel_LoadModel(args.input)
        status = model.ONMAModel_UpdateModel(json_contents, args.output)
        if status is False:
            print("Failed to update model")
            sys.exit(-1)

    session1 = load_model(args.input)
    session2 = load_model(args.output)

    input_data = generate_sample_input(session1)

    output1 = run_inference(session1, input_data)
    output2 = run_inference(session2, input_data)

    equivalent = compare_outputs(output1, output2)

    model_input = onnx.load(args.input)
    model_output = onnx.load(args.output)
    
    if equivalent and (model_input.SerializeToString() != model_output.SerializeToString()):
        print("Result: Equivalent")
    else:
        print("Result: NOT Equivalent")

if main() == False:
    sys.exit(-1)
