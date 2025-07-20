import sys
import onnx
import argparse
import numpy as np
import json
from typing import Any, Sequence
import onnxruntime
import re
import itertools
import os
from pathlib import Path

file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = Path(file_path).as_posix()
sys.path.append(file_path)
from ONMA.ONMAModel import ONMAModel

"""
Purpose: this script will detect and run similar operators.

Usage:
Run this script by command: python Tools/ONMAConvertOnnx.py --input yolop-640-640.onnx --output yolop-640-640.json
"""

def compare_outputs(output1, output2, atol=1e-5, rtol=1e-5):
    if len(output1) != len(output2):
        print("Mismatch in the number of outputs")
        return False
    return all(np.allclose(out1, out2, atol=atol, rtol=rtol) for out1, out2 in zip(output1, output2))

def main():
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument("--sameoperator", "-so", help="Config that stores operators has same behavior", default="Sample/SameOperators.json")
    parser.add_argument("--operators", "-ops", help="Group of operators", default="Abs")
    parser.add_argument("--same_input", "-si", help="Use same input for all of networks", default="True")
    args = parser.parse_args()

    print(f'Check similar cases for operators {args.operators} ...')

    model = ONMAModel()
    with open(args.sameoperator) as user_file:
        file_contents = user_file.read()
    json_contents = json.loads(file_contents)

    inf_result = []

    # Get data of first network
    input_data = json_contents[args.operators]["list"][0]["inputs"]
    if args.same_input:
        first_key, first_value = next(iter(input_data.items()))

    for item in json_contents[args.operators]["list"]:
        if args.same_input:
            # Assign input of next networks by the input of first network
            _key, _value = next(iter((item["inputs"]).items()))
            item["inputs"][_key] = first_value

        model.ONMAModel_CreateNetworkFromGraph(item)
        inf = model.ONMAModel_Inference(item["inputs"])
        inf_result.append(inf)

    for index in range(1, len(inf_result)):
        equivalent = compare_outputs(inf_result[index-1], inf_result[index])
        if equivalent:
            print("Result: Equivalent")
        else:
            print("Result: NOT Equivalent")

if main() == False:
    sys.exit(-1)
