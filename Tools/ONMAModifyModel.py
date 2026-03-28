import sys
import onnx
import argparse
import numpy as np
import json
import os
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
        model.ONMAModel_UpdateModel(json_contents, args.output)

if main() == False:
    sys.exit(-1)
