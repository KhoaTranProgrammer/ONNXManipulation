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
Purpose: this script will generate onnx file from ONMA json format.

Usage:
Run this script by command: python Tools/ONMACreateGraph.py --input Abs.json --output Abs.onnx
"""

def main():
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-in", help="Create graph from json", default="Abs.json")
    parser.add_argument("--output", "-ou", help="Output onnx file", default="Sample.onnx")
    args = parser.parse_args()

    with open(args.input) as user_file:
        file_contents = user_file.read()
    json_contents = json.loads(file_contents)
    model = ONMAModel()
    model.ONMAModel_CreateNetworkFromGraph(json_contents)
    model.ONMAModel_SaveModel(args.output)

if main() == False:
    sys.exit(-1)
