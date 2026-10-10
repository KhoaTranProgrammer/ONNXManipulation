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

# [Purpose]
"""
This tool is designed to convert a model represented in the ONMA JSON format into an ONNX file.
It reads and parses the input ONMA JSON file, processes the model structure and its associated parameters,
and then generates an ONNX model file that can be used with ONNX-compatible frameworks and inference runtimes.
"""

# [Usage]
"""
Usage:
python Tools/ONMACreateGraph.py --input Sample/Conv.json --output Sample/Conv.onnx
"""

# [Image]
"""
Docs/Sample/ConvSample.gif
"""

# [Reference]
"""
Sample/Conv.json
Sample/Conv.onnx
"""

def main():
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-in", help="Create graph from json", default="Sample/Conv.json")
    parser.add_argument("--output", "-ou", help="Output onnx file", default="Sample/Conv.onnx")
    args = parser.parse_args()

    with open(args.input) as user_file:
        file_contents = user_file.read()
    json_contents = json.loads(file_contents)
    model = ONMAModel()
    model.ONMAModel_CreateNetworkFromGraph(json_contents)
    model.ONMAModel_SaveModel(args.output)

if main() == False:
    sys.exit(-1)
