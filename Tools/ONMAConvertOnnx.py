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
Purpose: this script will convert onnx file to ONMA json format.

Usage:
Run this script by command: python Tools/ONMAConvertOnnx.py --input yolop-640-640.onnx --output yolop-640-640.json
"""

def main():
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-in", help="ONNX file")
    parser.add_argument("--output", "-out", help="JSON file")
    args = parser.parse_args()
    model = ONMAModel()

    model.ONMAModel_ConvertONNXToJson(args.input, args.output)

if main() == False:
    sys.exit(-1)
