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
Purpose: This tool converts ONNX models into human-readable JSON format, allowing you to easily inspect,
edit, and analyze model architecture and weights.
"""

# [Usage]
"""
Usage:
python Tools/ONMAConvertOnnx.py --input Sample/Abs.onnx --output Sample/Abs.json
"""

# [Image]
"""
Sample/AbsSample.png
"""

# [Reference]
"""
Sample/Abs.json
Sample/Abs.onnx
"""

def main():
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-in", help="ONNX file")
    parser.add_argument("--output", "-out", help="JSON file")
    parser.add_argument("--store_npy", "-sn", default=False, action="store_true", help="Stores initializers to npy files")
    args = parser.parse_args()
    model = ONMAModel()

    model.ONMAModel_ConvertONNXToJson(args.input, args.output, args.store_npy)

if main() == False:
    sys.exit(-1)
