import sys
import argparse
import subprocess
import numpy as np
import os
from pathlib import Path
import csv

file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
file_path = Path(file_path).as_posix()
sys.path.append(file_path)

# Format: Sample onnx json - Modify pattern json
TEST_DATA = []

# Open the test cases from the CSV file
with open('Tests/Combination/ONMAModifyModelTesting.csv', 'r', newline='') as f:
    reader = csv.reader(f)
    next(reader)  # skip first row
    for row in reader:
        TEST_DATA.append(row)

def pytest_generate_tests(metafunc):
    if {"onnxjson", "modifyjson", "type"} <= set(metafunc.fixturenames):
        metafunc.parametrize("onnxjson,modifyjson,type", TEST_DATA)

def test_execute(onnxjson, modifyjson, type):
    # Create onnx from json
    # Run the called script with arguments
    log = subprocess.run(['python', 'Tools/ONMACreateGraph.py', \
                    "--input", onnxjson, \
                    "--output", "Tests/Combination/ModifyModelSample.onnx"], \
                    capture_output=True, \
                    text=True )

    # Modify network
    result = subprocess.run(['python', 'Tools/ONMAModifyModel.py', \
                    "--modify", modifyjson, \
                    "--input", "Tests/Combination/ModifyModelSample.onnx", \
                    "--output", "Tests/Combination/ModifyModelSample_output.onnx"], \
                    capture_output=True, \
                    text=True )
    
    if "Result: Equivalent" in str(result):
        assert True
    else:
        assert False
