import sys
import argparse
import subprocess
import numpy as np
import os
from pathlib import Path

file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
file_path = Path(file_path).as_posix()
sys.path.append(file_path)

# Format: Sample onnx json - Modify pattern json
TEST_DATA = []
TEST_DATA.append(["Tests/Combination/OnnxSample/Softmax.json", "Sample/ModifyNetwork/Transformations/Replace_Softmax_By_Exp_RS_Div.json"])
TEST_DATA.append(["Tests/Combination/OnnxSample/Conv_WithBias_Add_InitializerAt0.json", "Sample/ModifyNetwork/Optimizations/Fuse_Add_Scalar_Into_Conv.json"])
TEST_DATA.append(["Tests/Combination/OnnxSample/Conv_WithBias_Add_InitializerAt1.json", "Sample/ModifyNetwork/Optimizations/Fuse_Add_Scalar_Into_Conv.json"])
TEST_DATA.append(["Tests/Combination/OnnxSample/Conv_NoneBias_Add_InitializerAt0.json", "Sample/ModifyNetwork/Optimizations/Fuse_Add_Scalar_Into_Conv.json"])
TEST_DATA.append(["Tests/Combination/OnnxSample/Conv_NoneBias_Add_InitializerAt1.json", "Sample/ModifyNetwork/Optimizations/Fuse_Add_Scalar_Into_Conv.json"])

def pytest_generate_tests(metafunc):
    if {"onnxjson", "modifyjson"} <= set(metafunc.fixturenames):
        metafunc.parametrize("onnxjson,modifyjson", TEST_DATA)

def test_execute(onnxjson, modifyjson):
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
