import sys
import onnx
import argparse
import numpy as np
import json
from typing import Any, Sequence
import onnxruntime
import re
import itertools

"""
Purpose: this script will convert Operators.md to JSON format for easy processing.

Example:
- Abs operator in Operators.md
================================
Inputs
X (differentiable) : T
Input tensor
Outputs
Y (differentiable) : T
Output tensor
Type Constraints
T : tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
================================

- Abs operator in JSON
================================
"Abs": {
    "Attributes": {},
    "Inputs": {
        "X": "uint8,uint16,uint32,uint64,int8,int16,int32,int64,float16,float,double,bfloat16"
    },
    "Outputs": {
        "Y": "uint8,uint16,uint32,uint64,int8,int16,int32,int64,float16,float,double,bfloat16"
    }
}
================================

Usage:
In order to run this script, need to download https://github.com/onnx/onnx/blob/main/docs/Operators.md
And put Operators.md at ONNXManipulation

Run this script by command: python Tools/ONMAOperatorsSpec.py
Example version of JSON format is put at: ONNXManipulation/Sample/ONNXOperatorsSpec.json
"""

def readSpecForOneNode(spec):
    isInput = False
    input = []
    isOutput = False
    output = []
    isTypeConstraints = False
    type = []
    isAttributes = False
    attributes = []

    node_name = ""
    node_spec = {}
    node_input = {}
    node_output = {}
    node_type = {}
    node_attributes = {}

    for line in spec:
        result = re.findall(r"^### <a name=*>*", line) # Get start segment of node
        if result != []:
            node_name = (line.split(">**"))[1]
            node_name = (node_name.split("**<"))[0]

        if  "#### Inputs" in line:
            isInput = True
            isOutput = False
            isTypeConstraints = False
            isAttributes = False

        if "#### Outputs" in line:
            isInput = False
            isOutput = True
            isTypeConstraints = False
            isAttributes = False

        if "#### Type Constraints" in line:
            isInput = False
            isOutput = False
            isTypeConstraints = True
            isAttributes = False

        if "#### Examples" in line:
            isInput = False
            isOutput = False
            isTypeConstraints = False
            isAttributes = False

        if "#### Attributes" in line:
            isInput = False
            isOutput = False
            isTypeConstraints = False
            isAttributes = True

        if isInput: input.append(line)
        if isOutput: output.append(line)
        if isTypeConstraints: type.append(line)
        if isAttributes: attributes.append(line)
    
    for item in type:
        if "<dt><tt>" in item: # <dt><tt>T</tt> : tensor(uint8), tensor(uint16)
            item_sepa = item.split(" : ")
            type_name = item_sepa[0]
            type_name = type_name.replace("<dt><tt>", "")
            type_name = type_name.replace("</tt>", "")
            type_value = item_sepa[1]
            type_value = type_value.replace("</dt>", "")
            type_value = type_value.replace("tensor(", "")
            type_value = type_value.replace("), ", ",")
            type_value = type_value.replace(")\n", "")
            node_type[type_name] = type_value

    for item in input:
        if "<dt><tt>" in item: # <dt><tt>X</tt> (differentiable) : T</dt>
            item_sepa = item.split(" : ")
            input_name = item_sepa[0]
            input_name = input_name.replace("<dt><tt>", "")
            input_name = input_name.replace("</tt> (differentiable)", "")
            input_name = input_name.replace("</tt> (non-differentiable)", "")
            input_name = input_name.replace("</tt> (variadic, differentiable)", "")
            input_name = input_name.replace("</tt> (optional, non-differentiable)", "(optional)")
            input_name = input_name.replace("</tt> (optional, differentiable)", "(optional)")
            input_name = input_name.replace("</tt>", "")
            input_value = item_sepa[1]
            input_value = input_value.replace("</dt>\n", "")
            input_value = input_value.replace("tensor(", "")
            input_value = input_value.replace(")", "")
            try:
                node_input[input_name] = node_type[input_value]
            except:
                node_input[input_name] = input_value

    for item in output:
        if "<dt><tt>" in item: # <dt><tt>X</tt> (differentiable) : T</dt>
            item_sepa = item.split(" : ")
            output_name = item_sepa[0]
            output_name = output_name.replace("<dt><tt>", "")
            output_name = output_name.replace("</tt> (differentiable)", "")
            output_name = output_name.replace("</tt> (non-differentiable)", "")
            output_name = output_name.replace("</tt> (optional, non-differentiable)", "(optional)")
            output_name = output_name.replace("</tt> (optional, differentiable)", "(optional)")
            output_name = output_name.replace("</tt> (variadic, differentiable)", "(variadic)")
            output_name = output_name.replace("</tt>", "")
            output_value = item_sepa[1]
            output_value = output_value.replace("</dt>\n", "")
            output_value = output_value.replace("tensor(", "")
            output_value = output_value.replace(")", "")
            try:
                node_output[output_name] = node_type[output_value]
            except:
                node_output[output_name] = output_value

    # for item in attributes:
    for i, item in enumerate(attributes):
        if "<dt><tt>" in item: # <dt><tt>axis</tt> : int (default is 0)</dt>
            item_sepa = item.split(" : ")
            attributes_name = item_sepa[0]
            attributes_name = attributes_name.replace("<dt><tt>", "")
            attributes_name = attributes_name.replace("</tt>", "")
            attributes_value = item_sepa[1]
            attributes_value = attributes_value.replace("</dt>\n", "")
            if "(Optional)" in attributes[i+1]:
                attributes_name = attributes_name + "(optional)"
            node_attributes[attributes_name] = attributes_value

    node_spec["Attributes"] = node_attributes
    node_spec["Inputs"] = node_input
    node_spec["Outputs"] = node_output
    return node_name, node_spec

def ONMAOperatorsSpec_ReadOnnxNodeSpec():
    node_dict = {}
    
    # Open the file in read mode
    with open('Operators.md', 'r') as file:
        isStart = False
        spec = []
        # Read each line in the file
        for line in file:
            if ("## ai.onnx.preview.training" in line or "(deprecated)" in line) and spec != []:
                isStart = False
                node_name, node_spec = readSpecForOneNode(spec)
                node_dict[node_name] = node_spec
                node_dict[node_name] = node_spec
                node_dict[node_name] = node_spec
                spec = []

            result = re.findall(r"^### <a name=*>*", line) # Get start segment of node
            if result != []:
                if "ai.onnx.preview.training" not in line and "(deprecated)" not in line:
                    isStart = True
                    if spec != []:
                        node_name, node_spec = readSpecForOneNode(spec)
                        node_dict[node_name] = node_spec
                        node_dict[node_name] = node_spec
                        spec = []
                else:
                    isStart = False
                    spec = []

            if isStart == True:
                spec.append(line)
    return node_dict

def main():
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-out", help="JSON file to store ONNX spec", default="ONNXOperatorsSpec.json")
    args = parser.parse_args()

    node_dict = ONMAOperatorsSpec_ReadOnnxNodeSpec()
    with open(args.output, "w") as fp:
        json.dump(node_dict, fp, indent = 4)

if main() == False:
    sys.exit(-1)
