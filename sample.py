import sys
import onnx
import argparse
import numpy as np
import json
from typing import Any, Sequence
import onnxruntime
from ONMAModel import ONMAModel
from onnx.backend.test.case.node import _extract_value_info
from onnx.backend.test.case.node.affinegrid import create_theta_2d
from onnx.backend.test.case.node.roialign import get_roi_align_input_values
from onnx.backend.test.case.node.layernormalization import calculate_normalized_shape
import re

def expect(
    node: onnx.NodeProto,
    inputs: Sequence[np.ndarray],
    outputs: Sequence[np.ndarray],
    name: str,
    **kwargs: Any,
) -> None:
    # Builds the model
    present_inputs = [x for x in node.input if (x != "")]
    present_outputs = [x for x in node.output if (x != "")]
    input_type_protos = [None] * len(inputs)
    if "input_type_protos" in kwargs:
        input_type_protos = kwargs["input_type_protos"]
        del kwargs["input_type_protos"]
    output_type_protos = [None] * len(outputs)
    if "output_type_protos" in kwargs:
        output_type_protos = kwargs["output_type_protos"]
        del kwargs["output_type_protos"]
    inputs_vi = [
        _extract_value_info(arr, arr_name, input_type)
        for arr, arr_name, input_type in zip(inputs, present_inputs, input_type_protos)
    ]
    outputs_vi = [
        _extract_value_info(arr, arr_name, output_type)
        for arr, arr_name, output_type in zip(
            outputs, present_outputs, output_type_protos
        )
    ]
    graph = onnx.helper.make_graph(
        nodes=[node], name=name, inputs=inputs_vi, outputs=outputs_vi
    )
    kwargs["producer_name"] = "backend-test"

    if "opset_imports" not in kwargs:
        # To make sure the model will be produced with the same opset_version after opset changes
        # By default, it uses since_version as opset_version for produced models
        produce_opset_version = onnx.defs.get_schema(
            node.op_type, domain=node.domain
        ).since_version
        kwargs["opset_imports"] = [
            onnx.helper.make_operatorsetid(node.domain, produce_opset_version)
        ]

    model = onnx.helper.make_model_gen_version(graph, **kwargs)

    # Checking the produces are the expected ones.
    try:
        sess = onnxruntime.InferenceSession(model.SerializeToString(),
                                            providers=["CPUExecutionProvider"])
        feeds = {name: value for name, value in zip(node.input, inputs)}
        results = sess.run(None, feeds)
        return 1
    except:
        return 0
    # for expected, output in zip(outputs, results):
        # np.testing.assert_allclose(expected, output)
        # print(f'expected: {expected}')
        # print(f'output: {output}')

# Element type: https://onnx.ai/onnx/intro/concepts.html#element-type
element_type = {
    "onnx.TensorProto.FLOAT": np.float32,
    "onnx.TensorProto.UINT8": np.uint8,
    "onnx.TensorProto.INT8": np.int8,
    "onnx.TensorProto.UINT16": np.uint16,
    "onnx.TensorProto.INT16": np.int16,
    "onnx.TensorProto.INT32": np.int32,
    "onnx.TensorProto.INT64": np.int64,
    "onnx.TensorProto.STRING": np.string_,
    "onnx.TensorProto.BOOL": np.bool_,
    "onnx.TensorProto.FLOAT16": np.float16,
    "onnx.TensorProto.DOUBLE": np.float64,
    "onnx.TensorProto.UINT32": np.uint32,
    "onnx.TensorProto.UINT64": np.uint64,
    "onnx.TensorProto.COMPLEX64": np.complex64,
    "onnx.TensorProto.COMPLEX128": np.complex128,
    "onnx.TensorProto.BFLOAT16": "",
    "onnx.TensorProto.FLOAT8E4M3FN": "",
    "onnx.TensorProto.FLOAT8E4M3FNUZ": "",
    "onnx.TensorProto.FLOAT8E5M2": "",
    "onnx.TensorProto.FLOAT8E5M2FNUZ": "",
    "onnx.TensorProto.UINT4": "",
    "onnx.TensorProto.INT4": "",
    "onnx.TensorProto.FLOAT4E2M1": ""
}

def createSampleData(dimentions, datatype):
    return np.random.randn(*dimentions).astype(datatype)

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
            input_name = input_name.replace("</tt> (optional, non-differentiable)", "")
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
            output_name = output_name.replace("</tt> (optional, non-differentiable)", "")
            output_name = output_name.replace("</tt>", "")
            output_value = item_sepa[1]
            output_value = output_value.replace("</dt>\n", "")
            output_value = output_value.replace("tensor(", "")
            output_value = output_value.replace(")", "")
            try:
                node_output[output_name] = node_type[output_value]
            except:
                node_output[output_name] = output_value

    for item in attributes:
        if "<dt><tt>" in item: # <dt><tt>axis</tt> : int (default is 0)</dt>
            item_sepa = item.split(" : ")
            attributes_name = item_sepa[0]
            attributes_name = attributes_name.replace("<dt><tt>", "")
            attributes_name = attributes_name.replace("</tt>", "")
            attributes_value = item_sepa[1]
            attributes_value = attributes_value.replace("</dt>\n", "")
            node_attributes[attributes_name] = attributes_value

    node_spec["Attributes"] = node_attributes
    node_spec["Inputs"] = node_input
    node_spec["Outputs"] = node_output
    # node_spec["Type"] = node_type
    return node_name, node_spec

def readOnnxNodeSpec():
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
                # print(node_name)
                # print(node_spec)
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
                        # print(node_name)
                        # print(node_spec)
                        spec = []
                else:
                    isStart = False
                    spec = []

            if isStart == True:
                spec.append(line)
    return node_dict

# for item in nodes_list:
#     if item == "Add":
#         print(f'Node: {item}')
#         node = onnx.helper.make_node(
#             item,
#             inputs=["x1"],
#             outputs=["y"],
#         )
#         print(f'Node: {item} is created OK')
#         x = createSampleData([1, 2, 3], np.float32)
#         y = x
#         status = expect(node, inputs=[x, x], outputs=[y], name="test_abs")
#         if status: print("Inferrence is OK")
#         else: print("Inferrence is FAILED")

        # for datatype in element_type:
        #     try:
        #         x = createSampleData([1, 2, 3], element_type[datatype])
        #         y = x
        #         status = expect(node, inputs=[x, x], outputs=[y], name="test_abs")
        #         if status: print("Inferrence is OK")
        #         else: print("Inferrence is FAILED")
        #     except:
        #         pass

def generate_TestCases_Combinations(config_values, currentindex, numberofconfig, combination, output_list, config_names):
    if currentindex < numberofconfig - 1:
        for item in config_values[currentindex]:
            combination.append(item)
            generate_TestCases_Combinations(config_values, currentindex + 1, numberofconfig, combination, output_list, config_names)
        if combination: combination.pop()
    else:
        for item in config_values[currentindex]:
            dict_combination = {}
            for i in range(0, len(config_names) - 1):
                dict_combination[config_names[i]] = combination[i]
            dict_combination[config_names[-1]] = item
            output_list.append(dict_combination)
        if combination: combination.pop()

# create input and output
def createOneTestCase(node_name, node, combination):
    node_input = []
    network_input = []
    node_output = []
    network_output = []
    status = 0
    try:
        for item in combination:
            if item in node["Inputs"]:
                node_input.append(item)
                network_input.append(createSampleData([1, 2, 3], combination[item]))
            if item in node["Outputs"]:
                node_output.append(item)
                network_output.append(createSampleData([1, 2, 3], combination[item]))
        node = onnx.helper.make_node(
            node_name,
            inputs=node_input,
            outputs=node_output,
        )
        
        status = expect(node, inputs=network_input, outputs=network_output, name="test_abs")
    except:
        pass

    if status == 1:
        print(combination)

node_dict = readOnnxNodeSpec()

for node in node_dict:
    config_names = []
    config_values = []
    if node == "Abs":
        for item in node_dict[node]["Inputs"]:
            config_names.append(item)
            config_values.append((node_dict[node]["Inputs"][item]).split(","))
        for item in node_dict[node]["Outputs"]:
            config_names.append(item)
            config_values.append((node_dict[node]["Outputs"][item]).split(","))

        combination = []
        output_list = []
        generate_TestCases_Combinations(config_values, 0, len(config_values), combination, output_list, config_names)
        for onenode in output_list:
            createOneTestCase(node, node_dict[node], onenode)

