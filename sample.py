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
from onnx.backend.test.case.node.batchnorm import _batchnorm_training_mode
import re
from ONMANode import ONMANode
import itertools

global args

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
        try:
            results = sess.run(None, feeds)
        except Exception as e:
            return 0
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

attributes = {
    "direction": ["RIGHT", "LEFT"],
    "to": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
    "kernel_shape": [[3, 3]],
    "axis": [0],
    "keepdims": [1],
    "select_last_index": [0],
    "saturate": [1],
    "alpha": [1.0]
}

input_special = {
    "AveragePool": {
        'Inputs': {
            'X': [1, 2, 3, 3]
        },
        'Outputs': {
            'Y': [1, 2, 1, 1]
        },
        'Attributes': {
            'auto_pad': ["NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID"],
            'ceil_mode': [0],
            'count_include_pad': [0],
            'dilations': [(1, 1)],
            'kernel_shape': [(3, 3)],
            'pads': [(0, 0, 0 ,0)],
            'strides': [(1, 1)]
        }
    },
    "BatchNormalization": {
        'Inputs': {
            'X': [2, 3, 4, 5],
            'scale': [3],
            'B': [3],
            'input_mean': [3],
            'input_var': [3]
        },
        'Outputs': {
            'Y': [2, 3, 4, 5],
            'running_mean(optional)': [1],
            'running_var(optional)': [1],
        },
        'Attributes': {
            'epsilon': [1e-05],
            'momentum': [0.9],
            'training_mode': [0, 1]
        }
    },
    "Clip": {
        'Inputs': {
            'input': [1, 2, 3, 3],
            'min(optional)': [1],
            'max(optional)': [1],
        },
        'Outputs': {
            'output': [1, 2, 3, 3]
        },
    },
    "Conv": {
        "list":
        [
            {
                'Inputs': {
                    'X': [1, 1, 5, 5],
                    'W': [1, 1, 3, 3],
                    'B(optional)': [1]
                },
                'Outputs': {
                    'Y': [1, 1, 3, 3]
                },
                'Attributes': {
                    'auto_pad': ["NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID"],
                    'dilations': [(1, 1)],
                    'kernel_shape': [(3, 3)],
                    'strides': [(1, 1)]
                }
            },
            {
                'Inputs': {
                    'X': [1, 1, 5, 5],
                    'W': [1, 1, 3, 3],
                    'B(optional)': [1]
                },
                'Outputs': {
                    'Y': [1, 1, 3, 3]
                },
                'Attributes': {
                    'dilations': [(1, 1)],
                    'kernel_shape': [(3, 3)],
                    'pads': [(0, 0, 0 ,0)],
                    'strides': [(1, 1)]
                }
            }
        ]
    },
    "Resize": {
        'Inputs': {
            'X': [1, 1, 4, 4],
            'roi': [1, 4],
            'sizes': [1, 2],
            'scales': [1, 4]
        },
        'Outputs': {
        }
    },
    "Bernoulli": {
        'Attributes': {
            "dtype": [1],
            "seed": [5]
        }
    }
}

def createSampleData(dimentions, datatype):
    if datatype == "unit8": return np.random.randn(*dimentions).astype(np.unit8)
    elif datatype == "uint16": return np.random.randn(*dimentions).astype(np.uint16)
    elif datatype == "uint32": return np.random.randn(*dimentions).astype(np.uint32)
    elif datatype == "uint64": return np.random.randn(*dimentions).astype(np.uint64)
    elif datatype == "int8": return np.random.randn(*dimentions).astype(np.int8)
    elif datatype == "int16": return np.random.randn(*dimentions).astype(np.int16)
    elif datatype == "int32": return np.random.randn(*dimentions).astype(np.int32)
    elif datatype == "int64": return np.random.randn(*dimentions).astype(np.int64)
    elif datatype == "float16": return np.random.randn(*dimentions).astype(np.float16)
    elif datatype == "float": return np.random.randn(*dimentions).astype(np.float32)
    elif datatype == "double": return np.random.randn(*dimentions).astype(np.double)
    else: return np.random.randn(*dimentions).astype(datatype)

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
def checkCombination(node_name, node, combination, one_special_input):
    node_input = []
    network_input = []
    node_output = []
    network_output = []
    attri = {}
    status = 0
    try:
        for item in combination:
            # if node_name in input_special:
            try:
                if item in one_special_input["Inputs"]:
                    node_input.append(item)
                    x = createSampleData(one_special_input["Inputs"][item], combination[item])
                    x[x == 0.0] = 0.5
                    x[x == 0] = 1
                    network_input.append(x)
            except:
                if item in node["Inputs"]:
                    node_input.append(item)
                    x = createSampleData([1, 2, 3, 3], combination[item])
                    x[x == 0.0] = 0.5
                    x[x == 0] = 1
                    network_input.append(x)
            
            try:
                if item in one_special_input["Outputs"]:
                    node_output.append(item)
                    x = createSampleData(one_special_input["Outputs"][item], combination[item])
                    x[x == 0.0] = 0.5
                    x[x == 0] = 1
                    network_output.append(x)
            except:
                if item in node["Outputs"]:
                    node_output.append(item)
                    x = createSampleData([1, 2, 3, 3], combination[item])
                    x[x == 0.0] = 0.5
                    x[x == 0] = 1
                    network_output.append(x)
            
            if item in node["Attributes"]:
                attri[item] = combination[item]

        onma_node = ONMANode()
        onma_node.ONMANode_MakeNode(
            node_name, inputs=node_input, outputs=node_output, **attri
        )
        status = expect(onma_node.ONMANode_GetNode(), inputs=network_input, outputs=network_output, name="test_abs")
    except:
        pass

    return status

def create_TCs_for_one_input_setting(node, node_data, one_special_input):
    config_names = []
    config_values = []
    config_names_option = []
    config_values_option = []
    
    for item in node_data["Inputs"]:
        if "(optional)" in item:
            config_names_option.append(item)
            config_values_option.append((node_data["Inputs"][item]).split(","))
        else:
            config_names.append(item)
            config_values.append((node_data["Inputs"][item]).split(","))
    for item in node_data["Outputs"]:
        if "(optional)" in item:
            config_names_option.append(item)
            config_values_option.append((node_data["Outputs"][item]).split(","))
        else:
            config_names.append(item)
            config_values.append((node_data["Outputs"][item]).split(","))
    for item in node_data["Attributes"]:
        try:
            if "(optional)" in item:
                reitem = item.replace("(optional)", "")
                try:
                    config_values_option.append(one_special_input["Attributes"][reitem])
                except:
                    config_values_option.append(attributes[reitem])
                config_names_option.append(reitem)
            else:
                try:
                    config_values.append(one_special_input["Attributes"][item])
                except:
                    config_values.append(attributes[item])
                config_names.append(item)
        except:
            pass

    # print(config_names)
    # print(config_values)
    # print(config_names_option)
    # print(config_values_option)

    full_config_names = []
    full_config_values = []
    full_config_names = full_config_names + config_names
    full_config_values = full_config_values + config_values
    # print(f'Config name: {full_config_names}')
    # print(f'Config value: {full_config_values}')

    combination = []
    output_list = []
    try:
        generate_TestCases_Combinations(full_config_values, 0, len(full_config_values), combination, output_list, full_config_names)
        for onenode in output_list:
            status = checkCombination(node, node_data, onenode, one_special_input)
            if status == 1: print(onenode)
    except:
        pass

    for i in range(0, len(config_names_option)):
        options_combination = list(itertools.combinations(config_names_option, i + 1))
        for item in options_combination:
            renew_config_names_option = []
            renew_config_values_option = []
            reitem = str(item)
            reitem = reitem.replace("('", "'")
            reitem = reitem.replace(",)", "")
            reitem = reitem.replace("')", "'")
            reitem = reitem.replace("'", "")
            reitem = reitem.split(", ")
            for new_option in reitem:
                renew_config_names_option.append(new_option)
                for j in range(0, len(config_names_option)):
                    if new_option == config_names_option[j]:
                        renew_config_values_option.append(config_values_option[j])
            # print(renew_config_names_option)
            # print(renew_config_values_option)

            full_config_names = []
            full_config_values = []
            full_config_names = full_config_names + config_names
            full_config_names = full_config_names + renew_config_names_option
            full_config_values = full_config_values + config_values
            full_config_values = full_config_values + renew_config_values_option
            # print(f'Config name: {full_config_names}')
            # print(f'Config value: {full_config_values}')

            combination = []
            output_list = []                    
            try:
                generate_TestCases_Combinations(full_config_values, 0, len(full_config_values), combination, output_list, full_config_names)
                for onenode in output_list:
                    status = checkCombination(node, node_data, onenode, one_special_input)
                    if status == 1: print(onenode)
            except:
                pass

def createTC(node_dict, node_name):
    for node in node_dict:
        if (node == node_name or node_name == "ALL") and node != "LSTM":
            print(f'======{node}======')
            one_special_input = {}
            node_data = node_dict[node]
            try:
                one_special_input = input_special[node]
                if "list" in one_special_input: # "list" inside
                    for item in one_special_input["list"]:
                        create_TCs_for_one_input_setting(node, node_data, item)
                else:
                    create_TCs_for_one_input_setting(node, node_data, one_special_input)
            except:
                create_TCs_for_one_input_setting(node, node_data, one_special_input)

def main():
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", "-op", help="Operator name", default="")
    args = parser.parse_args()

    node_dict = readOnnxNodeSpec()
    createTC(node_dict, args.operator)

if main() == False:
    sys.exit(-1)
