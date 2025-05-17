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
from onnx.backend.test.case.node import _extract_value_info
from onnx.backend.test.case.node.affinegrid import create_theta_2d
from onnx.backend.test.case.node.roialign import get_roi_align_input_values
from onnx.backend.test.case.node.layernormalization import calculate_normalized_shape
from onnx.backend.test.case.node.batchnorm import _batchnorm_training_mode

file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = Path(file_path).as_posix()
sys.path.append(file_path)
from ONMA.ONMANode import ONMANode

"""
Purpose: this script will generate test spec from onnx operators spec.

Rule for generation:
- Generate all of possible combinations from Inputs/Outputs/Attributes
- The test cases those can be successfully run in onnxruntime will be saved in test spec json

Example:
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

- Possible test cases
================================
{'X': 'uint8', 'Y': 'uint8'}
{'X': 'uint16', 'Y': 'uint16'}
{'X': 'uint32', 'Y': 'uint32'}
{'X': 'uint64', 'Y': 'uint64'}
{'X': 'int8', 'Y': 'int8'}
{'X': 'int16', 'Y': 'int16'}
{'X': 'int32', 'Y': 'int32'}
{'X': 'int64', 'Y': 'int64'}
{'X': 'float16', 'Y': 'float16'}
{'X': 'float', 'Y': 'float'}
{'X': 'double', 'Y': 'double'}
================================

Usage:
In order to run this script, user need to prepare below items:
- ONNX spec in JSON format: this file can be generated using script ONMAOperatorsSpec.py
  + Sample file: Sample/ONNXOperatorsSpec.json
- JSON format that stores the list of attributes and there value uses in testing
  + Sample file: Sample/Attributes_Values.json
- JSON format that stores the list of input for nodes
  + Sample file: Sample/Input_Special.json
- JSON file that stores the test spec
  + Sample file: Tests/reference_data.json

Run this script by command: Tools/ONMACreateTestSpec.py
Example version of test spec is put at: Tests/reference_data.json
"""

exclude_node = []

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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
    if "" in node.input: node.input.remove("")
    try:
        sess = onnxruntime.InferenceSession(model.SerializeToString(),
                                            providers=["CPUExecutionProvider"])
        feeds = {name: value for name, value in zip(node.input, inputs)}
        try:
            results = sess.run(None, feeds)
        except Exception as e:
            return 0
        return 1, results
    except:
        return 0

def find_index(output_dictionary, node):
    for i in range(1000):
        TC_Name = f'{node}_{i:03}'
        if TC_Name not in output_dictionary:
            return TC_Name

def convert2Dictionary(node_name, node_input, node_output, node_attri, network_input, network_output):
    graph_dictionary = {}
    inputs = {}
    outputs = {}
    node_data = {}
    node_data["Action"] = "Add"
    node_data["Category"] = "Node"
    node_data["Type"] = node_name
    node_data_input = {}
    node_data_output = {}

    # Get node input
    for i in range(0, len(node_input)):
        one_input = {}
        input_name = node_input[i] # X
        node_data_input[node_input[i]] = node_input[i]
    
    # Get network input
    node_input_refine = node_input
    if "" in node_input_refine:
        node_input_refine.remove("") # Remove empty input

    for i in range(0, len(node_input_refine)):
        one_input = {}
        input_name = node_input_refine[i] # X
        data = network_input[i]
        one_input["data"] = data
        one_input["type"] = str(data.dtype)
        inputs[input_name] = one_input

    for i in range(0, len(node_output)):
        one_output = {}
        output_name = node_output[i] # X
        node_data_output[node_output[i]] = node_output[i]
        data = network_output[i]
        one_output["data"] = data
        one_output["type"] = str(data.dtype)
        outputs[output_name] = one_output
    
    for item in node_attri:
        node_data[item] = node_attri[item]

    node_data["inputs"] = node_data_input
    node_data["outputs"] = node_data_output

    graph_dictionary["graph_name"] = f'{node_name}_Graph'
    graph_dictionary["inputs"] = inputs
    graph_dictionary["outputs"] = outputs
    graph_dictionary[f'{node_name}_Node'] = node_data
    
    return graph_dictionary

def createSampleData(dimentions, datatype, min=None, max=None):
    if datatype == "unit8": x = np.random.randn(*dimentions).astype(np.unit8)
    elif datatype == "uint16": x = np.random.randn(*dimentions).astype(np.uint16)
    elif datatype == "uint32": x = np.random.randn(*dimentions).astype(np.uint32)
    elif datatype == "uint64": x = np.random.randn(*dimentions).astype(np.uint64)
    elif datatype == "int8": x = np.random.randn(*dimentions).astype(np.int8)
    elif datatype == "int16": x = np.random.randn(*dimentions).astype(np.int16)
    elif datatype == "int32": x = np.random.randn(*dimentions).astype(np.int32)
    elif datatype == "int64": x = np.random.randn(*dimentions).astype(np.int64)
    elif datatype == "float16": x = np.random.randn(*dimentions).astype(np.float16)
    elif datatype == "float": x = np.random.randn(*dimentions).astype(np.float32)
    elif datatype == "double": x = np.random.randn(*dimentions).astype(np.double)
    elif datatype == "bool": x = np.random.randn(*dimentions).astype(bool)
    else: x = np.random.randn(*dimentions).astype(datatype)

    if min != None:
        x[x < min] = min
    if max != None:
        x[x > max] = max
    return x

def createArrayFromData(data, datatype):
    if datatype == "string": return np.array(data).astype(object)
    elif datatype == "float": return np.array(data).astype(np.float32)
    elif datatype == "int64": return np.array(data).astype(np.int64)
    else: return np.array(data).astype(datatype)

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

def checkCombination(node_name, node, combination, one_special_input):
    node_input = []
    network_input = []
    node_output = []
    network_output = []
    attri = {}
    status = 0
    try:
        for item in combination:
            try:
                if item in one_special_input["Inputs"]:
                    node_input.append(item)
                    try:
                        if "dimension" in one_special_input["Inputs"][item]:
                            min = None
                            max = None

                            try:
                                min = one_special_input["Inputs"][item]["min"]
                            except:
                                pass

                            try:
                                max = one_special_input["Inputs"][item]["max"]
                            except:
                                pass

                            x = createSampleData(one_special_input["Inputs"][item]["dimension"], combination[item], min=min, max=max)
                            x[x == 0.0] = 0.5
                            x[x == 0] = 1
                        else:
                            x = createArrayFromData(one_special_input["Inputs"][item], combination[item])
                    except:
                        x = createArrayFromData(one_special_input["Inputs"][item], combination[item])
                    network_input.append(x)
                else: # Input from test spec is not available in specific input or node input
                    if item in node["Inputs"]:
                        node_input = []
                        network_input = []
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
                    if "dimension" in one_special_input["Outputs"][item]:
                        x = createSampleData(one_special_input["Outputs"][item]["dimension"], combination[item])
                    else:
                        x = createArrayFromData(one_special_input["Outputs"][item], combination[item])
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

        # Process for empty input
        if one_special_input != {}:
            for index, item in enumerate(one_special_input["Inputs"]):
                if item == "": node_input.insert(index, item)

        onma_node = ONMANode()
        onma_node.ONMANode_MakeNode(
            node_name, inputs=node_input, outputs=node_output, name="Sample_Node", **attri
        )
        status, result = expect(onma_node.ONMANode_GetNode(), inputs=network_input, outputs=network_output, name="test_sample")
    except:
        pass

    graph_dictionary = {}
    if status == 1:
        print(f"{combination}: Successfully")
        graph_dictionary = convert2Dictionary(node_name, node_input, node_output, attri, network_input, result)
    return status, graph_dictionary

def create_TCs_for_one_input_setting(node, node_data, one_special_input, output_dictionary, attributes):
    result = False
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
            if "(variadic)" in item:
                for oneoutput in one_special_input["Outputs"]:
                    if item in oneoutput:
                        config_names.append(oneoutput)
                        config_values.append((node_data["Outputs"][item]).split(","))
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

    full_config_names = []
    full_config_values = []
    full_config_names = full_config_names + config_names
    full_config_values = full_config_values + config_values

    combination = []
    output_list = []
    try:
        generate_TestCases_Combinations(full_config_values, 0, len(full_config_values), combination, output_list, full_config_names)
        for onenode in output_list:
            status, graph_dictionary = checkCombination(node, node_data, onenode, one_special_input)
            if status == 1:
                result = True
                TC_name = find_index(output_dictionary, node)
                output_dictionary[TC_name] = graph_dictionary
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

            full_config_names = []
            full_config_values = []
            full_config_names = full_config_names + config_names
            full_config_names = full_config_names + renew_config_names_option
            full_config_values = full_config_values + config_values
            full_config_values = full_config_values + renew_config_values_option

            combination = []
            output_list = []                    
            try:
                generate_TestCases_Combinations(full_config_values, 0, len(full_config_values), combination, output_list, full_config_names)
                for onenode in output_list:
                    status, graph_dictionary = checkCombination(node, node_data, onenode, one_special_input)
                    if status == 1:
                        result = True
                        TC_name = find_index(output_dictionary, node)
                        output_dictionary[TC_name] = graph_dictionary
            except:
                pass
    return result

def createTC(node_dict, node_name, input_special, attributes, test_spec):
    global exclude_node
    output_dictionary = {}
    for node in node_dict:
        status = False
        if (node == node_name or node_name == "ALL") and node != "Attention" and node != "LSTM" and node != "Bernoulli" \
            and node != "Dropout" and node != "LeakyRelu" and node != "Mish" and node != "PRelu" and node != "RandomNormalLike" \
            and node != "ReduceLogSumExp" and node != "RandomUniformLike"  and node != "ConvTranspose" \
            and node != "LpNormalization" and node != "RoiAlign" and node != "Shrink" and node != "NonMaxSuppression":
            print(f'======{node}======')
            one_special_input = {}
            node_data = node_dict[node]
            try:
                one_special_input = input_special[node]
                if "list" in one_special_input: # "list" inside
                    for item in one_special_input["list"]:
                        status = create_TCs_for_one_input_setting(node, node_data, item, output_dictionary, attributes)
                else:
                    status = create_TCs_for_one_input_setting(node, node_data, one_special_input, output_dictionary, attributes)
            except:
                status = create_TCs_for_one_input_setting(node, node_data, one_special_input, output_dictionary, attributes)
        if status == False:
            exclude_node.append(node)

    with open(test_spec, "w") as fp:
        json.dump(output_dictionary, fp, indent = 4, cls=NumpyEncoder)

def main():
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", "-op", help="Operator name", default="ALL")
    parser.add_argument("--onnxspec", help="ONNX spec in JSON format", default="Sample/ONNXOperatorsSpec.json")
    parser.add_argument("--attributes_list", help="JSON format that stores the list of attributes and there value uses in testing", default="Sample/Attributes_Values.json")
    parser.add_argument("--input_special", help="JSON format that stores the list of input for nodes", default="Sample/Input_Special.json")
    parser.add_argument("--test_spec", help="JSON file that stores the test spec", default="Tests/reference_data.json")
    args = parser.parse_args()

    # Open and read the JSON file
    with open(args.onnxspec, 'r') as file:
        node_dict = json.load(file)

    with open(args.attributes_list, 'r') as file:
        attributes = json.load(file)

    with open(args.input_special, 'r') as file:
        input_special = json.load(file)
        try:
            for item in input_special["IsInf"]["Inputs"]:
                refine_input = []
                for one in input_special["IsInf"]["Inputs"][item]:
                    if one == "np.nan":  refine_input.append(np.nan)
                    elif one == "np.inf":  refine_input.append(np.inf)
                    elif one == "-np.inf":  refine_input.append(-np.inf)
                    else: refine_input.append(one)
                input_special["IsInf"]["Inputs"][item] = refine_input
        except:
            pass

    createTC(node_dict, args.operator, input_special, attributes, args.test_spec)
    with open('Sample/Exclude_Node.txt', 'w') as filehandle:
        for line in exclude_node:
            print("{}".format(line), file=filehandle)

if main() == False:
    sys.exit(-1)
