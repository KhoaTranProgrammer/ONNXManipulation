import numpy as np
import onnx
import json
import ast
from onnx.helper import (
    make_node, make_graph, make_model, make_tensor_value_info)
from onnx.numpy_helper import from_array
from onnx.checker import check_model
from onnxruntime import InferenceSession
from ONMA.ONMANode import ONMANode
from ONMA.ONMAGraph_Internal import \
    NumpyDataTypeFromTensor, GetTensorDataTypeFromnp, CreateInitializerTensor,  \
    GetInitializerByName, GetPreviousNodeFromInputName, NextNode, \
    checkOneCondition, checkConditionOfSearchBy, UpdateGraphUsingPattern

def UpdateInitializer(initializers, name, data):
    # Get the initializer by name
    index, initializer = GetInitializerByName(initializers, name)

    if initializer != None:
        # Get action: Add/Remove/Modify
        action = data["Action"]
        
        if action == "Modify" or action == "Add":
            values = np.array(data["tensor"]["data"], dtype=data["tensor"]["type"])

            # Create new initializer with new value
            new_initializer = CreateInitializerTensor(
                        name=initializer.name,
                        tensor_array=values,
                        data_type=GetTensorDataTypeFromnp(values.dtype))
            initializers.pop(index)
            initializers.insert(index, new_initializer)
        elif action == "Remove":
            initializers.remove(initializer)
        else:
            pass
    else:
        if "Action" not in data or data["Action"] == "Add":
            values = np.array(data["data"], dtype=data["data_type"])
            # Create new initializer with new value
            new_initializer = CreateInitializerTensor(
                        name=name,
                        tensor_array=values,
                        data_type=GetTensorDataTypeFromnp(values.dtype))
            initializers.append(new_initializer)

def UpdateNode(graph, name, data):
    if "Action" not in data or data["Action"] == "Add":
        op_type = data["op_type"]
        inputs=data["inputs"]
        outputs=data["outputs"]

        node_ind,_ = GetPreviousNodeFromInputName(graph, inputs)

        try:
            if "values" in data:
                if type(data["values"]) is list:
                    data["values"] = np.array(data["values"])
        except:
            pass

        onma_node = ONMANode()
        try:
            onma_node.ONMANode_MakeNode(
                op_type, inputs=inputs, outputs=outputs, name=name, **data["attributes"]
            )
        except:
            onma_node.ONMANode_MakeNode(
                op_type, inputs=inputs, outputs=outputs, name=name
            )

        if len(node_ind) == 0:
            graph.node.append(onma_node.ONMANode_GetNode())
        else:
            index = 0
            for i in range(0, len(node_ind)):
                if index < node_ind[i]:
                    index = node_ind[i]
            graph.node.insert(index + 1, onma_node.ONMANode_GetNode())
    elif data["Action"] == "Modify":
        for i, node in enumerate(graph.node):
            if node.name == name:
                onma_node = ONMANode()
                
                # Decide input
                try:
                    if data["inputs"]: inputs = data["inputs"]
                except:
                    inputs = node.input

                # Decide output
                try:
                    if data["outputs"]: outputs = data["outputs"]
                except:
                    outputs = node.output

                onma_node.ONMANode_MakeNode(
                    node.op_type, inputs=inputs, outputs=outputs, name=name
                )

                for attribute in node.attribute:
                    isReplace = False
                    for item in data:
                        if item == attribute.name:
                            isReplace = True
                            new_attribute = onnx.helper.make_attribute(item, data[item])
                            onma_node.ONMANode_GetNode().attribute.append(new_attribute)
                    if isReplace == False:
                        onma_node.ONMANode_GetNode().attribute.append(attribute)

                graph.node.pop(i)
                graph.node.insert(i, onma_node.ONMANode_GetNode())
    elif data["Action"] == "Remove":
        for i, node in enumerate(graph.node):
            if node.name == name:
                graph.node.pop(i)

def GetAtrributeValue(attribute):
    result = []
    items = str(attribute).split("\n")
    items.pop(0)
    items.pop(-1)
    type = items[-1]
    items.pop(-1)
    for item in items:
        data = item.split(": ")
        if "INT" in type: data = int(data[1])
        elif "FLOAT" in type: data = float(data[1])
        elif "STRING" in type:
            data = str(data[1])
            data = data.replace("\"", "")
        else: data = data[1]
        result.append(data)
    if len(result) == 1: return result[0]
    else: return result

def ConvertONNXToJson(graph, output_path, store_npy=False):
    graph_dictionary = {}
    inputs = []
    outputs = []
    nodes = []

    graph_dictionary["name"] = graph.name
    graph_dictionary["graph"] = {}

    # Create initializers
    initializer_dic_array = []
    for initializer in graph.initializer:
        init_dic = {}

        data = onnx.numpy_helper.to_array(initializer)
        if store_npy:
            npy_data = {}
            npy_data["npy"] = f'{initializer.name}.npy'
            npy_data["npy"] = (npy_data["npy"]).replace("::", "_")
            np.save(npy_data["npy"], data)
            init_dic["data"] = npy_data
        else:
            data = onnx.numpy_helper.to_array(initializer)
            init_dic["data"] = data.tolist()
        init_dic["name"] = initializer.name
        init_dic["shape"] = data.shape
        init_dic["data_type"] = str(data.dtype)

        initializer_dic_array.append(init_dic)

    if initializer_dic_array != []:
        graph_dictionary["graph"]["initializers"] = initializer_dic_array

    for input in graph.input:
        input_shape = []
        for d in input.type.tensor_type.shape.dim:
            if d.dim_value == 0:
                input_shape.append(None)
            else:
                input_shape.append(d.dim_value)
        input_dic = {}
        input_dic["name"] = input.name
        input_dic["shape"] = input_shape
        input_dic["data_type"] = NumpyDataTypeFromTensor(input.type.tensor_type.elem_type)
        inputs.append(input_dic)

    if inputs != []:
        graph_dictionary["graph"]["inputs"] = inputs

    for output in graph.output:
        output_shape = []
        for d in output.type.tensor_type.shape.dim:
            if d.dim_value == 0:
                output_shape.append(None)
            else:
                output_shape.append(d.dim_value)
        output_dic = {}
        output_dic["name"] = output.name
        output_dic["shape"] = output_shape
        output_dic["data_type"] = NumpyDataTypeFromTensor(output.type.tensor_type.elem_type)
        outputs.append(output_dic)

    if outputs != []:
        graph_dictionary["graph"]["outputs"] = outputs

    for node in graph.node:
        node_dic = {}
        node_dic["name"] = node.name
        node_dic["op_type"] = node.op_type

        input_dic = []
        for nodeinput in node.input:
            input_dic.append(nodeinput)

        output_dic = []
        for nodeoutput in node.output:
            output_dic.append(nodeoutput)

        node_dic["inputs"] = input_dic
        node_dic["outputs"] = output_dic

        if node.op_type == "Constant":
            items = str((node.attribute[0]).t).split("\n")

            dimension = []
            data_type = 1
            name = ""
            data_list = []
            for item in items:
                item_list = str(item).split(": ")
                if 'dims' in item:
                    value = int(item_list[1])
                    dimension.append(value)
                elif 'data_type' in item:
                    data_type = int(item_list[1])
                elif 'name' in item:
                    name = item_list[1]
                else:
                    if item != "":
                        data_list.append(item_list[1])

            data_list = np.array(data_list, dtype=NumpyDataTypeFromTensor(data_type))
            data_list = data_list.reshape(dimension)
            node_dic["values"] = data_list.tolist()
        else:
            attributes = {}
            for attribute in node.attribute:
                result = GetAtrributeValue(attribute)
                attributes[attribute.name] = result
            if attributes != {}:
                node_dic["attributes"] = attributes

        nodes.append(node_dic)

    graph_dictionary["graph"]["nodes"] = nodes

    with open(output_path, "w") as fp:
        json.dump(graph_dictionary, fp, indent = 4)

class ONMAGraph:
    def __init__(self):
        self._graph = None

    def ONMAGraph_MakeGraph(self, name, nodes, inputs, outputs):
        self._graph = onnx.helper.make_graph(nodes, name, inputs, outputs)
    
    def ONMAGraph_GetGraph(self):
        return self._graph
    
    def ONMAGraph_SetGraph(self, graph):
        self._graph = graph

    def ONMAGraph_CreateInput(self, name, type, dimension):
        return make_tensor_value_info(name, type, dimension)

    def ONMAModel_UpdateGraph(self, data):
        if "graph" in data:
            if "initializers" in data["graph"]:
                for initializer in data["graph"]["initializers"]:
                    if "data" in initializer:
                        try:
                            initializer["data"] = np.array(initializer["data"], dtype=initializer["data_type"])
                        except:
                            if "npy" in initializer["data"]:
                                data_fromnpy = np.load(initializer["data"]["npy"], allow_pickle=True)
                                initializer["data"] = data_fromnpy.tolist()
                            else: # "data": "np.random.rand(1, 2, 16, 16).astype(np.float32)"
                                initializer["data"] = eval(initializer["data"])
                                initializer["data_type"] = (str((initializer["data"]).dtype))
                    else:
                        initializer["data"] = np.random.randn(*initializer["shape"]).astype(initializer["data_type"])
                    UpdateInitializer(self._graph.initializer, initializer["name"], initializer)

            if "nodes" in data["graph"]:
                for node in data["graph"]["nodes"]:
                    UpdateNode(self._graph, node["name"], node)
        else:
            for item in data:
                if "SearchBy" in data[item] and "ReplaceBy" in data[item]:
                    decompose_pattern = UpdateGraphUsingPattern(self._graph, data[item])
                    if decompose_pattern != None:
                        return self.ONMAModel_UpdateGraph(decompose_pattern)
                    else:
                        return False
        
        return True

    def ONMAGraph_CreateNetworkFromGraph(self, data):
        # Remove empty input
        refine_input = {}
        inputs = data["graph"]["inputs"]
        for oneInput in inputs:
            if "data" in oneInput:
                try:
                    refine_input[oneInput["name"]] = np.array(oneInput["data"], dtype=oneInput["data_type"])
                except:
                    pass
            else:
                try:
                    refine_input[oneInput["name"]] = np.random.randn(*oneInput["shape"]).astype(oneInput["data_type"])
                except:
                    pass

        # Create graph input
        graph_input = []
        for i in range(0, len(list(refine_input.keys()))):
            graph_input.append(self.ONMAGraph_CreateInput(list(refine_input.keys())[i], GetTensorDataTypeFromnp((list(refine_input.values())[i]).dtype), (list(refine_input.values())[i]).shape))

        # Create graph output
        outputs = data["graph"]["outputs"]
        graph_output = []
        for item in outputs:
            if "shape" not in item and "data" not in item:
                graph_output.append(self.ONMAGraph_CreateInput(item["name"], GetTensorDataTypeFromnp(np.array((list(refine_input.values())[0])).dtype), np.array(list(refine_input.values())[0]).shape))
            else:
                graph_output.append(self.ONMAGraph_CreateInput(item["name"], GetTensorDataTypeFromnp(item["data_type"]), item["shape"]))

        self.ONMAGraph_MakeGraph(data["name"], [], graph_input, graph_output)

        self.ONMAModel_UpdateGraph(data)

    def ONMAGraph_UpdateOutputDimension(self, output_value):
        output_name = []
        for one_output in self.ONMAGraph_GetGraph().output:
            output_name.append(one_output.name)

        for i in range(0, len(self.ONMAGraph_GetGraph().output)):
            self.ONMAGraph_GetGraph().output.pop(i)

        for i, item in enumerate(output_name):
            self.ONMAGraph_GetGraph().output.append(self.ONMAGraph_CreateInput(item, GetTensorDataTypeFromnp(output_value[i].dtype), output_value[i].shape))

    def ONMAModel_ConvertONNXToJson(self, output_path, store_npy=False):
        ConvertONNXToJson(self.ONMAGraph_GetGraph(), output_path, store_npy)
