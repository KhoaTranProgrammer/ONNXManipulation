import numpy as np
import onnx
import json
from onnx.helper import (
    make_node, make_graph, make_model, make_tensor_value_info)
from onnx.numpy_helper import from_array
from onnx.checker import check_model
from onnxruntime import InferenceSession
from ONMA.ONMANode import ONMANode

def NumpyDataTypeFromTensor(type):
    if type == 1: return "float32"
    elif type == 2: return "uint8"
    elif type == 3: return "int8"
    elif type == 4: return "uint16"
    elif type == 5: return "int16"
    elif type == 6: return "int32"
    elif type == 7: return "int64"
    elif type == 8: return "object"
    elif type == 9: return "bool"
    elif type == 10: return "float16"
    elif type == 11: return "float64"
    elif type == 12: return "uint32"
    elif type == 13: return "uint64"
    else: return "float32"

def GetTensorDataTypeFromnp(npdtype):
    datatype = onnx.TensorProto.FLOAT
    if npdtype == "float32":
        datatype = onnx.TensorProto.FLOAT
    elif npdtype == "float64":
        datatype = onnx.TensorProto.DOUBLE
    elif npdtype == "int64":
        datatype = onnx.TensorProto.INT64
    elif npdtype == "uint16":
        datatype = onnx.TensorProto.UINT16
    elif npdtype == "bool":
        datatype = onnx.TensorProto.BOOL
    elif npdtype == "uint8":
        datatype = onnx.TensorProto.UINT8
    elif npdtype == "int32":
        datatype = onnx.TensorProto.INT32
    elif npdtype == "object":
        datatype = onnx.TensorProto.STRING
    elif npdtype == "uint32":
        datatype = onnx.TensorProto.UINT32
    elif npdtype == "uint64":
        datatype = onnx.TensorProto.UINT64
    elif npdtype == "int8":
        datatype = onnx.TensorProto.INT8
    elif npdtype == "int16":
        datatype = onnx.TensorProto.INT16
    elif npdtype == "float16":
        datatype = onnx.TensorProto.FLOAT16
    return datatype

def CreateInitializerTensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist()
    )
    return initializer_tensor

def GetInitializerByName(initializers, name):
    result = None
    index = -1
    for i, initializer in enumerate(initializers):
        if initializer.name == name:
            result = initializer
            index = i
    return index, result

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
        action = data["Action"]
        if action == "Add":
            values = np.array(data["tensor"]["data"], dtype=data["tensor"]["type"])
            # Create new initializer with new value
            new_initializer = CreateInitializerTensor(
                        name=name,
                        tensor_array=values,
                        data_type=GetTensorDataTypeFromnp(values.dtype))
            initializers.append(new_initializer)

def GetPreviousNodeFromInputName(graph, inputs):
    node_res = []
    node_ind = []
    for i, node in enumerate(graph.node):
        for input in inputs:
            if inputs[input] in node.output:
                node_res.append(node)
                node_ind.append(i)
    return node_ind, node_res

def UpdateNode(graph, name, data):
    if data["Action"] == "Add":
        op_type = data["Type"]
        inputs=list((data["inputs"]).values())
        outputs=list((data["outputs"]).values())

        node_ind,_ = GetPreviousNodeFromInputName(graph, data["inputs"])

        # Simplify data
        data.pop("Action")
        data.pop("Category")
        data.pop("Type")
        data.pop("inputs")
        data.pop("outputs")
        
        onma_node = ONMANode()
        onma_node.ONMANode_MakeNode(
            op_type, inputs=inputs, outputs=outputs, name=name, **data
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
                    if data["inputs"]: inputs = list((data["inputs"]).values())
                except:
                    inputs = node.input

                # Decide output
                try:
                    if data["outputs"]: outputs = list((data["outputs"]).values())
                except:
                    outputs = node.output

                onma_node.ONMANode_MakeNode(
                    data["Type"], inputs=inputs, outputs=outputs, name=name
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

def ConvertONNXToJson(graph, output_path):
    graph_dictionary = {}
    inputs = {}
    outputs = {}

    graph_dictionary["graph_name"] = graph.name

    for initializer in graph.initializer:
        init_dic = {}
        init_dic["Action"] = "Add"
        init_dic["Category"] = "Initializer"

        tensor_dic = {}
        data = onnx.numpy_helper.to_array(initializer)
        tensor_dic["data"] = data.tolist()
        tensor_dic["type"] = str(data.dtype)

        init_dic["tensor"] = tensor_dic

        graph_dictionary[initializer.name] = init_dic

    for input in graph.input:
        input_shape = []
        for d in input.type.tensor_type.shape.dim:
            if d.dim_value == 0:
                input_shape.append(None)
            else:
                input_shape.append(d.dim_value)
        input_dic = {}
        input_dic["data"] = {"dimensions": input_shape}
        input_dic["type"] = NumpyDataTypeFromTensor(input.type.tensor_type.elem_type)
        inputs[input.name] = input_dic

    for output in graph.output:
        output_shape = []
        for d in output.type.tensor_type.shape.dim:
            if d.dim_value == 0:
                output_shape.append(None)
            else:
                output_shape.append(d.dim_value)
        output_dic = {}
        output_dic["data"] = {"dimensions": output_shape}
        output_dic["type"] = NumpyDataTypeFromTensor(output.type.tensor_type.elem_type)
        outputs[output.name] = output_dic
    
    graph_dictionary["inputs"] = inputs
    graph_dictionary["outputs"] = outputs

    for node in graph.node:
        node_dic = {}
        node_dic["Action"] = "Add"
        node_dic["Category"] = "Node"
        node_dic["Type"] = node.op_type

        input_dic = {}
        for nodeinput in node.input:
            inputname = nodeinput
            index = 0
            while inputname in input_dic:
                inputname = f'{nodeinput}_{index}'
                index = index + 1
            input_dic[inputname] = nodeinput

        output_dic = {}
        for nodeoutput in node.output:
            output_dic[nodeoutput] = nodeoutput

        node_dic["inputs"] = input_dic
        node_dic["outputs"] = output_dic

        for attribute in node.attribute:
            result = GetAtrributeValue(attribute)
            node_dic[attribute.name] = result

        graph_dictionary[node.name] = node_dic

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
        for item in data:
            try:
                if data[item]["Category"] == "Initializer":
                    if type(data[item]["tensor"]) is list:
                        data[item]["tensor"] = np.array(data[item]["tensor"])
                    UpdateInitializer(self._graph.initializer, item, data[item])
                elif data[item]["Category"] == "Node":
                    UpdateNode(self._graph, item, data[item])
            except:
                pass

    def ONMAGraph_CreateNetworkFromGraph(self, data):
        # Remove empty input
        refine_input = {}
        inputs = data["inputs"]
        for key, value in inputs.items():
            if key != "":
                if "dimensions" in value["data"]:
                    dimensions = value["data"]["dimensions"]
                    try:
                        value_np = np.random.randn(*dimensions).astype(value["data"]["type"])
                    except:
                        value_np = np.random.randn(*dimensions).astype("float32")
                else:
                    try:
                        value_np = np.array(value["data"], dtype=value["type"])
                    except:
                        value_np = np.array(value["data"], dtype='float32')
                refine_input[key] = value_np
        
        # Create graph input
        graph_input = []
        for i in range(0, len(list(refine_input.keys()))):
            graph_input.append(self.ONMAGraph_CreateInput(list(refine_input.keys())[i], GetTensorDataTypeFromnp((list(refine_input.values())[i]).dtype), (list(refine_input.values())[i]).shape))

        # Create graph output
        outputs = data["outputs"]
        graph_output = []
        for item in outputs:
            if outputs[item] == None or outputs[item] == "None":
                graph_output.append(self.ONMAGraph_CreateInput(item, GetTensorDataTypeFromnp(np.array((list(refine_input.values())[0])).dtype), np.array(list(refine_input.values())[0]).shape))
            else:
                try:
                    dimensions = outputs[item]["data"]["dimensions"]
                    try:
                        graph_output.append(self.ONMAGraph_CreateInput(item, GetTensorDataTypeFromnp(outputs[item]["type"]), dimensions))
                    except:
                        graph_output.append(self.ONMAGraph_CreateInput(item, GetTensorDataTypeFromnp("float32"), dimensions))
                except:
                    try:
                        graph_output.append(self.ONMAGraph_CreateInput(item, GetTensorDataTypeFromnp(outputs[item]["type"]), np.array(outputs[item]["data"]).shape))
                    except:
                        graph_output.append(self.ONMAGraph_CreateInput(item, GetTensorDataTypeFromnp("float32"), np.array(outputs[item]["data"]).shape))

        self.ONMAGraph_MakeGraph(data["graph_name"], [], graph_input, graph_output)

        self.ONMAModel_UpdateGraph(data)

    def ONMAGraph_UpdateOutputDimension(self, output_value):
        output_name = []
        for one_output in self.ONMAGraph_GetGraph().output:
            output_name.append(one_output.name)

        for i in range(0, len(self.ONMAGraph_GetGraph().output)):
            self.ONMAGraph_GetGraph().output.pop(i)

        for i, item in enumerate(output_name):
            self.ONMAGraph_GetGraph().output.append(self.ONMAGraph_CreateInput(item, GetTensorDataTypeFromnp(output_value[i].dtype), output_value[i].shape))

    def ONMAModel_ConvertONNXToJson(self, output_path):
        ConvertONNXToJson(self.ONMAGraph_GetGraph(), output_path)
