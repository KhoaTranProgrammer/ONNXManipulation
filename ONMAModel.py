import numpy as np
import onnx
from onnx.helper import (
    make_node, make_graph, make_model, make_tensor_value_info)
from onnx.numpy_helper import from_array
from onnx.checker import check_model
from onnxruntime import InferenceSession
from ONMANode import ONMANode
from ONMAGraph import ONMAGraph

def GetTensorDataTypeFromnp(npdtype):
    print(f'Np datatype: {npdtype}')
    datatype = onnx.TensorProto.FLOAT
    if npdtype == "float32":
        datatype = onnx.TensorProto.FLOAT
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
            values = data["tensor"]

            # Create new initializer with new value
            new_initializer = CreateInitializerTensor(
                        name=initializer.name,
                        tensor_array=values,
                        data_type=GetTensorDataTypeFromnp(values.dtype))
            initializers.pop(index)
            initializers.insert(index, new_initializer)
        elif action == "Remove":
            initializers.remove(initializer)
        elif action == "Add":
            values = data[action]["values"]

            # Create new initializer with new value
            new_initializer = CreateInitializerTensor(
                        name=name,
                        tensor_array=values,
                        data_type=GetTensorDataTypeFromnp(values.dtype))
            initializers.append(new_initializer)
        else:
            pass

def GetPreviousNodeFromInputName(model, inputs):
    node_res = []
    node_ind = []
    for i, node in enumerate(model.graph.node):
        for input in inputs:
            if inputs[input] in node.output:
                node_res.append(node)
                node_ind.append(i)
    return node_ind, node_res

def UpdateNode(model, name, data):
    if data["Action"] == "Add":
        op_type = data["Type"]
        inputs=list((data["inputs"]).values())
        outputs=list((data["outputs"]).values())

        node_ind,_ = GetPreviousNodeFromInputName(model, data["inputs"])

        # Simplify data
        data.pop("Action")
        data.pop("Category")
        data.pop("Type")
        data.pop("inputs")
        data.pop("outputs")

        onma_node = ONMANode()
        onma_node.ONMAMakeNode(
            op_type, inputs=inputs, outputs=outputs, name=name, **data
        )
        model.graph.node.insert(node_ind[0] + 1, onma_node.ONMAGetNode())
    elif data["Action"] == "Modify":
        for i, node in enumerate(model.graph.node):
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

                onma_node.ONMAMakeNode(
                    data["Type"], inputs=inputs, outputs=outputs, name=name
                )

                for attribute in node.attribute:
                    isReplace = False
                    for item in data:
                        if item == attribute.name:
                            isReplace = True
                            new_attribute = onnx.helper.make_attribute(item, data[item])
                            onma_node.ONMAGetNode().attribute.append(new_attribute)
                    if isReplace == False:
                        onma_node.ONMAGetNode().attribute.append(attribute)

                model.graph.node.pop(i)
                model.graph.node.insert(i, onma_node.ONMAGetNode())
    elif data["Action"] == "Remove":
        for i, node in enumerate(model.graph.node):
            if node.name == name:
                model.graph.node.pop(i)

class ONMAModel:
    def __init__(self):
        self._model = None

    def ONMAModel_MakeModel(self, graph):
        self._model = make_model(graph)
        check_model(self._model)

    def ONMAModel_LoadModel(self, input_path):
        self._model = onnx.load(input_path)

    def ONMAModel_Inference(self, infer_input):
        sess = InferenceSession(self._model.SerializeToString(), providers=["CPUExecutionProvider"])
        res = sess.run(None, infer_input)
        return res
    
    def ONMAModel_CreateNetworkWithOperator(
        self,
        operator_name,
        graph_name,
        inputs,
        outputs,
        **kwargs
    ):
        # Create Node
        onma_node = ONMANode()
        onma_node.ONMANode_MakeNode(
            operator_name, inputs=list(inputs.keys()), outputs=list(outputs.keys()), name=graph_name, **kwargs
        )

        # Remove empty input
        refine_input = {}
        for key, value in inputs.items():
            if key != "":
                refine_input[key] = value

        # Create graph input
        graph_input = []
        for i in range(0, len(list(refine_input.keys()))):
            graph_input.append(onma_node.ONMANode_CreateInput(list(refine_input.keys())[i], GetTensorDataTypeFromnp((list(refine_input.values())[i]).dtype), (list(refine_input.values())[i]).shape))

        # Create graph output
        graph_output = []
        for item in outputs:
            if outputs[item] == None:
                graph_output.append(onma_node.ONMANode_CreateInput(item, GetTensorDataTypeFromnp((list(refine_input.values())[0]).dtype), (list(refine_input.values())[0]).shape))
            else:
                graph_output.append(onma_node.ONMANode_CreateInput(item, GetTensorDataTypeFromnp((outputs[item]).dtype), (outputs[item]).shape))

        onma_graph = ONMAGraph()
        onma_graph.ONMAGraph_MakeGraph(graph_name, [onma_node.ONMANode_GetNode()], graph_input, graph_output)

        self.ONMAModel_MakeModel(onma_graph.ONMAGraph_GetGraph())

        return self.ONMAModel_Inference(refine_input)
    
    def ONMAModel_DisplayInformation(self, results, **argv):
        for oneargv in argv:
            if oneargv == "inputs":
                print("\n==========INPUT==========")
                for key, value in argv["inputs"].items():
                    print(f'Name: {key} - Shape: {value.shape}')
                    for dim in range(0, len(value.shape) - 1):
                        if dim == (len(value.shape) - 2):
                            print(value)
            elif oneargv == "outputs":
                print("\n==========OUTPUT==========")
                outputs = list(argv["outputs"].keys())
                for index in range(0, len(outputs)):
                    result = results[index]
                    print(f'Name: {outputs[index]} - Shape: {result.shape}')
                    for dim in range(0, len(result.shape) - 1):
                        if dim == (len(result.shape) - 2):
                            print(result)
            else:
                print(f'\nName: {oneargv} - Value: {argv[oneargv]}')

    def ONMAModel_UpdateModel(self, data, output_onnx):
        for item in data:
            try:
                if data[item]["Category"] == "Initializer":
                    if type(data[item]["tensor"]) is list:
                        data[item]["tensor"] = np.array(data[item]["tensor"])
                    UpdateInitializer(self._model.graph.initializer, item, data[item])
                elif data[item]["Category"] == "Node":
                    UpdateNode(self._model, item, data[item])
            except:
                pass

        # Save model
        inferred_model = onnx.shape_inference.infer_shapes(self._model)
        onnx.checker.check_model(inferred_model)
        onnx.save(inferred_model, output_onnx)
