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

class ONMAModel:
    def __init__(self):
        self._model = None

    def ONMAMakeModel(self, onma_graph):
        self._model = make_model(onma_graph.ONMAGetGraph())
        check_model(self._model)

    def ONMAInference(self, infer_input):
        sess = InferenceSession(self._model.SerializeToString(), providers=["CPUExecutionProvider"])
        res = sess.run(None, infer_input)
        return res
    
    def ONNXCreateNetworkWithOperator(
        self,
        operator_name,
        graph_name,
        inputs,
        outputs,
        **kwargs
    ):
        # Create Node
        onma_node = ONMANode()
        onma_node.ONMAMakeNode(
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
            graph_input.append(onma_node.ONMACreateInput(list(refine_input.keys())[i], GetTensorDataTypeFromnp((list(refine_input.values())[i]).dtype), (list(refine_input.values())[i]).shape))

        # Create graph output
        graph_output = []
        for item in outputs:
            if outputs[item] == None:
                graph_output.append(onma_node.ONMACreateInput(item, GetTensorDataTypeFromnp((list(refine_input.values())[0]).dtype), (list(refine_input.values())[0]).shape))
            else:
                graph_output.append(onma_node.ONMACreateInput(item, GetTensorDataTypeFromnp((outputs[item]).dtype), (outputs[item]).shape))

        onma_graph = ONMAGraph()
        onma_graph.ONMAMakeGraph(graph_name, [onma_node.ONMAGetNode()], graph_input, graph_output)

        self.ONMAMakeModel(onma_graph)

        return self.ONMAInference(refine_input)
    
    def ONMADisplayInformation(self, results, **argv):
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
