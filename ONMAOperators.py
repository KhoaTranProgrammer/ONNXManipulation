import numpy as np
import onnx
from onnx.helper import (
    make_node, make_graph, make_model, make_tensor_value_info)
from onnx.numpy_helper import from_array
from onnx.checker import check_model
from onnxruntime import InferenceSession
from ONMANode import ONMANode
from ONMAGraph import ONMAGraph
from ONMAModel import ONMAModel

default_input = \
{
    "Abs": np.array([[ 0.06329948, -1.0832994 ,  0.37930292],
                     [ 0.71035045, -1.6637981 ,  1.0044696 ]]).astype(np.float32),
    "Acos": np.array([[ 0.06329948, 0.0832994 ,  0.37930292],
                      [ 0.71035045, 0.6637981 ,  0.0044696 ]]).astype(np.float32),
    "Acosh": np.array([[ 10, np.e, 1]]).astype(np.float32),
    "Add": {
        "Input1": np.array([[ 0.06329948, -1.0832994 ,  0.37930292],
                     [ 0.71035045, -1.6637981 ,  1.0044696 ]]).astype(np.float32),
        "Input2": np.array([[ 0.71035045, 0.0832994 ,  1.37930292],
                     [ 0.71035045, -1.6637981 ,  1.0044696 ]]).astype(np.float32),
    },
    "And": {
        "Input1": (np.random.randn(3, 4, 5) > 0).astype(bool),
        "Input2": (np.random.randn(3, 4, 5) > 0).astype(bool)
    }
}

def ONMARandomInput(dimensions, datatype=onnx.TensorProto.FLOAT):
    if datatype == onnx.TensorProto.FLOAT:
        return np.random.randn(*dimensions).astype(np.float32)
    elif datatype == onnx.TensorProto.INT32:
        return np.random.randn(*dimensions).astype(np.int32)
    return None

def Operator_1_Input_1_Output(operator_name, graph_name, inputs=["X"], outputs=["Y"], datatype=onnx.TensorProto.FLOAT, input_data=None):
    onma_node = ONMANode()
    onma_node.ONMAMakeNode(operator_name, inputs, outputs)

    try:
        if input_data == None:
            x = default_input[operator_name]
            infer_input = {inputs[0]: x}
            input = onma_node.ONMACreateInput(inputs[0], datatype, x.shape)
            output = onma_node.ONMACreateInput(outputs[0], datatype, x.shape)
    except:
        pass

    try:
        if input_data.all():
            infer_input = {inputs[0]: input_data}
            input = onma_node.ONMACreateInput(inputs[0], datatype, input_data.shape)
            output = onma_node.ONMACreateInput(outputs[0], datatype, input_data.shape)
    except:
        pass

    onma_graph = ONMAGraph()
    onma_graph.ONMAMakeGraph(graph_name, [onma_node.ONMAGetNode()], [input], [output])

    onma_model = ONMAModel()
    onma_model.ONMAMakeModel(onma_graph)

    onma_model.ONMAInference(infer_input)

def Operator_2_Inputs_1_Output(operator_name, graph_name, inputs=["X1", "X2"], outputs=["Y"], datatype=onnx.TensorProto.FLOAT, input_data1=None, input_data2=None):
    onma_node = ONMANode()
    onma_node.ONMAMakeNode(operator_name, inputs, outputs)

    try:
        if input_data1 == None or input_data2 == None:
            x1 = default_input[operator_name]["Input1"]
            x2 = default_input[operator_name]["Input2"]
            infer_input = {inputs[0]: x1, inputs[1]: x2}
            input1 = onma_node.ONMACreateInput(inputs[0], datatype, x1.shape)
            input2 = onma_node.ONMACreateInput(inputs[1], datatype, x2.shape)
            output = onma_node.ONMACreateInput(outputs[0], datatype, x1.shape)
    except:
        pass

    try:
        if input_data1.all() and input_data2.all():
            infer_input = {inputs[0]: input_data1, inputs[1]: input_data2}
            input1 = onma_node.ONMACreateInput(inputs[0], datatype, input_data1.shape)
            input2 = onma_node.ONMACreateInput(inputs[1], datatype, input_data2.shape)
            output = onma_node.ONMACreateInput(outputs[0], datatype, input_data1.shape)
    except:
        pass

    onma_graph = ONMAGraph()
    onma_graph.ONMAMakeGraph(graph_name, [onma_node.ONMAGetNode()], [input1, input2], [output])

    onma_model = ONMAModel()
    onma_model.ONMAMakeModel(onma_graph)

    onma_model.ONMAInference(infer_input)

class ONMAOperators:
    def ONMAOperator_1_Input_1_Output(operator_name, graph_name, inputs=["X"], outputs=["Y"], datatype=onnx.TensorProto.FLOAT, input_data=None):
        Operator_1_Input_1_Output(operator_name, graph_name, inputs, outputs, datatype, input_data)

    def ONMAOperator_2_Inputs_1_Output(operator_name, graph_name, inputs=["X1", "X2"], outputs=["Y"], datatype=onnx.TensorProto.FLOAT, input_data1=None, input_data2=None):
        Operator_2_Inputs_1_Output(operator_name, graph_name, inputs, outputs, datatype, input_data1, input_data2)
