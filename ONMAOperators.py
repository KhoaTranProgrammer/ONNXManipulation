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
    },
    "Asin": np.random.rand(3, 4, 5).astype(np.float32),
    "Asinh": np.random.rand(3, 4, 5).astype(np.float32),
    "Atan": np.random.rand(3, 4, 5).astype(np.float32),
    "Atanh": np.random.rand(3, 4, 5).astype(np.float32),
    "BitShift": {
        "Input1": np.array([16, 4, 1]).astype(np.uint8),
        "Input2": np.array([1, 2, 3]).astype(np.uint8)
    },
    "BitwiseAnd": {
        "Input1": np.random.randint(1, high = 9, size=(3, 4, 5)),
        "Input2": np.random.randint(1, high = 9, size=(3, 4, 5))
    },
    "BitwiseNot": np.random.randint(1, high = 9, size=(3, 4, 5), dtype=np.uint16),
    "BitwiseOr": {
        "Input1": np.random.randint(1, high = 9, size=(3, 4, 5)),
        "Input2": np.random.randint(1, high = 9, size=(3, 4, 5))
    },
    "BitwiseXor": {
        "Input1": np.random.randint(1, high = 9, size=(3, 4, 5)),
        "Input2": np.random.randint(1, high = 9, size=(3, 4, 5))
    },
    "Ceil": np.array([-1.5, 1.2]).astype(np.float32),
    "Celu": np.array(
                [
                    [
                        [[0.8439683], [0.5665144], [0.05836735]],
                        [[0.02916367], [0.12964272], [0.5060197]],
                        [[0.79538304], [0.9411346], [0.9546573]],
                    ],
                    [
                        [[0.17730942], [0.46192095], [0.26480448]],
                        [[0.6746842], [0.01665257], [0.62473077]],
                        [[0.9240844], [0.9722341], [0.11965699]],
                    ],
                    [
                        [[0.41356155], [0.9129373], [0.59330076]],
                        [[0.81929934], [0.7862604], [0.11799799]],
                        [[0.69248444], [0.54119414], [0.07513223]],
                    ],
                ],
                dtype=np.float32,
            ),
    "CenterCropPad": {
        "Input1": np.random.randn(20, 8, 3).astype(np.float32),
        "Input2": np.array([10, 9], dtype=np.int64)
    },
    "Clip": {
        "Input1": np.array([-2, 0, 2]).astype(np.float32),
        "Input2": np.array([-1]).astype(np.float32),
        "Input3": np.array([1]).astype(np.float32)
    },
    "Col2Im": {
        "Input1": np.array(
                    [
                        [
                            [1.0, 6.0, 11.0, 16.0, 21.0],  # (1, 5, 5)
                            [2.0, 7.0, 12.0, 17.0, 22.0],
                            [3.0, 8.0, 13.0, 18.0, 23.0],
                            [4.0, 9.0, 14.0, 19.0, 24.0],
                            [5.0, 0.0, 15.0, 20.0, 25.0],
                        ]
                    ]).astype(np.float32),
        "Input2": np.array([5, 5]).astype(np.int64),
        "Input3": np.array([1, 5]).astype(np.int64)
    }
}

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
    return datatype

def ONMARandomInput(dimensions, datatype=onnx.TensorProto.FLOAT):
    if datatype == onnx.TensorProto.FLOAT:
        return np.random.randn(*dimensions).astype(np.float32)
    elif datatype == onnx.TensorProto.INT32:
        return np.random.randn(*dimensions).astype(np.int32)
    return None

def Operator_1_Input_1_Output(operator_name, graph_name, inputs=["X"], outputs=["Y"], input_data=None, alpha=None):
    onma_node = ONMANode()
    onma_node.ONMAMakeNode(operator_name, inputs=inputs, outputs=outputs, alpha=alpha)

    try:
        if input_data == None:
            x = default_input[operator_name]
            infer_input = {inputs[0]: x}
            input = onma_node.ONMACreateInput(inputs[0], GetTensorDataTypeFromnp(x.dtype), x.shape)
            output = onma_node.ONMACreateInput(outputs[0], GetTensorDataTypeFromnp(x.dtype), x.shape)
    except:
        pass

    try:
        if input_data.all():
            infer_input = {inputs[0]: input_data}
            input = onma_node.ONMACreateInput(inputs[0], GetTensorDataTypeFromnp(input_data.dtype), input_data.shape)
            output = onma_node.ONMACreateInput(outputs[0], GetTensorDataTypeFromnp(input_data.dtype), input_data.shape)
    except:
        pass

    onma_graph = ONMAGraph()
    onma_graph.ONMAMakeGraph(graph_name, [onma_node.ONMAGetNode()], [input], [output])

    onma_model = ONMAModel()
    onma_model.ONMAMakeModel(onma_graph)

    onma_model.ONMAInference(infer_input)

def Operator_2_Inputs_1_Output(operator_name, graph_name, inputs=["X1", "X2"], outputs=["Y"], input_data1=None, input_data2=None, direction=None, axes=None):
    onma_node = ONMANode()
    onma_node.ONMAMakeNode(operator_name, inputs=inputs, outputs=outputs, direction=direction, axes=axes)

    try:
        if input_data1 == None or input_data2 == None:
            x1 = default_input[operator_name]["Input1"]
            x2 = default_input[operator_name]["Input2"]
            infer_input = {inputs[0]: x1, inputs[1]: x2}
            input1 = onma_node.ONMACreateInput(inputs[0], GetTensorDataTypeFromnp(x1.dtype), x1.shape)
            input2 = onma_node.ONMACreateInput(inputs[1], GetTensorDataTypeFromnp(x2.dtype), x2.shape)
            output = onma_node.ONMACreateInput(outputs[0], GetTensorDataTypeFromnp(x1.dtype), x1.shape)
    except:
        pass

    try:
        if input_data1.all() and input_data2.all():
            infer_input = {inputs[0]: input_data1, inputs[1]: input_data2}
            input1 = onma_node.ONMACreateInput(inputs[0], GetTensorDataTypeFromnp(input_data1.dtype), input_data1.shape)
            input2 = onma_node.ONMACreateInput(inputs[1], GetTensorDataTypeFromnp(input_data2.dtype), input_data2.shape)
            output = onma_node.ONMACreateInput(outputs[0], GetTensorDataTypeFromnp(input_data1.dtype), input_data1.shape)
    except:
        pass

    onma_graph = ONMAGraph()
    onma_graph.ONMAMakeGraph(graph_name, [onma_node.ONMAGetNode()], [input1, input2], [output])

    onma_model = ONMAModel()
    onma_model.ONMAMakeModel(onma_graph)

    onma_model.ONMAInference(infer_input)

def Operator_3_Inputs_1_Output(operator_name, graph_name, inputs=["X1", "X2", "X3"], outputs=["Y"], input_data1=None, input_data2=None, input_data3=None):
    onma_node = ONMANode()
    onma_node.ONMAMakeNode(operator_name, inputs=inputs, outputs=outputs)

    try:
        if input_data1 == None or input_data2 == None or input_data3 == None:
            x1 = default_input[operator_name]["Input1"]
            x2 = default_input[operator_name]["Input2"]
            x3 = default_input[operator_name]["Input3"]
            infer_input = {inputs[0]: x1, inputs[1]: x2, inputs[2]: x3}
            input1 = onma_node.ONMACreateInput(inputs[0], GetTensorDataTypeFromnp(x1.dtype), x1.shape)
            input2 = onma_node.ONMACreateInput(inputs[1], GetTensorDataTypeFromnp(x2.dtype), x2.shape)
            input3 = onma_node.ONMACreateInput(inputs[2], GetTensorDataTypeFromnp(x3.dtype), x3.shape)
            output = onma_node.ONMACreateInput(outputs[0], GetTensorDataTypeFromnp(x1.dtype), x1.shape)
    except:
        pass

    try:
        if input_data1.all() and input_data2.all() and input_data3.all():
            infer_input = {inputs[0]: input_data1, inputs[1]: input_data2, inputs[2]: input_data3}
            input1 = onma_node.ONMACreateInput(inputs[0], GetTensorDataTypeFromnp(input_data1.dtype), input_data1.shape)
            input2 = onma_node.ONMACreateInput(inputs[1], GetTensorDataTypeFromnp(input_data2.dtype), input_data2.shape)
            input3 = onma_node.ONMACreateInput(inputs[2], GetTensorDataTypeFromnp(input_data3.dtype), input_data3.shape)
            output = onma_node.ONMACreateInput(outputs[0], GetTensorDataTypeFromnp(input_data1.dtype), input_data1.shape)
    except:
        pass

    onma_graph = ONMAGraph()
    onma_graph.ONMAMakeGraph(graph_name, [onma_node.ONMAGetNode()], [input1, input2, input3], [output])

    onma_model = ONMAModel()
    onma_model.ONMAMakeModel(onma_graph)

    onma_model.ONMAInference(infer_input)

class ONMAOperators:
    def ONMAOperator_1_Input_1_Output(operator_name, graph_name, inputs=["X"], outputs=["Y"], input_data=None, alpha=None):
        Operator_1_Input_1_Output(operator_name, graph_name, inputs=inputs, outputs=outputs, input_data=input_data, alpha=alpha)

    def ONMAOperator_2_Inputs_1_Output(operator_name, graph_name, inputs=["X1", "X2"], outputs=["Y"], input_data1=None, input_data2=None, direction=None, axes=None):
        Operator_2_Inputs_1_Output(operator_name, graph_name, inputs=inputs, outputs=outputs, input_data1=input_data1, input_data2=input_data2, direction=direction, axes=axes)

    def ONMAOperator_3_Inputs_1_Output(operator_name, graph_name, inputs=["X1", "X2", "X3"], outputs=["Y"], input_data1=None, input_data2=None, input_data3=None):
        Operator_3_Inputs_1_Output(operator_name, graph_name, inputs=inputs, outputs=outputs, input_data1=input_data1, input_data2=input_data2, input_data3=input_data3)
