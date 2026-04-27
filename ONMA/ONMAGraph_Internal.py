import numpy as np
import onnx
import json
import ast
from onnx.helper import (
    make_node, make_graph, make_model, make_tensor_value_info)
from onnx.numpy_helper import from_array
from onnx.checker import check_model
from onnxruntime import InferenceSession

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

def GetPreviousNodeFromInputName(graph, inputs):
    node_res = []
    node_ind = []
    for i, node in enumerate(graph.node):
        for input in inputs:
            if input in node.output:
                node_res.append(node)
                node_ind.append(i)
    return node_ind, node_res
