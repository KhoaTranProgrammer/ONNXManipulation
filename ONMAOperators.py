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

# operator_name: "Abs"
# graph_name: "Sample"
def CreateNetworkWithOperator(  \
        operator_name,          \
        graph_name,             \
        inputs,                 \
        outputs,                \
        output_dimension=None,  \
        output_datatype=None,   \
        direction=None,         \
        axes=None,              \
        axis=None,              \
        kernel_shape=None,      \
        pads=None,              \
        allowzero=None,         \
        exclusive=None,         \
        reverse=None,           \
        alpha=None,             \
        values=None,            \
        equation=None,          \
        beta=None,              \
        detect_positive=None,   \
        detect_negative=None,   \
        bias=None,              \
        size=None,              \
        fmod=None,              \
        lambd=None,             \
        align_corners=None,     \
        keepdims=None,          \
        select_last_index=None, \
        strides=None,           \
        ceil_mode=None,         \
        dilations=None,         \
        count_include_pad=None, \
        auto_pad=None,          \
        epsilon=None,           \
        training_mode=None,     \
        seed=None,              \
        periodic=None,          \
):
    # Create Node
    onma_node = ONMANode()
    onma_node.ONMAMakeNode(
        operator_name, inputs=list(inputs.keys()), outputs=list(outputs.keys()), direction=direction, axes=axes, axis=axis, \
        kernel_shape=kernel_shape, pads=pads, allowzero=allowzero, exclusive=exclusive, reverse=reverse, alpha=alpha, values=values, equation=equation, \
        beta=beta, detect_positive=detect_positive, detect_negative=detect_negative, bias=bias, size=size, fmod=fmod, lambd=lambd, align_corners=align_corners, \
        keepdims=keepdims, select_last_index=select_last_index, strides=strides, ceil_mode=ceil_mode, dilations=dilations, count_include_pad=count_include_pad, \
        auto_pad=auto_pad, epsilon=epsilon, training_mode=training_mode, seed=seed, periodic=periodic
    )

    # Create graph input
    graph_input = []
    for i in range(0, len(list(inputs.keys()))):
        graph_input.append(onma_node.ONMACreateInput(list(inputs.keys())[i], GetTensorDataTypeFromnp((list(inputs.values())[i]).dtype), (list(inputs.values())[i]).shape))

    # Create graph output
    graph_output = []
    try:
        if list(outputs.values()) == [None]:
            graph_output.append(onma_node.ONMACreateInput(list(outputs.keys())[0], GetTensorDataTypeFromnp((list(inputs.values())[0]).dtype), (list(inputs.values())[0]).shape))
    except:
        pass

    try:
        if len(list(outputs.values())) != 0:
            graph_output.append(onma_node.ONMACreateInput(list(outputs.keys())[0], GetTensorDataTypeFromnp((list(outputs.values())[0]).dtype), (list(outputs.values())[0]).shape))
    except:
        pass

    onma_graph = ONMAGraph()
    onma_graph.ONMAMakeGraph(graph_name, [onma_node.ONMAGetNode()], graph_input, graph_output)

    onma_model = ONMAModel()
    onma_model.ONMAMakeModel(onma_graph)

    onma_model.ONMAInference(inputs)

class ONMAOperators:
    def ONNX_CreateNetworkWithOperator(  \
        operator_name,          \
        graph_name,             \
        inputs,                 \
        outputs,                \
        output_dimension=None,  \
        output_datatype=None,   \
        direction=None,         \
        axes=None,              \
        axis=None,              \
        kernel_shape=None,      \
        pads=None,              \
        allowzero=None,         \
        exclusive=None,         \
        reverse=None,           \
        alpha=None,             \
        values=None,            \
        equation=None,          \
        beta=None,              \
        detect_positive=None,   \
        detect_negative=None,   \
        bias=None,              \
        size=None,              \
        fmod=None,              \
        lambd=None,             \
        align_corners=None,     \
        keepdims=None,          \
        select_last_index=None, \
        strides=None,           \
        ceil_mode=None,         \
        dilations=None,         \
        count_include_pad=None, \
        auto_pad=None,          \
        epsilon=None,           \
        training_mode=None,     \
        seed=None,              \
        periodic=None,          \
    ):
        CreateNetworkWithOperator(                  \
            operator_name,                          \
            graph_name,                             \
            inputs,                                 \
            outputs,                                \
            output_dimension=output_dimension,      \
            output_datatype=output_dimension,       \
            direction=direction,                    \
            axes=axes,                              \
            axis=axis,                              \
            kernel_shape=kernel_shape,              \
            pads=pads,                              \
            allowzero=allowzero,                    \
            exclusive=exclusive,                    \
            reverse=reverse,                        \
            alpha=alpha,                            \
            values=values,                          \
            equation=equation,                      \
            beta=beta,                              \
            detect_positive=detect_positive,        \
            detect_negative=detect_negative,        \
            bias=bias,                              \
            size=size,                              \
            fmod=fmod,                              \
            lambd=lambd,                            \
            align_corners=align_corners,            \
            keepdims=keepdims,                      \
            select_last_index=select_last_index,    \
            strides=strides,                        \
            ceil_mode=ceil_mode,                    \
            dilations=dilations,                    \
            count_include_pad=count_include_pad,    \
            auto_pad=auto_pad,                      \
            epsilon=epsilon,                        \
            training_mode=training_mode,            \
            seed=seed,                              \
            periodic=periodic,                      \
        )
