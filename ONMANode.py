import numpy as np
import onnx
from onnx.helper import (
    make_node, make_graph, make_model, make_tensor_value_info)
from onnx.numpy_helper import from_array
from onnx.checker import check_model
from onnxruntime import InferenceSession

class ONMANode:
    def __init__(self):
        self._node = None

    def ONMAMakeNode(
            self,                   \
            name,                   \
            inputs,                 \
            outputs,                \
            direction=None,         \
            alpha=None,             \
            axes=None,              \
            axis=None,              \
            values=None,            \
            kernel_shape=None,      \
            pads=None,              \
            allowzero=None,         \
            exclusive=None,         \
            reverse=None,           \
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
            pattern=None,           \
        ):
        try:
            if values == None:
                self._node = onnx.helper.make_node(
                    name,
                    inputs=inputs,
                    outputs=outputs,
                    direction=direction,
                    alpha=alpha,
                    axes=axes,
                    axis=axis,
                    kernel_shape=kernel_shape,
                    pads=pads,
                    allowzero=allowzero,
                    exclusive=exclusive,
                    reverse=reverse,
                    equation=equation,
                    beta=beta,
                    detect_positive=detect_positive,
                    detect_negative=detect_negative,
                    bias=bias,
                    size=size,
                    fmod=fmod,
                    lambd=lambd,
                    align_corners=align_corners,
                    keepdims=keepdims,
                    select_last_index=select_last_index,
                    strides=strides,
                    ceil_mode=ceil_mode,
                    dilations=dilations,
                    count_include_pad=count_include_pad,
                    auto_pad=auto_pad,
                    epsilon=epsilon,
                    training_mode=training_mode,
                    seed=seed,
                    periodic=periodic,
                    pattern=pattern,
                )
        except:
            pass

        try:
            if values.all():
                self._node = onnx.helper.make_node(
                    name,
                    inputs=inputs,
                    outputs=outputs,
                    value=onnx.helper.make_tensor(
                        name="const_tensor",
                        data_type=onnx.TensorProto.FLOAT,
                        dims=values.shape,
                        vals=values.flatten().astype(float),
                    )
                )
        except:
            pass

    def ONMAGetNode(self):
        return self._node
    
    def ONMACreateInput(self, name, type, dimension):
        return make_tensor_value_info(name, type, dimension)
