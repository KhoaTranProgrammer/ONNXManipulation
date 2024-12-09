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

    def ONMAMakeNode(self, name, inputs, outputs, direction=None):
        if direction == None:
            self._node = onnx.helper.make_node(
                name,
                inputs=inputs,
                outputs=outputs
            )
        else:
            self._node = onnx.helper.make_node(
                name,
                inputs=inputs,
                outputs=outputs,
                direction=direction
            )
    
    def ONMAGetNode(self):
        return self._node
    
    def ONMACreateInput(self, name, type, dimension):
        return make_tensor_value_info(name, type, dimension)
