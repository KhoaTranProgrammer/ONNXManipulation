import numpy as np
import onnx
from onnx.helper import (
    make_node, make_graph, make_model, make_tensor_value_info)
from onnx.numpy_helper import from_array
from onnx.checker import check_model
from onnxruntime import InferenceSession

class ONMAGraph:
    def __init__(self):
        self._graph = None

    def ONMAMakeGraph(self, name, nodes, inputs, outputs):
        self._graph = onnx.helper.make_graph(nodes, name, inputs, outputs)
    
    def ONMAGetGraph(self):
        return self._graph
