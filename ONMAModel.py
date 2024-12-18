import numpy as np
import onnx
from onnx.helper import (
    make_node, make_graph, make_model, make_tensor_value_info)
from onnx.numpy_helper import from_array
from onnx.checker import check_model
from onnxruntime import InferenceSession
from ONMANode import ONMANode
from ONMAGraph import ONMAGraph

class ONMAModel:
    def __init__(self):
        self._model = None

    def ONMAMakeModel(self, onma_graph):
        self._model = make_model(onma_graph.ONMAGetGraph())
        check_model(self._model)

    def ONMAInference(self, infer_input):
        sess = InferenceSession(self._model.SerializeToString(), providers=["CPUExecutionProvider"])
        res = sess.run(None, infer_input)
        print("input", infer_input)
        print(f'result: {res}')
