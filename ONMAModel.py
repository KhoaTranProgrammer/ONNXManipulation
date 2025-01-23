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

    def ONMAModel_MakeModel(self, graph):
        self._model = make_model(graph)
        check_model(self._model)

    def ONMAModel_LoadModel(self, input_path):
        self._model = onnx.load(input_path)
    
    def ONMAModel_SaveModel(self, output_path):
        inferred_model = onnx.shape_inference.infer_shapes(self._model)
        onnx.checker.check_model(inferred_model)
        onnx.save(inferred_model, output_path)

    def ONMAModel_Inference(self, infer_input):
        sess = InferenceSession(self._model.SerializeToString(), providers=["CPUExecutionProvider"])
        res = sess.run(None, infer_input)
        return res

    def ONMAModel_CreateNetworkFromGraph(self, data):
        refine_input = {}
        inputs = data["inputs"]
        for key, value in inputs.items():
            if key != "":
                try:
                    value_np = np.array(value["data"], dtype=value["type"])
                except:
                    value_np = np.array(value["data"], dtype='float32')
                refine_input[key] = value_np

        onma_graph = ONMAGraph()
        onma_graph.ONMAGraph_CreateNetworkFromGraph(data)
        self.ONMAModel_MakeModel(onma_graph.ONMAGraph_GetGraph())

        return self.ONMAModel_Inference(refine_input)
    
    def ONMAModel_DisplayInformation(self, results, **argv):
        for oneargv in argv:
            if oneargv == "inputs":
                print("\n==========INPUT==========")
                for key, value in argv["inputs"].items():
                    try:
                        value = np.array(value["data"], dtype=value["type"])
                    except:
                        try:
                            value = np.array(value["data"], dtype="float32")
                        except:
                            value = None
                    if value is not None:
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
                            print(np.array(result, dtype="float32"))
                    if len(result.shape) == 1:
                        print(result)

                    try:
                        try:
                            expect = np.array(argv["outputs"][outputs[index]]["data"], dtype=argv["outputs"][outputs[index]]["type"])
                        except:
                            expect = np.array(argv["outputs"][outputs[index]]["data"], dtype="float32")
                        if (result==expect).all():
                            print("Inference and expect are SAME")
                        else:
                            print("Inference and expect are DIFFRENT")
                    except:
                        print("Inference and expect are not available")
            else:
                pass
