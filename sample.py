import sys
import onnx
import argparse
import numpy as np
import json
from typing import Any, Sequence
import onnxruntime
from ONMAModel import ONMAModel
from onnx.backend.test.case.node import _extract_value_info
from onnx.backend.test.case.node.affinegrid import create_theta_2d
from onnx.backend.test.case.node.roialign import get_roi_align_input_values
from onnx.backend.test.case.node.layernormalization import calculate_normalized_shape

def expect(
    node: onnx.NodeProto,
    inputs: Sequence[np.ndarray],
    outputs: Sequence[np.ndarray],
    name: str,
    **kwargs: Any,
) -> None:
    # Builds the model
    present_inputs = [x for x in node.input if (x != "")]
    present_outputs = [x for x in node.output if (x != "")]
    input_type_protos = [None] * len(inputs)
    if "input_type_protos" in kwargs:
        input_type_protos = kwargs["input_type_protos"]
        del kwargs["input_type_protos"]
    output_type_protos = [None] * len(outputs)
    if "output_type_protos" in kwargs:
        output_type_protos = kwargs["output_type_protos"]
        del kwargs["output_type_protos"]
    inputs_vi = [
        _extract_value_info(arr, arr_name, input_type)
        for arr, arr_name, input_type in zip(inputs, present_inputs, input_type_protos)
    ]
    outputs_vi = [
        _extract_value_info(arr, arr_name, output_type)
        for arr, arr_name, output_type in zip(
            outputs, present_outputs, output_type_protos
        )
    ]
    graph = onnx.helper.make_graph(
        nodes=[node], name=name, inputs=inputs_vi, outputs=outputs_vi
    )
    kwargs["producer_name"] = "backend-test"

    if "opset_imports" not in kwargs:
        # To make sure the model will be produced with the same opset_version after opset changes
        # By default, it uses since_version as opset_version for produced models
        produce_opset_version = onnx.defs.get_schema(
            node.op_type, domain=node.domain
        ).since_version
        kwargs["opset_imports"] = [
            onnx.helper.make_operatorsetid(node.domain, produce_opset_version)
        ]

    model = onnx.helper.make_model_gen_version(graph, **kwargs)

    # Checking the produces are the expected ones.
    sess = onnxruntime.InferenceSession(model.SerializeToString(),
                                        providers=["CPUExecutionProvider"])
    feeds = {name: value for name, value in zip(node.input, inputs)}
    results = sess.run(None, feeds)
    for expected, output in zip(outputs, results):
        np.testing.assert_allclose(expected, output)
        print(f'expected: {expected}')
        print(f'output: {output}')

# Element type: https://onnx.ai/onnx/intro/concepts.html#element-type
element_type = {
    "onnx.TensorProto.FLOAT": np.float32,
    "onnx.TensorProto.UINT8": np.uint8,
    "onnx.TensorProto.INT8": np.int8,
    "onnx.TensorProto.UINT16": np.uint16,
    "onnx.TensorProto.INT16": np.int16,
    "onnx.TensorProto.INT32": np.int32,
    "onnx.TensorProto.INT64": np.int64,
    "onnx.TensorProto.STRING": np.U7,
    "onnx.TensorProto.BOOL": np.bool_,
    "onnx.TensorProto.FLOAT16": np.float16,
    "onnx.TensorProto.DOUBLE": np.float64,
    "onnx.TensorProto.UINT32": np.uint32,
    "onnx.TensorProto.UINT64": np.uint64,
    "onnx.TensorProto.COMPLEX64": np.complex64,
    "onnx.TensorProto.COMPLEX128": np.complex128,
    "onnx.TensorProto.BFLOAT16": "",
    "onnx.TensorProto.FLOAT8E4M3FN": "",
    "onnx.TensorProto.FLOAT8E4M3FNUZ": "",
    "onnx.TensorProto.FLOAT8E5M2": "",
    "onnx.TensorProto.FLOAT8E5M2FNUZ": "",
    "onnx.TensorProto.UINT4": "",
    "onnx.TensorProto.INT4": "",
    "onnx.TensorProto.FLOAT4E2M1": ""
}

def createSampleData(dimentions, datatype):
    return np.random.randn(*dimentions).astype(datatype)

node = onnx.helper.make_node(
    "Abs",
    inputs=["x"],
    outputs=["y"],
)
x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.abs(x)

print(createSampleData([1, 2, 3], element_type["onnx.TensorProto.INT8"]))
# expect(node, inputs=[x], outputs=[y], name="test_abs")
