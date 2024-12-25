import sys
import onnx
import argparse
import numpy as np
from ONMAOperators import ONMAOperators
from onnx.backend.test.case.node.affinegrid import create_theta_2d
from onnx.backend.test.case.node.roialign import get_roi_align_input_values

global args

default_input = \
{
    "Abs": {
        "graph_name": "Reshape_sample",
        "inputs": {
            "X": np.array([[ 0.06329948, -1.0832994 ,  0.37930292],
                           [ 0.71035045, -1.6637981 ,  1.0044696 ]]).astype(np.float32)
        },
        "outputs": {
            "Y": None
        },
    },
    "Acos": {
        "graph_name": "Acos_sample",
        "inputs": {
            "X": np.array([[ 0.06329948, 0.0832994 ,  0.37930292],
                      [ 0.71035045, 0.6637981 ,  0.0044696 ]]).astype(np.float32),
        },
        "outputs": {
            "Y": None
        },
    },
    "Acosh": {
        "graph_name": "Acosh_sample",
        "inputs": {
            "X": np.array([[ 10, np.e, 1]]).astype(np.float32),
        },
        "outputs": {
            "Y": None
        },
    },
    "Add": {
        "graph_name": "Add_sample",
        "inputs": {
            "X1": np.array([[ 0.06329948, -1.0832994 ,  0.37930292],
                     [ 0.71035045, -1.6637981 ,  1.0044696 ]]).astype(np.float32),
            "X2": np.array([[ 0.71035045, 0.0832994 ,  1.37930292],
                        [ 0.71035045, -1.6637981 ,  1.0044696 ]]).astype(np.float32),
        },
        "outputs": {
            "Y": None
        },
    },
    "AffineGrid": {
        "graph_name": "AffineGrid_sample",
        "inputs": {
            "theta": create_theta_2d(),
            "size": np.array([len(create_theta_2d()), 3, 5, 6], dtype=np.int64),
        },
        "outputs": {
            "Y": None
        },
        "align_corners": 0
    },
    "And": {
        "graph_name": "And_sample",
        "inputs": {
            "X1": (np.random.randn(3, 4, 5) > 0).astype(bool),
            "X2": (np.random.randn(3, 4, 5) > 0).astype(bool)
        },
        "outputs": {
            "Y": None
        },
    },
    "ArgMax": {
        "graph_name": "ArgMax_sample",
        "inputs": {
            "data": np.array([[2, 2], [3, 10]], dtype=np.float32)
        },
        "outputs": {
            "result": np.empty(shape=(1), dtype=np.int64)
        },
        "axis": 0,
        "keepdims": 0,
        "select_last_index": True,
    },
    "ArgMin": {
        "graph_name": "ArgMin_sample",
        "inputs": {
            "data": np.array([[2, 2], [3, 10]], dtype=np.float32)
        },
        "outputs": {
            "result": np.empty(shape=(1), dtype=np.int64)
        },
        "axis": 1,
        "keepdims": 1,
        "select_last_index": True,
    },
    "Asin": {
        "graph_name": "Asin_sample",
        "inputs": {
            "X": np.random.rand(3, 4, 5).astype(np.float32),
        },
        "outputs": {
            "Y": None
        },
    },
    "Asinh": {
        "graph_name": "Asinh_sample",
        "inputs": {
            "X": np.random.rand(3, 4, 5).astype(np.float32),
        },
        "outputs": {
            "Y": None
        },
    },
    "Atan": {
        "graph_name": "Atan_sample",
        "inputs": {
            "X": np.random.rand(3, 4, 5).astype(np.float32),
        },
        "outputs": {
            "Y": None
        },
    },
    "Atanh": {
        "graph_name": "Atanh_sample",
        "inputs": {
            "X": np.random.rand(3, 4, 5).astype(np.float32),
        },
        "outputs": {
            "Y": None
        },
    },
    "AveragePool": {
        "graph_name": "AveragePool_sample",
        "inputs": {
            "X": np.random.randn(1, 1, *(32, 32, 32)).astype(np.float32)
        },
        "outputs": {
            "Y": None
        },
        "kernel_shape": (5, 5, 5),
        "strides": (3, 3, 3),
        "dilations": (2, 2, 2),
        "count_include_pad": 0,
        "ceil_mode": True,
        "auto_pad": "SAME_UPPER"
    },
    "BatchNormalization": {
        "graph_name": "BatchNormalization_sample",
        "inputs": {
            "x": np.random.randn(2, 3, 4, 5).astype(np.float32),
            "s": np.random.randn(3).astype(np.float32),
            "bias": np.random.randn(3).astype(np.float32),
            "mean": np.random.randn(3).astype(np.float32),
            "var": np.random.rand(3).astype(np.float32),
        },
        "outputs": {
            "Y": None,
        },
        "epsilon": 1e-2,
    },
    "Bernoulli": {
        "graph_name": "Bernoulli_sample",
        "inputs": {
            "x": np.random.uniform(0.0, 1.0, 10).astype(np.float32)
        },
        "outputs": {
            "y": None
        },
        "seed": float(0),
    },
    "BitShift": {
        "graph_name": "BitShift_sample",
        "inputs": {
            "X1": np.array([16, 4, 1]).astype(np.uint8),
            "X2": np.array([1, 2, 3]).astype(np.uint8)
        },
        "outputs": {
            "Y": None
        },
        "direction": "LEFT",
    },
    "BitwiseAnd": {
        "graph_name": "BitwiseAnd_sample",
        "inputs": {
            "X1": np.random.randint(1, high = 9, size=(3, 4, 5)),
            "X2": np.random.randint(1, high = 9, size=(3, 4, 5))
        },
        "outputs": {
            "Y": None
        },
    },
    "BitwiseNot": {
        "graph_name": "BitwiseNot_sample",
        "inputs": {
            "X": np.random.randint(1, high = 9, size=(3, 4, 5), dtype=np.uint16)
        },
        "outputs": {
            "Y": None
        },
    },
    "BitwiseOr": {
        "graph_name": "BitwiseOr_sample",
        "inputs": {
            "X1": np.random.randint(1, high = 9, size=(3, 4, 5)),
            "X2": np.random.randint(1, high = 9, size=(3, 4, 5))
        },
        "outputs": {
            "Y": None
        },
    },
    "BitwiseXor": {
        "graph_name": "BitwiseXor_sample",
        "inputs": {
            "X1": np.random.randint(1, high = 9, size=(3, 4, 5)),
            "X2": np.random.randint(1, high = 9, size=(3, 4, 5))
        },
        "outputs": {
            "Y": None
        },
    },
    "BlackmanWindow": {
        "graph_name": "BlackmanWindow_sample",
        "inputs": {
            "x": np.array([10], dtype=np.int32),
        },
        "outputs": {
            "y": np.empty(shape=(1), dtype=np.float32)
        },
        "periodic": 0
    },
    "Cast": {
        "graph_name": "Cast_sample",
        "inputs": {
            "x": np.array(["0.47892547","0.48033667","0.49968487","0.81910545","0.47031248","0.7229038","1000000","1e-7","NaN","INF","+INF","-INF","-0.0000001","0.0000001","-1000000",], dtype=np.float32),
        },
        "outputs": {
            "y": np.empty(shape=(1), dtype=np.uint8)
        },
        "to": onnx.TensorProto.UINT8,
    },
    "CastLike": {
        "graph_name": "CastLike_sample",
        "inputs": {
            "x": np.array(["0.47892547","0.48033667","0.49968487","0.81910545","0.47031248","0.7229038","1000000","1e-7","NaN","INF","+INF","-INF","-0.0000001","0.0000001","-1000000",], dtype=np.float32),
            "like": np.empty(shape=(1), dtype=np.uint8)
        },
        "outputs": {
            "y": np.empty(shape=(1), dtype=np.uint8)
        },
    },
    "Ceil": {
        "graph_name": "Ceil_sample",
        "inputs": {
            "X": np.array([-1.5, 1.2]).astype(np.float32),
        },
        "outputs": {
            "Y": None
        },
    },
    "Celu": {
        "graph_name": "Celu_sample",
        "inputs": {
            "X": np.array(
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
        },
        "outputs": {
            "Y": None
        },
        "alpha": 2.0
    },
    "CenterCropPad": {
        "graph_name": "CenterCropPad_sample",
        "inputs": {
            "X1": np.random.randn(20, 8, 3).astype(np.float32),
            "X2": np.array([10, 9], dtype=np.int64)
        },
        "outputs": {
            "Y": None
        },
        "axes": [-3, -2]
    },
    "Clip": {
        "graph_name": "Clip_sample",
        "inputs": {
            "X1": np.array([-2, 0, 2]).astype(np.float32),
            "X2": np.array([-1]).astype(np.float32),
            "X3": np.array([1]).astype(np.float32)
        },
        "outputs": {
            "Y": None
        },
    },
    "Col2Im": {
        "graph_name": "Col2Im_sample",
        "inputs": {
            "X1": np.array(
                    [
                        [
                            [1.0, 6.0, 11.0, 16.0, 21.0],  # (1, 5, 5)
                            [2.0, 7.0, 12.0, 17.0, 22.0],
                            [3.0, 8.0, 13.0, 18.0, 23.0],
                            [4.0, 9.0, 14.0, 19.0, 24.0],
                            [5.0, 0.0, 15.0, 20.0, 25.0],
                        ]
                    ]).astype(np.float32),
            "X2": np.array([5, 5]).astype(np.int64),
            "X3": np.array([1, 5]).astype(np.int64)
        },
        "outputs": {
            "Y": None
        },
    },
    "Compress": {
        "graph_name": "Compress_sample",
        "inputs": {
            "X1": np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32),
            "X2": np.array([0, 1, 1]).astype(bool)
        },
        "outputs": {
            "Y": None
        },
        "axis": 0
    },
    "Concat": {
        "graph_name": "Concat_sample",
        "inputs": {
            "X1": np.array([[1, 2], [3, 4]]).astype(np.float32),
            "X2": np.array([[5, 6], [7, 8]]).astype(np.float32),
        },
        "outputs": {
            "Y": None
        },
        "axis": 1
    },
    "ConcatFromSequence": { },
    "Constant": {
        "graph_name": "Constant_sample",
        "inputs": {
            
        },
        "outputs": {
            "Y": np.empty(shape=(5, 5), dtype=np.float32)
        },
        "values": np.random.randn(5, 5).astype(np.float32)
    },
    "ConstantOfShape": {
        "graph_name": "ConstantOfShape_sample",
        "inputs": {
            "X": np.array([10, 6]).astype(np.int64)
        },
        "outputs": {
            "Y": np.empty(shape=(1), dtype=np.int32)
        },
        "values": onnx.helper.make_tensor("value", onnx.TensorProto.INT32, [1], [0])
    },
    "Conv": {
        "graph_name": "Conv_sample",
        "inputs": {
            "X1": np.array(
                        [
                            [
                                [
                                    [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 1, 5, 5) input tensor
                                    [5.0, 6.0, 7.0, 8.0, 9.0],
                                    [10.0, 11.0, 12.0, 13.0, 14.0],
                                    [15.0, 16.0, 17.0, 18.0, 19.0],
                                    [20.0, 21.0, 22.0, 23.0, 24.0],
                                ]
                            ]
                        ]
                    ).astype(np.float32),
            "X2": np.array(
                        [
                            [
                                [
                                    [1.0, 1.0, 1.0],  # (1, 1, 3, 3) tensor for convolution weights
                                    [1.0, 1.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ]
                            ]
                        ]
                    ).astype(np.float32)
        },
        "outputs": {
            "Y": None
        },
        "kernel_shape": [3, 3],
        "pads": [1, 1, 1, 1]
    },
    "ConvInteger": {
        "graph_name": "ConvInteger_sample",
        "inputs": {
            "x": (np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.uint8).reshape((1, 1, 3, 3))),
            "w": np.array([1, 1, 1, 1]).astype(np.uint8).reshape((1, 1, 2, 2)),
            "x_zero_point": np.array([1]).astype(np.uint8)
        },
        "outputs": {
            "Y": np.empty(shape=(1), dtype=np.int32)
        },
        "pads": [1, 1, 1, 1]
    },
    "ConvTranspose": {
        "graph_name": "ConvTranspose_sample",
        "inputs": {
            "x": np.array([[[0.0, 1.0, 2.0]]]).astype(np.float32),
            "w": np.array([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]).astype(np.float32),
        },
        "outputs": {
            "y": np.empty(shape=(1), dtype=np.float32)
        },
    },
    "Cos": {
        "graph_name": "Cos_sample",
        "inputs": {
            "X": np.array([-1, 0, 1]).astype(np.float32),
        },
        "outputs": {
            "Y": None
        }
    },
    "Cosh": {
        "graph_name": "Cosh_sample",
        "inputs": {
            "X": np.array([-1, 0, 1]).astype(np.float32),
        },
        "outputs": {
            "Y": None
        }
    },
    "CumSum": {
        "graph_name": "CumSum_sample",
        "inputs": {
            "X1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float32),
            "X2": np.array([0]).astype(np.int32)
        },
        "outputs": {
            "Y": None
        },
        "reverse": 1,
        "exclusive": 1
    },
    "DFT": {
        "graph_name": "DFT_sample",
        "inputs": {
            "x": np.arange(0, 100).reshape(10, 10).astype(np.float32).reshape(1, 10, 10, 1),
            "": "",
            "axis": np.array(1, dtype=np.int64)
        },
        "outputs": {
            "y": None
        }
    },
    "DeformConv": { },
    "DepthToSpace": {
        "graph_name": "DepthToSpace_sample",
        "inputs": {
            "x": np.array([[
                    [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                    [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]],
                    [[18.0, 19.0, 20.0], [21.0, 22.0, 23.0]],
                    [[27.0, 28.0, 29.0], [30.0, 31.0, 32.0]],
                    [[36.0, 37.0, 38.0], [39.0, 40.0, 41.0]],
                    [[45.0, 46.0, 47.0], [48.0, 49.0, 50.0]],
                    [[54.0, 55.0, 56.0], [57.0, 58.0, 59.0]],
                    [[63.0, 64.0, 65.0], [66.0, 67.0, 68.0]],]]).astype(np.float32)
        },
        "outputs": {
            "y": None
        },
        "blocksize": 2,
        "mode": "DCR"
    },
    "DequantizeLinear": {
        "graph_name": "DequantizeLinear_sample",
        "inputs": {
            "x": np.array([0, 3, 128, 255]).astype(np.uint8),
            "x_scale": np.array([2]).astype(np.float32),
            "x_zero_point": np.array([128]).astype(np.uint8)
        },
        "outputs": {
            "Y": np.empty(shape=(1), dtype=np.float32)
        }
    },
    "Det": {
        "graph_name": "Det_sample",
        "inputs": {
            "X": np.array([[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]).astype(np.float32),
        },
        "outputs": {
            "Y": None
        }
    },
    "Div": {
        "graph_name": "Div_sample",
        "inputs": {
            "X1": np.array([3, 4]).astype(np.float32),
            "X2": np.array([1, 2]).astype(np.float32)
        },
        "outputs": {
            "Y": None
        },
    },
    "Dropout": {
        "graph_name": "Dropout_sample",
        "inputs": {
            "x": np.random.randn(3, 4, 5).astype(np.float32),
            "r": np.array(0.75).astype(np.float32),
            "t": np.array(True).astype(np.bool_)
        },
        "outputs": {
            "y": None
        },
        "seed": 0
    },
    "DynamicQuantizeLinear": { },
    "Einsum": {
        "graph_name": "Einsum_sample",
        "inputs": {
            "X": np.random.randn(5).astype(np.float32),
            "Y": np.random.randn(5).astype(np.float32)
        },
        "outputs": {
            "Z": None
        },
        "equation": "i,i"
    },
    "Elu": {
        "graph_name": "Elu_sample",
        "inputs": {
            "X": np.random.randn(5).astype(np.float32)
        },
        "outputs": {
            "Y": None
        },
        "alpha": 2.0
    },
    "Equal": {
        "graph_name": "Equal_sample",
        "inputs": {
            "X": (np.random.randn(3, 4, 5) * 10).astype(np.int32),
            "Y": (np.random.randn(3, 4, 5) * 10).astype(np.int32)
        },
        "outputs": {
            "Z": np.empty(shape=(3, 4, 5), dtype=bool)
        }
    },
    "Erf": {
        "graph_name": "Erf_sample",
        "inputs": {
            "X": np.random.randn(1, 3, 32, 32).astype(np.float32)
        },
        "outputs": {
            "Y": None
        }
    },
    "Exp": {
        "graph_name": "Exp_sample",
        "inputs": {
            "X": np.array([-1, 0, 1]).astype(np.float32)
        },
        "outputs": {
            "Y": None
        }
    },
    "Expand": {
        "graph_name": "Expand_sample",
        "inputs": {
            "X": np.reshape(np.arange(1, np.prod([3, 1]) + 1, dtype=np.float32), [3, 1]),
            "Y": np.array([2, 1, 6], dtype=np.int64)
        },
        "outputs": {
            "Z": None
        }
    },
    "EyeLike": {
        "graph_name": "EyeLike_sample",
        "inputs": {
            "X": np.random.randint(0, 100, size=(4, 5), dtype=np.int32)
        },
        "outputs": {
            "Y": None
        },
        "k": 1,
        "dtype": onnx.TensorProto.FLOAT,
    },
    "Flatten": {
        "graph_name": "Flatten_sample",
        "inputs": {
            "X": np.random.random_sample((2, 3, 4)).astype(np.float32)
        },
        "outputs": {
            "Y": None
        },
        "axis": 2
    },
    "Floor": {
        "graph_name": "Floor_sample",
        "inputs": {
            "X": np.array([-1.5, 1.2, 2]).astype(np.float32)
        },
        "outputs": {
            "Y": None
        }
    },
    "GRU": { },
    "Gather": {
        "graph_name": "Gather_sample",
        "inputs": {
            "data": np.random.randn(3, 3, 4).astype(np.float32),
            "indices": np.array([1, 2])
        },
        "outputs": {
            "y": None
        },
        "axis": 0
    },
    "GatherElements": {
        "graph_name": "GatherElements_sample",
        "inputs": {
            "data": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
            "indices": np.array([[1, 2, 0], [2, 0, 0]], dtype=np.int32)
        },
        "outputs": {
            "y": None
        },
        "axis": 0
    },
    "GatherND": {
        "graph_name": "GatherND_sample",
        "inputs": {
            "data": np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int32),
            "indices": np.array([[1], [0]], dtype=np.int64)
        },
        "outputs": {
            "y": None
        },
        "batch_dims": 1
    },
    "Gelu": {
        "graph_name": "Gelu_sample",
        "inputs": {
            "X": np.array([-1, 0, 1]).astype(np.float32)
        },
        "outputs": {
            "Y": None
        }
    },
    "Gemm": {
        "graph_name": "Gemm_sample",
        "inputs": {
            "a": np.random.ranf([3, 5]).astype(np.float32),
            "b": np.random.ranf([5, 4]).astype(np.float32),
            "c": np.zeros([1, 4]).astype(np.float32)
        },
        "outputs": {
            "Y": None
        },
        "alpha": 0.5
    },
    "GlobalAveragePool": {
        "graph_name": "GlobalAveragePool_sample",
        "inputs": {
            "X": np.random.randn(1, 3, 2, 2).astype(np.float32)
        },
        "outputs": {
            "Y": None
        }
    },
    "GlobalLpPool": { },
    "GlobalMaxPool": {
        "graph_name": "GlobalMaxPool_sample",
        "inputs": {
            "X": np.random.randn(1, 3, 5, 5).astype(np.float32)
        },
        "outputs": {
            "Y": None
        }
    },
    "Greater": {
        "graph_name": "Greater_sample",
        "inputs": {
            "X": np.random.randn(3, 4, 5).astype(np.float32),
            "Y": np.random.randn(3, 4, 5).astype(np.float32)
        },
        "outputs": {
            "greater": np.empty(shape=(3, 4, 5), dtype=bool)
        }
    },
    "GreaterOrEqual": {
        "graph_name": "GreaterOrEqual_sample",
        "inputs": {
            "X": np.random.randn(3, 4, 5).astype(np.float32),
            "Y": np.random.randn(3, 4, 5).astype(np.float32)
        },
        "outputs": {
            "greater": np.empty(shape=(3, 4, 5), dtype=bool)
        }
    },
    "GridSample": {
        "graph_name": "GridSample_sample",
        "inputs": {
            "X": np.array([[[
                    [0.0, 1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [8.0, 9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0, 15.0],
                ]]],dtype=np.float32,),
            "Grid": np.array([[
                    [[-1.0000, -1.0000], [-0.6000, -1.0000], [-0.2000, -1.0000], [0.2000, -1.0000], [0.6000, -1.0000], [1.0000, -1.0000],],
                    [[-1.0000, -0.6000], [-0.6000, -0.6000], [-0.2000, -0.6000], [0.2000, -0.6000], [0.6000, -0.6000], [1.0000, -0.6000],],
                    [[-1.0000, -0.2000], [-0.6000, -0.2000], [-0.2000, -0.2000], [0.2000, -0.2000], [0.6000, -0.2000], [1.0000, -0.2000],],
                    [[-1.0000, 0.2000], [-0.6000, 0.2000], [-0.2000, 0.2000], [0.2000, 0.2000], [0.6000, 0.2000], [1.0000, 0.2000],],
                    [[-1.0000, 0.6000], [-0.6000, 0.6000], [-0.2000, 0.6000], [0.2000, 0.6000], [0.6000, 0.6000], [1.0000, 0.6000],],
                    [[-1.0000, 1.0000], [-0.6000, 1.0000], [-0.2000, 1.0000], [0.2000, 1.0000], [0.6000, 1.0000], [1.0000, 1.0000],],]],dtype=np.float32,)
        },
        "outputs": {
            "Y": None
        },
        "mode": "linear",
        "padding_mode": "zeros",
        "align_corners": 0,
    },
    "GroupNormalization": {
        "graph_name": "GroupNormalization_sample",
        "inputs": {
            "x": np.random.randn(3, 4, 2, 2).astype(np.float32),
            "scale": np.random.randn(4).astype(np.float32),
            "bias": np.random.randn(4).astype(np.float32)
        },
        "outputs": {
            "y": None
        },
        "epsilon": 1e-2,
        "num_groups": 2,
    },
    "HammingWindow": {
        "graph_name": "HammingWindow_sample",
        "inputs": {
            "size": np.array([10]).astype(np.int32)
        },
        "outputs": {
            "Y": np.empty(shape=(6), dtype=np.float32)
        },
        "periodic": 0
    },
    "HannWindow": {
        "graph_name": "HannWindow_sample",
        "inputs": {
            "size": np.array([10]).astype(np.int32)
        },
        "outputs": {
            "Y": np.empty(shape=(6), dtype=np.float32)
        },
        "periodic": 0
    },
    "HardSigmoid": {
        "graph_name": "HardSigmoid_sample",
        "inputs": {
            "X": np.array([-1, 0, 1]).astype(np.float32)
        },
        "outputs": {
            "Y": None
        },
        "alpha": 0.5, 
        "beta": 0.6
    },
    "HardSwish": {
        "graph_name": "HardSwish_sample",
        "inputs": {
            "X": np.random.randn(3, 4, 5).astype(np.float32)
        },
        "outputs": {
            "Y": None
        }
    },
    "Hardmax": {
        "graph_name": "Hardmax_sample",
        "inputs": {
            "X": np.array([[3, 0, 1, 2], [2, 5, 1, 0], [0, 1, 3, 2], [0, 1, 2, 3]]).astype(np.float32)
        },
        "outputs": {
            "Y": None
        }
    },
    "Identity": {
        "graph_name": "Identity_sample",
        "inputs": {
            "X": np.array([[[
                [1, 2],
                [3, 4],]]],dtype=np.float32,)
        },
        "outputs": {
            "Y": None
        }
    },
    "If": { },
    "ImageDecoder": { },
    "InstanceNormalization": {
        "graph_name": "InstanceNormalization_sample",
        "inputs": {
            "x": np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(np.float32),
            "s": np.array([1.0, 1.5]).astype(np.float32),
            "bias": np.array([0, 1]).astype(np.float32)
        },
        "outputs": {
            "y": None
        }
    },
    "IsInf": {
        "graph_name": "IsInf_sample",
        "inputs": {
            "x": np.array([-1.7, np.nan, np.inf, -3.6, -np.inf, np.inf], dtype=np.float32)
        },
        "outputs": {
            "y": np.empty(shape=(6), dtype=bool)
        },
        "detect_negative": 0
    },
    "IsNaN": {
        "graph_name": "IsNaN_sample",
        "inputs": {
            "x": np.array([-1.2, np.nan, np.inf, 2.8, -np.inf, np.inf], dtype=np.float32)
        },
        "outputs": {
            "y": np.empty(shape=(6), dtype=bool)
        }
    },
    "LRN": {
        "graph_name": "LRN_sample",
        "inputs": {
            "x": np.random.randn(5, 5, 5, 5).astype(np.float32)
        },
        "outputs": {
            "y": None
        },
        "alpha": 0.0002,
        "beta": 0.5,
        "bias": 2.0,
        "size": 3
    },
    "LSTM": { },
    "LayerNormalization": { },
    "LeakyRelu": {
        "graph_name": "LeakyRelu_sample",
        "inputs": {
            "x": np.array([-1, 0, 1]).astype(np.float32)
        },
        "outputs": {
            "y": None
        },
        "alpha": 0.1
    },
    "Less": {
        "graph_name": "Less_sample",
        "inputs": {
            "x": np.random.randn(3, 4, 5).astype(np.float32),
            "y": np.random.randn(5).astype(np.float32)
        },
        "outputs": {
            "less": np.empty(shape=(3, 4, 5), dtype=bool)
        }
    },
    "LessOrEqual": {
        "graph_name": "LessOrEqual_sample",
        "inputs": {
            "x": np.random.randn(3, 4, 5).astype(np.float32),
            "y": np.random.randn(5).astype(np.float32)
        },
        "outputs": {
            "less_equal": np.empty(shape=(3, 4, 5), dtype=bool)
        }
    },
    "Log": {
        "graph_name": "Log_sample",
        "inputs": {
            "x": np.array([1, 10]).astype(np.float32)
        },
        "outputs": {
            "y": None
        }
    },
    "LogSoftmax": {
        "graph_name": "LogSoftmax_sample",
        "inputs": {
            "x": np.array([[-1, 0, 1]]).astype(np.float32)
        },
        "outputs": {
            "y": None
        }
    },
    "Loop": { },
    "LpNormalization": { },
    "LpPool": {
        "graph_name": "LpPool_sample",
        "inputs": {
            "x": np.random.randn(1, 3, 32, 32).astype(np.float32)
        },
        "outputs": {
            "y": None
        },
        "kernel_shape": [2, 2],
        "auto_pad": "SAME_UPPER",
        "p": 2,
    },
    "MatMul": {
        "graph_name": "MatMul_sample",
        "inputs": {
            "a": np.random.randn(3, 4).astype(np.float32),
            "b": np.random.randn(4, 3).astype(np.float32)
        },
        "outputs": {
            "c": None
        }
    },
    "MatMulInteger": {
        "graph_name": "MatMulInteger_sample",
        "inputs": {
            "A": np.array([
                        [11, 7, 3],
                        [10, 6, 2],
                        [9, 5, 1],
                        [8, 4, 0],], dtype=np.uint8),
            "B": np.array([
                        [1, 4],
                        [2, 5],
                        [3, 6],], dtype=np.uint8),
            "a_zero_point": np.array([12], dtype=np.uint8),
            "b_zero_point": np.array([0], dtype=np.uint8)
        },
        "outputs": {
            "Y": np.empty(shape=(1), dtype=np.int32)
        }
    },
    "Max": {
        "graph_name": "Max_sample",
        "inputs": {
            "data_0": np.array([3, 2, 1]).astype(np.float32),
            "data_1": np.array([1, 4, 4]).astype(np.float32),
            "data_2": np.array([2, 5, 3]).astype(np.float32)
        },
        "outputs": {
            "result": None
        }
    },
    "MaxPool": { },
    "MaxRoiPool": { },
    "MaxUnpool": {
        "graph_name": "MaxUnpool_sample",
        "inputs": {
            "xT": np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
            "xI": np.array([[[[5, 7], [13, 15]]]], dtype=np.int64)
        },
        "outputs": {
            "y": None
        },
        "kernel_shape": [2, 2],
        "strides": [2, 2]
    },
    "Mean": {
        "graph_name": "Mean_sample",
        "inputs": {
            "data_0": np.array([3, 0, 2]).astype(np.float32),
            "data_1": np.array([1, 3, 4]).astype(np.float32),
            "data_2": np.array([2, 6, 6]).astype(np.float32)
        },
        "outputs": {
            "result": None
        }
    },
    "MeanVarianceNormalization": {
        "graph_name": "MeanVarianceNormalization_sample",
        "inputs": {
            "X": np.array(
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
                ],dtype=np.float32,)
        },
        "outputs": {
            "Y": None
        }
    },
    "MelWeightMatrix": {
        "graph_name": "MelWeightMatrix_sample",
        "inputs": {
            "num_mel_bins": np.array([8]).astype(np.int32),
            "dft_length": np.array([16]).astype(np.int32),
            "sample_rate": np.array([8192]).astype(np.int32),
            "lower_edge_hertz": np.array([0]).astype(np.float32),
            "upper_edge_hertz": np.array([8192 / 2]).astype(np.float32),
        },
        "outputs": {
            "output": np.empty(shape=(3, 4, 5), dtype=np.float32)
        },
    },
    "Min": {
        "graph_name": "Min_sample",
        "inputs": {
            "data_0": np.array([3, 2, 1]).astype(np.float32),
            "data_1": np.array([1, 4, 4]).astype(np.float32),
            "data_2": np.array([2, 5, 3]).astype(np.float32)
        },
        "outputs": {
            "result": None
        }
    },
    "Mish": {
        "graph_name": "Mish_sample",
        "inputs": {
            "X": np.linspace(-10, 10, 10000, dtype=np.float32)
        },
        "outputs": {
            "Y": None
        }
    },
    "Mod": {
        "graph_name": "Mod_sample",
        "inputs": {
            "x": np.array([-4.3, 7.2, 5.0, 4.3, -7.2, 8.0]).astype(np.float32),
            "y": np.array([2.1, -3.4, 8.0, -2.1, 3.4, 5.0]).astype(np.float32)
        },
        "outputs": {
            "Y": None
        },
        "fmod": 1
    },
    "Mul": {
        "graph_name": "Mul_sample",
        "inputs": {
            "x": np.array([1, 2, 3]).astype(np.float32),
            "y": np.array([4, 5, 6]).astype(np.float32)
        },
        "outputs": {
            "z": None
        }
    },
    "Multinomial": { },
    "Neg": {
        "graph_name": "Neg_sample",
        "inputs": {
            "x": np.array([-4, 2]).astype(np.float32)
        },
        "outputs": {
            "y": None
        }
    },
    "NegativeLogLikelihoodLoss": {
        "graph_name": "NegativeLogLikelihoodLoss_sample",
        "inputs": {
            "input": np.random.rand(3, 5, 6, 6).astype(np.float32),
            "target": np.random.randint(0, high=5, size=(3, 6, 6)).astype(np.int64),
            "weight": np.random.rand(5).astype(np.float32)
        },
        "outputs": {
            "y": None
        },
        "reduction": "sum"
    },
    "NonMaxSuppression": {
        "graph_name": "LpPool_sample",
        "inputs": {
            "boxes": np.array([[
                        [0.5, 0.5, 1.0, 1.0],
                        [0.5, 0.6, 1.0, 1.0],
                        [0.5, 0.4, 1.0, 1.0],
                        [0.5, 10.5, 1.0, 1.0],
                        [0.5, 10.6, 1.0, 1.0],
                        [0.5, 100.5, 1.0, 1.0],]]).astype(np.float32),
            "scores": np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32),
            "max_output_boxes_per_class": np.array([3]).astype(np.int64),
            "iou_threshold": np.array([0.5]).astype(np.float32),
            "score_threshold": np.array([0.0]).astype(np.float32),
        },
        "outputs": {
            "y": np.empty(shape=(1), dtype=np.int64)
        },
        "center_point_box": 1,
    },
    "NonZero": {
        "graph_name": "NonZero_sample",
        "inputs": {
            "condition": np.array([[1, 0], [1, 1]], dtype=bool)
        },
        "outputs": {
            "result": np.empty(shape=(1), dtype=np.int64)
        }
    },
    "Not": {
        "graph_name": "Not_sample",
        "inputs": {
            "x": (np.random.randn(3, 4, 5) > 0).astype(bool)
        },
        "outputs": {
            "not": None
        }
    },
    "OneHot": {
        "graph_name": "OneHot_sample",
        "inputs": {
            "indices": np.array([[1, 9], [2, 4]], dtype=np.float32),
            "depth": np.array([10], dtype=np.float32),
            "values": np.array([1, 3], dtype=np.float32)
        },
        "outputs": {
            "y": None
        },
        "axis": 1,
    },
    "Optional": { },
    "OptionalGetElement": { },
    "OptionalHasElement": { },
    "Or": {
        "graph_name": "Or_sample",
        "inputs": {
            "x": (np.random.randn(3, 4, 5) > 0).astype(bool),
            "y": (np.random.randn(3, 4, 5) > 0).astype(bool)
        },
        "outputs": {
            "z": None
        }
    },
    "PRelu": {
        "graph_name": "PRelu_sample",
        "inputs": {
            "x": np.random.randn(3, 4, 5).astype(np.float32),
            "slope": np.random.randn(3, 4, 5).astype(np.float32)
        },
        "outputs": {
            "y": None
        }
    },
    "Pad": {
        "graph_name": "Pad_sample",
        "inputs": {
            "x": np.random.randn(1, 3, 4, 5).astype(np.float32),
            "pads": np.array([0, 3, 0, 4]).astype(np.int64),
            "value": np.array(1.2, dtype=np.float32),
            "axes": np.array([1, 3], dtype=np.int64)
        },
        "outputs": {
            "y": None
        },
        "mode": "constant"
    },
    "Pow": {
        "graph_name": "Pow_sample",
        "inputs": {
            "x": np.array([1, 2, 3]).astype(np.float32),
            "y": np.array([4, 5, 6]).astype(np.float32)
        },
        "outputs": {
            "z": None
        }
    },
    "QLinearConv": {
        "graph_name": "QLinearConv_sample",
        "inputs": {
            "x": np.array([
                    [255, 174, 162, 25, 203, 168, 58],
                    [15, 59, 237, 95, 129, 0, 64],
                    [56, 242, 153, 221, 168, 12, 166],
                    [232, 178, 186, 195, 237, 162, 237],
                    [188, 39, 124, 77, 80, 102, 43],
                    [127, 230, 21, 83, 41, 40, 134],
                    [255, 154, 92, 141, 42, 148, 247],],dtype=np.uint8,).reshape((1, 1, 7, 7)),
            "x_scale": np.array(0.00369204697, dtype=np.float32),
            "x_zero_point": np.array(132, dtype=np.uint8),
            "w": np.array([0], dtype=np.uint8).reshape((1, 1, 1, 1)),
            "w_scale": np.array([0.00172794575], dtype=np.float32),
            "w_zero_point": np.array([255], dtype=np.uint8),
            "y_scale": np.array(0.00162681262, dtype=np.float32),
            "y_zero_point": np.array(123, dtype=np.uint8),
        },
        "outputs": {
            "y": None
        },
    },
    "QLinearMatMul": {
        "graph_name": "QLinearMatMul_sample",
        "inputs": {
            "a": np.array([[208, 236, 0, 238], [3, 214, 255, 29]]).astype(np.uint8),
            "a_scale": np.array([0.0066], dtype=np.float32),
            "a_zero_point": np.array([113], dtype=np.uint8),
            "b": np.array([[152, 51, 244], [60, 26, 255], [0, 127, 246], [127, 254, 247]]).astype(np.uint8),
            "b_scale": np.array([0.00705], dtype=np.float32),
            "b_zero_point": np.array([114], dtype=np.uint8),
            "y_scale": np.array([0.0107], dtype=np.float32),
            "y_zero_point": np.array([118], dtype=np.uint8)
        },
        "outputs": {
            "y": None
        },
    },
    "QuantizeLinear": {
        "graph_name": "QuantizeLinear_sample",
        "inputs": {
            "x": np.array([
                        [6.0, -8, -10, 5.0],
                        [1.0, 8.0, 4.0, 5.0],
                        [0.0, 20.0, 10.0, 4.0],],dtype=np.float32),
            "y_scale": np.array([
                        [1.5, 2.5],
                        [3.0, 4.9],
                        [5.1, 6.9],],dtype=np.float32)
        },
        "outputs": {
            "y": np.empty(shape=(1), dtype=np.uint8)
        },
        "axis": 1,
        "block_size": 2,
        "output_dtype": onnx.TensorProto.UINT8,
    },
    "RNN": { },
    "RandomNormal": { },
    "RandomNormalLike": { },
    "RandomUniform": { },
    "RandomUniformLike": { },
    "Range": {
        "graph_name": "Range_sample",
        "inputs": {
            "start": np.array([1]).astype(np.float32),
            "limit": np.array([5]).astype(np.float32),
            "delta": np.array([2]).astype(np.float32)
        },
        "outputs": {
            "output": None
        }
    },
    "Reciprocal": {
        "graph_name": "Reciprocal_sample",
        "inputs": {
            "x": np.array([-4, 2]).astype(np.float32),
        },
        "outputs": {
            "y": None
        }
    },
    "ReduceL1": {
        "graph_name": "ReduceL1_sample",
        "inputs": {
            "data": np.reshape(np.arange(1, np.prod([3, 2, 2]) + 1, dtype=np.float32), [3, 2, 2]),
            "axes": np.array([], dtype=np.int64)
        },
        "outputs": {
            "reduced": None
        },
        "keepdims": 1
    },
    "ReduceL2": {
        "graph_name": "ReduceL2_sample",
        "inputs": {
            "data": np.reshape(np.arange(1, np.prod([3, 2, 2]) + 1, dtype=np.float32), [3, 2, 2]),
            "axes": np.array([], dtype=np.int64)
        },
        "outputs": {
            "reduced": None
        },
        "keepdims": 1
    },
    "ReduceLogSum": {
        "graph_name": "ReduceLogSum_sample",
        "inputs": {
            "data": np.array([], dtype=np.float32).reshape([2, 0, 4]),
            "axes": np.array([1], dtype=np.int64)
        },
        "outputs": {
            "reduced": None
        },
        "keepdims": 1
    },
    "ReduceLogSumExp": {
        "graph_name": "ReduceLogSumExp_sample",
        "inputs": {
            "data": np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32),
            "axes": np.array([], dtype=np.int64)
        },
        "outputs": {
            "reduced": None
        },
        "keepdims": 1
    },
    "ReduceMax": {
        "graph_name": "ReduceMax_sample",
        "inputs": {
            "data": np.array([[True, True], [True, False], [False, True], [False, False]],),
            "axes": np.array([1], dtype=np.int64)
        },
        "outputs": {
            "reduced": None
        },
        "keepdims": 1
    },
    "ReduceMean": {
        "graph_name": "ReduceMean_sample",
        "inputs": {
            "data": np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],dtype=np.float32,),
            "axes": np.array([], dtype=np.int64)
        },
        "outputs": {
            "reduced": None
        },
        "keepdims": 1
    },
    "ReduceMin": {
        "graph_name": "ReduceMin_sample",
        "inputs": {
            "data": np.array([[True, True], [True, False], [False, True], [False, False]],),
            "axes": np.array([1], dtype=np.int64)
        },
        "outputs": {
            "reduced": None
        },
        "keepdims": 1
    },
    "ReduceProd": {
        "graph_name": "ReduceProd_sample",
        "inputs": {
            "data": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
        },
        "outputs": {
            "reduced": None
        },
        "keepdims": 1
    },
    "ReduceSum": {
        "graph_name": "ReduceSum_sample",
        "inputs": {
            "data": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32),
            "axes": np.array([], dtype=np.int64)
        },
        "outputs": {
            "reduced": None
        },
        "keepdims": 1
    },
    "ReduceSumSquare": {
        "graph_name": "ReduceSumSquare_sample",
        "inputs": {
            "data": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32),
            "axes": np.array([], dtype=np.int64)
        },
        "outputs": {
            "reduced": None
        },
        "keepdims": 1
    },
    "RegexFullMatch": {
        "graph_name": "RegexFullMatch_sample",
        "inputs": {
            "X": np.array(["www.google.com", "www.facebook.com", "www.bbc.co.uk"]).astype(object)
        },
        "outputs": {
            "Y": np.empty(shape=(1), dtype=bool)
        },
        "pattern": r"www\.[\w.-]+\.\bcom\b"
    },
    "Relu": {
        "graph_name": "Relu_sample",
        "inputs": {
            "x": np.random.randn(3, 4, 5).astype(np.float32)
        },
        "outputs": {
            "y": None
        }
    },
    "Reshape": {
        "graph_name": "Reshape_sample",
        "inputs": {
            "X1": np.random.random_sample([2, 3, 4]).astype(np.float32),
            "X2": np.array([4, 2, 3], dtype=np.int64)
        },
        "outputs": {
            "Y": None
        },
        "allowzero": 1
    },
    "Resize": {
        "graph_name": "Resize_sample",
        "inputs": {
            "X": np.array([[[
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]]],dtype=np.float32,),
            "": "",
            "scales": np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)
        },
        "outputs": {
            "Y": None
        },
        "mode": "cubic"
    },
    "ReverseSequence": {
        "graph_name": "ReverseSequence_sample",
        "inputs": {
            "x": np.array([
                        [0.0, 1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0, 7.0],
                        [8.0, 9.0, 10.0, 11.0],
                        [12.0, 13.0, 14.0, 15.0],],dtype=np.float32,),
            "sequence_lens": np.array([1, 2, 3, 4], dtype=np.int64)
        },
        "outputs": {
            "y": None
        },
        "time_axis": 1,
        "batch_axis": 0
    },
    "RoiAlign": {
        "graph_name": "RoiAlign_sample",
        "inputs": {
            "X": np.array([[[
                            [0.2764, 0.7150, 0.1958, 0.3416, 0.4638, 0.0259, 0.2963, 0.6518, 0.4856, 0.7250,],
                            [0.9637, 0.0895, 0.2919, 0.6753, 0.0234, 0.6132, 0.8085, 0.5324, 0.8992, 0.4467,],
                            [0.3265, 0.8479, 0.9698, 0.2471, 0.9336, 0.1878, 0.4766, 0.4308, 0.3400, 0.2162,],
                            [0.0206, 0.1720, 0.2155, 0.4394, 0.0653, 0.3406, 0.7724, 0.3921, 0.2541, 0.5799,],
                            [0.4062, 0.2194, 0.4473, 0.4687, 0.7109, 0.9327, 0.9815, 0.6320, 0.1728, 0.6119,],
                            [0.3097, 0.1283, 0.4984, 0.5068, 0.4279, 0.0173, 0.4388, 0.0430, 0.4671, 0.7119,],
                            [0.1011, 0.8477, 0.4726, 0.1777, 0.9923, 0.4042, 0.1869, 0.7795, 0.9946, 0.9689,],
                            [0.1366, 0.3671, 0.7011, 0.6234, 0.9867, 0.5585, 0.6985, 0.5609, 0.8788, 0.9928,],
                            [0.5697, 0.8511, 0.6711, 0.9406, 0.8751, 0.7496, 0.1650, 0.1049, 0.1559, 0.2514,],
                            [0.7012, 0.4056, 0.7879, 0.3461, 0.0415, 0.2998, 0.5094, 0.3727, 0.5482, 0.0502,],]]],dtype=np.float32,),
            "rois": np.array([[0, 0, 9, 9], [0, 5, 4, 9], [5, 5, 9, 9]], dtype=np.float32),
            "batch_indices": np.array([0, 0, 0], dtype=np.int64),
        },
        "outputs": {
            "y": None
        },
        "spatial_scale": 1.0,
        "output_height": 5,
        "output_width": 5,
        "sampling_ratio": 2,
        "coordinate_transformation_mode": "half_pixel",
    },
    "Round": {
        "graph_name": "Round_sample",
        "inputs": {
            "x": np.array([0.1, 0.5, 0.9, 1.2, 1.5, 1.8, 2.3, 2.5, 2.7, -1.1, -1.5, -1.9, -2.2,  -2.5, -2.8,]).astype(np.float32)
        },
        "outputs": {
            "y": None
        }
    },
    "STFT": { },
    "Scan": { },
    "Scatter": { },
    "ScatterElements": {
        "graph_name": "ScatterElements_sample",
        "inputs": {
            "data": np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
            "indices": np.array([[1, 1]], dtype=np.int64),
            "updates": np.array([[1.1, 2.1]], dtype=np.float32)
        },
        "outputs": {
            "y": None
        },
        "axis": 1,
        "reduction": "add",
    },
    "ScatterND": {
        "graph_name": "ScatterND_sample",
        "inputs": {
            "data": np.array([
                    [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                    [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                    [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
                    [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],],dtype=np.float32,),
            "indices": np.array([[0], [0]], dtype=np.int64),
            "updates": np.array([
                    [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                    [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],],dtype=np.float32,)
        },
        "outputs": {
            "y": None
        },
        "reduction": "mul",
    },
    "Selu": {
        "graph_name": "Selu_sample",
        "inputs": {
            "x": np.array([-1, 0, 1]).astype(np.float32)
        },
        "outputs": {
            "y": None
        },
        "alpha": 2.0,
        "gamma": 3.0
    },
    "SequenceAt": { },
    "SequenceConstruct": { },
    "SequenceEmpty": { },
    "SequenceErase": { },
    "SequenceInsert": { },
    "SequenceLength": { },
    "SequenceMap": { },
    "Shape": {
        "graph_name": "Shape_sample",
        "inputs": {
            "x": np.random.randn(3, 4, 5).astype(np.float32)
        },
        "outputs": {
            "y": np.empty(shape=(1), dtype=np.int64)
        },
        "start": 1,
        "end": 2,
    },
    "Shrink": {
        "graph_name": "Shrink_sample",
        "inputs": {
            "x": np.arange(-2.0, 2.1, dtype=np.float32)
        },
        "outputs": {
            "y": None
        },
        "lambd": 1.5,
        "bias": 1.5
    },
    "Sigmoid": {
        "graph_name": "Sigmoid_sample",
        "inputs": {
            "x": np.array([-1, 0, 1]).astype(np.float32)
        },
        "outputs": {
            "y": None
        }
    },
    "Sign": {
        "graph_name": "Sign_sample",
        "inputs": {
            "x": np.array(range(-5, 6)).astype(np.float32)
        },
        "outputs": {
            "y": None
        }
    },
    "Sin": {
        "graph_name": "Sin_sample",
        "inputs": {
            "x": np.array([-1, 0, 1]).astype(np.float32)
        },
        "outputs": {
            "y": None
        }
    },
    "Sinh": {
        "graph_name": "Sinh_sample",
        "inputs": {
            "x": np.array([-1, 0, 1]).astype(np.float32)
        },
        "outputs": {
            "y": None
        }
    },
    "Size": {
        "graph_name": "Size_sample",
        "inputs": {
            "x": np.array([
                        [1, 2, 3],
                        [4, 5, 6],]).astype(np.float32)
        },
        "outputs": {
            "y": np.empty(shape=(1), dtype=np.int64)
        }
    },
    "Slice": {
        "graph_name": "Slice_sample",
        "inputs": {
            "x": np.random.randn(20, 10, 5).astype(np.float32),
            "starts": np.array([0, 0], dtype=np.int64),
            "ends": np.array([3, 10], dtype=np.int64),
            "axes": np.array([0, 1], dtype=np.int64),
            "steps": np.array([1, 1], dtype=np.int64)
        },
        "outputs": {
            "y": None
        },
    },
    "Softmax": {
        "graph_name": "Softmax_sample",
        "inputs": {
            "x": np.abs(np.random.randn(3, 4, 5).astype(np.float32))
        },
        "outputs": {
            "y": None
        },
        "axis": 0
    },
    "SoftmaxCrossEntropyLoss": { },
    "Softplus": {
        "graph_name": "Softplus_sample",
        "inputs": {
            "x": np.array([-1, 0, 1]).astype(np.float32)
        },
        "outputs": {
            "y": None
        }
    },
    "Softsign": {
        "graph_name": "Softsign_sample",
        "inputs": {
            "x": np.array([-1, 0, 1]).astype(np.float32)
        },
        "outputs": {
            "y": None
        }
    },
    "SpaceToDepth": {
        "graph_name": "SpaceToDepth_sample",
        "inputs": {
            "x": np.array([[[
                    [0, 6, 1, 7, 2, 8],
                    [12, 18, 13, 19, 14, 20],
                    [3, 9, 4, 10, 5, 11],
                    [15, 21, 16, 22, 17, 23],
                ]]]).astype(np.float32)
        },
        "outputs": {
            "y": None
        },
        "blocksize": 2
    },
    "Split": { },
    "SplitToSequence": { },
    "Sqrt": {
        "graph_name": "Sqrt_sample",
        "inputs": {
            "x": np.array([1, 4, 9]).astype(np.float32)
        },
        "outputs": {
            "y": None
        }
    },
    "Squeeze": {
        "graph_name": "Squeeze_sample",
        "inputs": {
            "x": np.random.randn(1, 3, 4, 5).astype(np.float32),
            "axes": np.array([0], dtype=np.int64)
        },
        "outputs": {
            "y": None
        }
    },
    "StringConcat": {
        "graph_name": "StringConcat_sample",
        "inputs": {
            "x": np.array(["abc", "def"]).astype("object"),
            "y": np.array([".com", ".net"]).astype("object")
        },
        "outputs": {
            "result": None
        }
    },
    "StringNormalizer": {
        "graph_name": "StringNormalizer_sample",
        "inputs": {
            "x": np.array(["monday", "tuesday", "wednesday", "thursday"]).astype(object)
        },
        "outputs": {
            "y": None
        },
        "case_change_action": "LOWER",
        "is_case_sensitive": 1,
        "stopwords": ["monday"],
    },
    "StringSplit": { },
    "Sub": {
        "graph_name": "Sub_sample",
        "inputs": {
            "x": np.array([1, 2, 3]).astype(np.float32),
            "y": np.array([3, 2, 1]).astype(np.float32)
        },
        "outputs": {
            "z": None
        }
    },
    "Sum": {
        "graph_name": "Sum_sample",
        "inputs": {
            "x1": np.array([3, 0, 2]).astype(np.float32),
            "x2": np.array([1, 3, 4]).astype(np.float32),
            "x3": np.array([2, 6, 6]).astype(np.float32)
        },
        "outputs": {
            "result": None
        }
    },
    "Tan": {
        "graph_name": "Tan_sample",
        "inputs": {
            "x": np.array([-1, 0, 1]).astype(np.float32)
        },
        "outputs": {
            "y": None
        }
    },
    "Tanh": {
        "graph_name": "Tanh_sample",
        "inputs": {
            "x": np.array([-1, 0, 1]).astype(np.float32)
        },
        "outputs": {
            "y": None
        }
    },
    "TfIdfVectorizer": { },
    "ThresholdedRelu": {
        "graph_name": "ThresholdedRelu_sample",
        "inputs": {
            "x": np.array([-1.5, 0.0, 1.2, 2.0, 2.2]).astype(np.float32)
        },
        "outputs": {
            "y": None
        },
        "alpha": 2.0
    },
    "Tile": {
        "graph_name": "Tile_sample",
        "inputs": {
            "x": np.array([[0, 1], [2, 3]], dtype=np.float32),
            "repeat": np.array([1, 2], dtype=np.int64)
        },
        "outputs": {
            "z": None
        }
    },
    "TopK": { },
    "Transpose": {
        "graph_name": "Transpose_sample",
        "inputs": {
            "x": np.random.random_sample((2, 3, 4)).astype(np.float32)
        },
        "outputs": {
            "y": None
        },
        "perm": (0, 2, 1)
    },
    "Trilu": {
        "graph_name": "Trilu_sample",
        "inputs": {
            "x": np.random.randint(10, size=(2, 3, 3)).astype(np.int64),
            "k": np.array(-1).astype(np.int64)
        },
        "outputs": {
            "y": None
        },
        "upper": 0
    },
    "Unique": { },
    "Unsqueeze": {
        "graph_name": "Unsqueeze_sample",
        "inputs": {
            "x": np.random.randn(1, 3, 1, 5).astype(np.float32),
            "axes": np.array([-2]).astype(np.int64)
        },
        "outputs": {
            "y": None
        }
    },
    "Upsample": { },
    "Where": {
        "graph_name": "Where_sample",
        "inputs": {
            "condition": np.array([[1, 0], [1, 1]], dtype=bool),
            "x": np.array([[1, 2], [3, 4]], dtype=np.int64),
            "y": np.array([[9, 8], [7, 6]], dtype=np.int64)
        },
        "outputs": {
            "z": np.empty(shape=(1), dtype=np.int64)
        }
    },
    "Xor": {
        "graph_name": "Xor_sample",
        "inputs": {
            "x": (np.random.randn(3, 4, 5) > 0).astype(bool),
            "y": (np.random.randn(3, 4, 5) > 0).astype(bool)
        },
        "outputs": {
            "z": None
        }
    },
}

def main():
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", "-op", help="Operator name", default="")
    args = parser.parse_args()

    operator_processing = getattr(ONMAOperators, "ONNX_CreateNetworkWithOperator")
    operator_processing(args.operator, **default_input[args.operator])

if main() == False:
    sys.exit(-1)
