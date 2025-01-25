import sys
import onnx
import argparse
import numpy as np
import json
from ONMAModel import ONMAModel
from onnx.backend.test.case.node.affinegrid import create_theta_2d
from onnx.backend.test.case.node.roialign import get_roi_align_input_values
from onnx.backend.test.case.node.layernormalization import calculate_normalized_shape

global args

then_out = onnx.helper.make_tensor_value_info(
    "then_out", onnx.TensorProto.FLOAT, [5]
)
else_out = onnx.helper.make_tensor_value_info(
    "else_out", onnx.TensorProto.FLOAT, [5]
)

x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
y = np.array([5, 4, 3, 2, 1]).astype(np.float32)

then_const_node = onnx.helper.make_node(
    "Constant",
    inputs=[],
    outputs=["then_out"],
    value=onnx.numpy_helper.from_array(x),
)

else_const_node = onnx.helper.make_node(
    "Constant",
    inputs=[],
    outputs=["else_out"],
    value=onnx.numpy_helper.from_array(y),
)

then_body = onnx.helper.make_graph(
    [then_const_node], "then_body", [], [then_out]
)

else_body = onnx.helper.make_graph(
    [else_const_node], "else_body", [], [else_out]
)

default_input = \
{
    "Abs": {
        "graph_name": "Reshape_sample",
        "inputs": {
            "X": {
                "data": [[0.06329948, -1.0832994 , 0.37930292], [0.71035045, -1.6637981 , 1.0044696]],
                "type": "float64"
            }
        },
        "outputs": {
            "Y": {
                "data": [[0.06329948, 1.0832994 , 0.37930292], [0.71035045, 1.6637981 , 1.0044696]],
                "type": "float64"
            }
        },
        "Abs_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Abs",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "Acos": {
        "graph_name": "Acos_sample",
        "inputs": {
            "X": {
                "data": [[0.06329948, 0.0832994 , 0.37930292], [0.71035045, 0.6637981, 0.0044696]]
            }
        },
        "outputs": {
            "Y": {
                "data": [[1.5074545, 1.4874003, 1.1817535], [0.78080034, 0.8449107, 1.5663267 ]]
            }
        },
        "Acos_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Acos",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "Acosh": {
        "graph_name": "Acosh_sample",
        "inputs": {
            "X": {
                "data": [[ 10, np.e, 1]],
            }
        },
        "outputs": {
            "Y": {
                "data": [[ 2.993223, 1.6574545, 0.0]],
            }
        },
        "Acosh_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Acosh",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "Add": {
        "graph_name": "Add_sample",
        "inputs": {
            "X1": {"data": [[ 0.06329948, -1.0832994 ,  0.37930292], [ 0.71035045, -1.6637981 ,  1.0044696 ]]},
            "X2": { "data":[[ 0.71035045, 0.0832994 ,  1.37930292], [ 0.71035045, -1.6637981 ,  1.0044696 ]]},
        },
        "outputs": {
            "Y": { "data": [[ 0.77364993, -1.0, 1.758606  ], [1.4207009, -3.3275962,   2.0089393 ]]}
        },
        "Add_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Add",
            "inputs": {
                "x1": "X1",
                "x2": "X2"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "AffineGrid": {
        "graph_name": "AffineGrid_sample",
        "inputs": {
            "theta": {"data": [[[ 1.0889444, -3.2880466, 5.0], [ 2.0223253, 1.0960155, -3.3]], [[ 0.83578837, -0.55442286, 2.5],[ 0.78762794, 0.8397114, 1.1, ]]]},
            "size": {"data": [2, 3, 5, 6], "type": "int64"}
        },
        "outputs": {
            "Y": {"data": [[[[ 6.722984,   -5.8620834 ], [ 7.085965,   -5.187975  ], [ 7.448947,   -4.5138664 ], [ 7.8119283,  -3.839758  ], [ 8.17491,    -3.1656497 ], [ 8.537891,   -2.4915414 ]],
                            [[ 5.407765,   -5.4236774 ] ,[ 5.770746,   -4.749569  ] ,[ 6.133728,   -4.0754604 ] ,[ 6.4967093,  -3.401352  ] ,[ 6.8596907,  -2.7272434 ] ,[ 7.2226725,  -2.0531352 ]],
                            [[ 4.0925465,  -4.985271  ] ,[ 4.455528,   -4.3111625 ] ,[ 4.818509,   -3.6370542 ] ,[ 5.181491,   -2.9629457 ] ,[ 5.544472,   -2.2888374 ] ,[ 5.9074535,  -1.6147289 ]],
                            [[ 2.7773275,  -4.5468645 ] ,[ 3.140309,   -3.8727565 ] ,[ 3.5032907,  -3.198648  ] ,[ 3.866272,   -2.5245395 ] ,[ 4.229254,   -1.8504311 ] ,[ 4.592235,   -1.1763227 ]],
                            [[ 1.4621091,  -4.1084585 ] ,[ 1.8250904,  -3.4343503 ] ,[ 2.188072,   -2.7602417 ] ,[ 2.5510535,  -2.0861332 ] ,[ 2.9140348,  -1.412025  ] ,[ 3.2770162,  -0.73791647]]],
                           [[[ 2.247048,   -0.22812569] ,[ 2.525644,    0.03441691] ,[ 2.8042402,   0.29695958] ,[ 3.0828364,   0.55950224] ,[ 3.3614326,   0.82204485] ,[ 3.6400285,   1.0845875 ]],
                            [[ 2.0252788,   0.10775888] ,[ 2.303875,    0.37030149] ,[ 2.5824711,   0.63284415] ,[ 2.8610673,   0.8953868 ] ,[ 3.1396632,   1.1579294 ] ,[ 3.4182594,   1.420472  ]],
                            [[ 1.8035097,   0.44364345] ,[ 2.0821059,   0.70618606] ,[ 2.360702,    0.9687287 ] ,[ 2.6392982,   1.2312714 ] ,[ 2.9178941,   1.493814  ] ,[ 3.1964903,   1.7563566 ]],
                            [[ 1.5817406,   0.779528  ] ,[ 1.8603367,   1.0420706 ] ,[ 2.1389327,   1.3046132 ] ,[ 2.4175289,   1.567156  ] ,[ 2.696125,    1.8296986 ] ,[ 2.9747212,   2.0922413 ]],
                            [[ 1.3599714,   1.1154126 ] ,[ 1.6385676,   1.3779552 ] ,[ 1.9171636,   1.6404979 ] ,[ 2.1957598,   1.9030405 ] ,[ 2.474356,    2.1655831 ] ,[ 2.752952,    2.4281259 ]]]]}
        },
        "AffineGrid_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "AffineGrid",
            "inputs": {
                "theta": "theta",
                "size": "size"
            },
            "outputs": {
                "y": "Y"
            },
            "align_corners": 0
        }
    },
    "And": {
        "graph_name": "And_sample",
        "inputs": {
            "X1": { "data": (np.random.randn(3, 4, 5) > 0).astype(bool), "type": "bool"},
            "X2": { "data": (np.random.randn(3, 4, 5) > 0).astype(bool), "type": "bool"},
        },
        "outputs": {
            "Y": {"data": [[[False, False, False,  True, False], [ True,  True, False,  True,  True], [False, False, False, False, False], [False,  True, True, False, False]],
                           [[ True, False,  True, False,  True], [ True, False, False, False,  True], [False, False, False, False, False], [ True,  True,  True, False,  True]],
                           [[ True, False,  True,  True, False], [False,  True, False, False,  True], [False,  True, False,  True,  True], [ True, False,  True,  True,  True]]],
                   "type": "bool"
                }
        },
        "And_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "And",
            "inputs": {
                "x1": "X1",
                "x2": "X1"
            },
            "outputs": {
                "y": "Y"
            },
        }
    },
    "ArgMax": {
        "graph_name": "ArgMax_sample",
        "inputs": {
            "data": {"data": [[2, 2], [3, 10]]}
        },
        "outputs": {
            "result": {"data": [1, 1], "type": "int64"}
        },
        "ArgMax_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "ArgMax",
            "inputs": {
                "x": "data"
            },
            "outputs": {
                "y": "result"
            },
            "axis": 0,
            "keepdims": 0,
            "select_last_index": True,
        }
    },
    "ArgMin": {
        "graph_name": "ArgMin_sample",
        "inputs": {
            "data": {"data": [[2, 2], [3, 10]]}
        },
        "outputs": {
            "result": {"data": [[1], [0]], "type": "int64"}
        },
        "ArgMin_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "ArgMin",
            "inputs": {
                "x": "data"
            },
            "outputs": {
                "y": "result"
            },
            "axis": 1,
            "keepdims": 1,
            "select_last_index": True,
        }
    },
    "Asin": {
        "graph_name": "Asin_sample",
        "inputs": {
            "X": {"data": [[[0.2724369,  0.3790569,  0.3742962,  0.74878824, 0.23780724], [0.1718531,  0.44929165, 0.3044684,  0.8391891,  0.23774183], [0.50238943, 0.9425836,  0.6339977,  0.8672894,  0.9402097 ], [0.75076485, 0.69957507, 0.96796554, 0.9944008,  0.45182168]],
                           [[0.07086978, 0.29279402, 0.1523547,  0.41748637, 0.13128933], [0.6041178,  0.38280806, 0.89538586, 0.96779466, 0.5468849 ], [0.27482358, 0.59223044, 0.8967612,  0.40673333, 0.55207825], [0.27165276, 0.45544416, 0.40171355, 0.24841346, 0.5058664 ]],
                           [[0.31038082, 0.37303486, 0.5249705,  0.75059503, 0.33350748], [0.92415875, 0.8623186,  0.0486903,  0.25364253, 0.44613552], [0.10462789, 0.348476,   0.7400975,  0.68051445, 0.6223844 ], [0.7105284,  0.20492369, 0.3416981,  0.6762425,  0.8792348 ]]]}
        },
        "outputs": {
            "Y": {"data": [[[0.2759248,  0.38877693, 0.38363767, 0.84623194, 0.2401077 ], [0.17271045, 0.4659723,  0.30938026, 0.9957905,  0.24004036], [0.52636003, 1.2302837,  0.6867117,  1.0497311,  1.2232454 ], [0.8492192,  0.7748026,  1.3169973,  1.4649243,  0.4688063 ]],
                           [[0.07092924, 0.2971476,  0.15295035, 0.43067732, 0.13166946], [0.6486584,  0.39283398, 1.109297,   1.3163176,  0.5786389 ], [0.2784062,  0.6338241,  1.1123953,  0.4188754,  0.5848547 ], [0.27510995, 0.47287107, 0.41338724, 0.25104204, 0.5303861 ]],
                           [[0.3155936,  0.38227785, 0.55268043, 0.8489621,  0.3400216 ], [1.1788275,  1.0398308,  0.04870956, 0.25644407, 0.46244264], [0.10481973, 0.3559447,  0.83321536, 0.7484645,  0.6717854 ], [0.7902488,  0.20638575, 0.34872317, 0.74265,    1.0742536 ]]]}
        },
        "Asin_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Asin",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "Asinh": {
        "graph_name": "Asinh_sample",
        "inputs": {
            "X": {"data": [[[0.2724369 , 0.3790569 , 0.3742962 , 0.74878824, 0.23780724], [0.1718531 , 0.44929165, 0.3044684 , 0.8391891 , 0.23774183], [0.50238943, 0.9425836 , 0.6339977 , 0.8672894 , 0.9402097 ], [0.75076485, 0.69957507, 0.96796554, 0.9944008 , 0.45182168]],
                          [[0.07086978, 0.29279402, 0.1523547 , 0.41748637, 0.13128933], [0.6041178 , 0.38280806, 0.89538586, 0.96779466, 0.5468849 ], [0.27482358, 0.59223044, 0.8967612 , 0.40673333, 0.55207825], [0.27165276, 0.45544416, 0.40171355, 0.24841346, 0.5058664 ]],
                          [[0.31038082, 0.37303486, 0.5249705 , 0.75059503, 0.33350748], [0.92415875, 0.8623186 , 0.0486903 , 0.25364253, 0.44613552], [0.10462789, 0.348476  , 0.7400975 , 0.68051445, 0.6223844 ], [0.7105284 , 0.20492369, 0.3416981 , 0.6762425 , 0.8792348 ]]]}
        },
        "outputs": {
            "Y": {"data": [[[0.26917458, 0.37052065, 0.36606553, 0.6921775 , 0.235621  ],
                            [0.17101826, 0.43540362, 0.29995036, 0.7629782 , 0.23555738],
                            [0.48334798, 0.8401858 , 0.5977583 , 0.78435487, 0.83845735],
                            [0.69375896, 0.6523185 , 0.8585394 , 0.8774088 , 0.43771034]],

                        [[0.07081059, 0.28876418, 0.15177137, 0.40622163, 0.13091506],
                            [0.57235265, 0.3740261 , 0.80543333, 0.8584167 , 0.5227491 ],
                            [0.27147663, 0.5621512 , 0.80645764, 0.39627978, 0.52730066],
                            [0.26841795, 0.44100925, 0.39162582, 0.24592699, 0.48645276]],

                        [[0.30560175, 0.364884  , 0.5034339 , 0.6936231 , 0.32761535],
                            [0.8267165 , 0.78059494, 0.04867108, 0.2509987 , 0.43252304],
                            [0.10443793, 0.34178278, 0.68520635, 0.6366303 , 0.58792436],
                            [0.66127044, 0.20351589, 0.33537567, 0.633095  , 0.79335237]]]}
        },
        "Asinh_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Asinh",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "Atan": {
        "graph_name": "Atan_sample",
        "inputs": {
            "X": {"data": np.random.rand(3, 4, 5).astype(np.float32)}
        },
        "outputs": {
            "Y": None
        },
        "Atan_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Atan",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "Atanh": {
        "graph_name": "Atanh_sample",
        "inputs": {
            "X": {"data": np.random.rand(3, 4, 5).astype(np.float32)}
        },
        "outputs": {
            "Y": None
        },
        "Atanh_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Atanh",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "AveragePool": {
        "graph_name": "AveragePool_sample",
        "inputs": {
            "X": {"data": np.random.randn(1, 1, *(32, 32, 32)).astype(np.float32)}
        },
        "outputs": {
            "Y": None
        },
        "AveragePool_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "AveragePool",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            },
            "kernel_shape": (5, 5, 5),
            "strides": (3, 3, 3),
            "dilations": (2, 2, 2),
            "count_include_pad": 0,
            "ceil_mode": True,
            "auto_pad": "SAME_UPPER"
        }
    },
    "BatchNormalization": {
        "graph_name": "BatchNormalization_sample",
        "inputs": {
            "x": {"data": np.random.randn(2, 3, 4, 5).astype(np.float32)},
            "s": {"data": np.random.randn(3).astype(np.float32)},
            "bias": {"data": np.random.randn(3).astype(np.float32)},
            "mean": {"data": np.random.randn(3).astype(np.float32)},
            "var": {"data": np.random.rand(3).astype(np.float32)},
        },
        "outputs": {
            "Y": None,
        },
        "BatchNormalization_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "BatchNormalization",
            "inputs": {
                "x": "x",
                "s": "s",
                "bias": "bias",
                "mean": "mean",
                "var": "var"
            },
            "outputs": {
                "y": "Y"
            },
            "epsilon": 1e-2,
        }
    },
    "Bernoulli": {
        "graph_name": "Bernoulli_sample",
        "inputs": {
            "x": {"data": np.random.uniform(0.0, 1.0, 10).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "Bernoulli_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Bernoulli",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            },
            "seed": float(0),
        }
    },
    "BitShift": {
        "graph_name": "BitShift_sample",
        "inputs": {
            "X1": {"data": np.array([16, 4, 1]).astype(np.uint8), "type": "uint8"},
            "X2": {"data": np.array([1, 2, 3]).astype(np.uint8), "type": "uint8"}
        },
        "outputs": {
            "Y": None
        },
        "BitShift_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "BitShift",
            "inputs": {
                "x1": "X1",
                "x2": "X2"
            },
            "outputs": {
                "y": "Y"
            },
            "direction": "LEFT"
        }
    },
    "BitwiseAnd": {
        "graph_name": "BitwiseAnd_sample",
        "inputs": {
            "X1": {"data": np.random.randint(1, high = 9, size=(3, 4, 5)), "type": "uint8"},
            "X2": {"data": np.random.randint(1, high = 9, size=(3, 4, 5)), "type": "uint8"}
        },
        "outputs": {
            "Y": None
        },
        "BitwiseAnd_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "BitwiseAnd",
            "inputs": {
                "x1": "X1",
                "x2": "X2"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "BitwiseNot": {
        "graph_name": "BitwiseNot_sample",
        "inputs": {
            "X": {"data": np.random.randint(1, high = 9, size=(3, 4, 5), dtype=np.uint16), "type": "uint16"}
        },
        "outputs": {
            "Y": None
        },
        "BitwiseNot_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "BitwiseNot",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "BitwiseOr": {
        "graph_name": "BitwiseOr_sample",
        "inputs": {
            "X1": {"data": np.random.randint(1, high = 9, size=(3, 4, 5)), "type": "uint16"},
            "X2": {"data": np.random.randint(1, high = 9, size=(3, 4, 5)), "type": "uint16"}
        },
        "outputs": {
            "Y": None
        },
        "BitwiseOr_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "BitwiseOr",
            "inputs": {
                "x1": "X1",
                "x2": "X2"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "BitwiseXor": {
        "graph_name": "BitwiseXor_sample",
        "inputs": {
            "X1": {"data":np.random.randint(1, high = 9, size=(3, 4, 5)), "type": "uint16"},
            "X2": {"data":np.random.randint(1, high = 9, size=(3, 4, 5)), "type": "uint16"}
        },
        "outputs": {
            "Y": None
        },
        "BitwiseXor_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "BitwiseXor",
            "inputs": {
                "x1": "X1",
                "x2": "X2"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "BlackmanWindow": {
        "graph_name": "BlackmanWindow_sample",
        "inputs": {
            "x": {"data": np.array([10], dtype=np.int32), "type": "int32"},
        },
        "outputs": {
            "y": {"data": np.empty(shape=(1), dtype=np.float32), "type": "float32"}
        },
        "BlackmanWindow_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "BlackmanWindow",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            },
            "periodic": 0
        }
    },
    "Cast": {
        "graph_name": "Cast_sample",
        "inputs": {
            "x": {"data": np.array(["0.47892547","0.48033667","0.49968487","0.81910545","0.47031248","0.7229038","1000000","1e-7","NaN","INF","+INF","-INF","-0.0000001","0.0000001","-1000000",], dtype=np.float32), "type": "float32"},
        },
        "outputs": {
            "y": {"data":np.empty(shape=(1), dtype=np.uint8), "type": "uint8"}
        },
        "Cast_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Cast",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            },
            "to": onnx.TensorProto.UINT8
        }
    },
    "CastLike": {
        "graph_name": "CastLike_sample",
        "inputs": {
            "x": {"data": np.array(["0.47892547","0.48033667","0.49968487","0.81910545","0.47031248","0.7229038","1000000","1e-7","NaN","INF","+INF","-INF","-0.0000001","0.0000001","-1000000",], dtype=np.float32), "type": "float32"},
            "like": {"data": np.empty(shape=(1), dtype=np.uint8), "type": "uint8"}
        },
        "outputs": {
            "y": {"data": np.empty(shape=(1), dtype=np.uint8), "type": "uint8"}
        },
        "CastLike_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "CastLike",
            "inputs": {
                "x": "x",
                "like": "like"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "Ceil": {
        "graph_name": "Ceil_sample",
        "inputs": {
            "X": {"data": np.array([-1.5, 1.2]).astype(np.float32)},
        },
        "outputs": {
            "Y": None
        },
        "Ceil_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Ceil",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "Celu": {
        "graph_name": "Celu_sample",
        "inputs": {
            "X": {"data": np.array(
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
            )}
        },
        "outputs": {
            "Y": None
        },
        "Celu_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Celu",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            },
            "alpha": 2.0
        }
    },
    "CenterCropPad": {
        "graph_name": "CenterCropPad_sample",
        "inputs": {
            "X1": {"data":np.random.randn(20, 8, 3).astype(np.float32)},
            "X2": {"data":np.array([10, 9], dtype=np.int64), "type": "int64"}
        },
        "outputs": {
            "Y": None
        },
        "CenterCropPad_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "CenterCropPad",
            "inputs": {
                "x1": "X1",
                "x2": "X2"
            },
            "outputs": {
                "y": "Y"
            },
            "axes": [-3, -2]
        }
    },
    "Clip": {
        "graph_name": "Clip_sample",
        "inputs": {
            "X1": {"data": np.array([-2, 0, 2]).astype(np.float32)},
            "X2": {"data": np.array([-1]).astype(np.float32)},
            "X3": {"data": np.array([1]).astype(np.float32)}
        },
        "outputs": {
            "Y": None
        },
        "Clip_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Clip",
            "inputs": {
                "x1": "X1",
                "x2": "X2",
                "x3": "X3"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "Col2Im": {
        "graph_name": "Col2Im_sample",
        "inputs": {
            "X1": {"data": np.array(
                    [
                        [
                            [1.0, 6.0, 11.0, 16.0, 21.0],  # (1, 5, 5)
                            [2.0, 7.0, 12.0, 17.0, 22.0],
                            [3.0, 8.0, 13.0, 18.0, 23.0],
                            [4.0, 9.0, 14.0, 19.0, 24.0],
                            [5.0, 0.0, 15.0, 20.0, 25.0],
                        ]
                    ]).astype(np.float32)},
            "X2": {"data": np.array([5, 5]).astype(np.int64), "type": "int64"},
            "X3": {"data": np.array([1, 5]).astype(np.int64), "type": "int64"}
        },
        "outputs": {
            "Y": None
        },
        "Col2Im_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Col2Im",
            "inputs": {
                "x1": "X1",
                "x2": "X2",
                "x3": "X3"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "Compress": {
        "graph_name": "Compress_sample",
        "inputs": {
            "X1": {"data": np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)},
            "X2": {"data": np.array([0, 1, 1]).astype(bool), "type": "bool"}
        },
        "outputs": {
            "Y": None
        },
        "Compress_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Compress",
            "inputs": {
                "x1": "X1",
                "x2": "X2"
            },
            "outputs": {
                "y": "Y"
            },
            "axis": 0
        }
    },
    "Concat": {
        "graph_name": "Concat_sample",
        "inputs": {
            "X1": {"data": np.array([[1, 2], [3, 4]]).astype(np.float32)},
            "X2": {"data": np.array([[5, 6], [7, 8]]).astype(np.float32)},
        },
        "outputs": {
            "Y": None
        },
        "Concat_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Concat",
            "inputs": {
                "x1": "X1",
                "x2": "X2"
            },
            "outputs": {
                "y": "Y"
            },
            "axis": 1
        }
    },
    "ConcatFromSequence": { },
    "Constant": {
        "graph_name": "Constant_sample",
        "inputs": {
            
        },
        "outputs": {
            "Y": {"data": np.empty(shape=(5, 5), dtype=np.float32)}
        },
        "Constant_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Constant",
            "inputs": {
                
            },
            "outputs": {
                "y": "Y"
            },
            "values": np.random.randn(5, 5).astype(np.float32)
        }
    },
    "ConstantOfShape": {
        "graph_name": "ConstantOfShape_sample",
        "inputs": {
            "X": {"data": np.array([10, 6]).astype(np.int64), "type": "int64"}
        },
        "outputs": {
            "Y": {"data": np.empty(shape=(1), dtype=np.int32), "type": "int32"}
        },
        "ConstantOfShape_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "ConstantOfShape",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            },
            "values": onnx.helper.make_tensor("value", onnx.TensorProto.INT32, [1], [0])
        }
    },
    "Conv": {
        "graph_name": "Conv_sample",
        "inputs": {
            "X1": {"data": np.array(
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
                    ).astype(np.float32)},
            "X2": {"data": np.array(
                        [
                            [
                                [
                                    [1.0, 1.0, 1.0],  # (1, 1, 3, 3) tensor for convolution weights
                                    [1.0, 1.0, 1.0],
                                    [1.0, 1.0, 1.0],
                                ]
                            ]
                        ]
                    ).astype(np.float32)}
        },
        "outputs": {
            "Y": None
        },
        "Conv_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Conv",
            "inputs": {
                "x1": "X1",
                "x2": "X2"
            },
            "outputs": {
                "y": "Y"
            },
            "kernel_shape": [3, 3],
            "pads": [1, 1, 1, 1]
        }
    },
    "ConvInteger": {
        "graph_name": "ConvInteger_sample",
        "inputs": {
            "x": {"data": np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.uint8).reshape((1, 1, 3, 3)), "type": "uint8"},
            "w": {"data": np.array([1, 1, 1, 1]).astype(np.uint8).reshape((1, 1, 2, 2)), "type": "uint8"},
            "x_zero_point": {"data": np.array([1]).astype(np.uint8), "type": "uint8"}
        },
        "outputs": {
            "Y": {"data": np.empty(shape=(1), dtype=np.int32), "type": "int32"}
        },
        "ConvInteger_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "ConvInteger",
            "inputs": {
                "x": "x",
                "w": "w",
                "x_zero_point": "x_zero_point"
            },
            "outputs": {
                "y": "Y"
            },
            "pads": [1, 1, 1, 1]
        }
    },
    "ConvTranspose": {
        "graph_name": "ConvTranspose_sample",
        "inputs": {
            "x": {"data": np.array([[[0.0, 1.0, 2.0]]]).astype(np.float32)},
            "w": {"data": np.array([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]).astype(np.float32)},
        },
        "outputs": {
            "y": {"data": np.empty(shape=(1), dtype=np.float32)}
        },
        "ConvTranspose_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "ConvTranspose",
            "inputs": {
                "x": "x",
                "w": "w"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "Cos": {
        "graph_name": "Cos_sample",
        "inputs": {
            "X": {"data": np.array([-1, 0, 1]).astype(np.float32)},
        },
        "outputs": {
            "Y": None
        },
        "Cos_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Cos",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "Cosh": {
        "graph_name": "Cosh_sample",
        "inputs": {
            "X": {"data": np.array([-1, 0, 1]).astype(np.float32)},
        },
        "outputs": {
            "Y": None
        },
        "Cosh_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Cosh",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "CumSum": {
        "graph_name": "CumSum_sample",
        "inputs": {
            "X1": {"data": np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float32)},
            "X2": {"data": np.array([0]).astype(np.int32), "type": "int32"}
        },
        "outputs": {
            "Y": None
        },
        "CumSum_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "CumSum",
            "inputs": {
                "x1": "X1",
                "x2": "X2"
            },
            "outputs": {
                "y": "Y"
            },
            "reverse": 1,
            "exclusive": 1
        }
    },
    "DFT": {
        "graph_name": "DFT_sample",
        "inputs": {
            "x": {"data": np.arange(0, 100).reshape(10, 10).astype(np.float32).reshape(1, 10, 10, 1)},
            "": "",
            "axis": {"data": np.array(1, dtype=np.int64), "type": "int64"}
        },
        "outputs": {
            "y": None
        },
        "DFT_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "DFT",
            "inputs": {
                "x": "x",
                "axis": "axis"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "DeformConv": { },
    "DepthToSpace": {
        "graph_name": "DepthToSpace_sample",
        "inputs": {
            "x": {"data": np.array([[
                    [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                    [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]],
                    [[18.0, 19.0, 20.0], [21.0, 22.0, 23.0]],
                    [[27.0, 28.0, 29.0], [30.0, 31.0, 32.0]],
                    [[36.0, 37.0, 38.0], [39.0, 40.0, 41.0]],
                    [[45.0, 46.0, 47.0], [48.0, 49.0, 50.0]],
                    [[54.0, 55.0, 56.0], [57.0, 58.0, 59.0]],
                    [[63.0, 64.0, 65.0], [66.0, 67.0, 68.0]],]]).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "DepthToSpace_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "DepthToSpace",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            },
            "blocksize": 2,
            "mode": "DCR"
        }
    },
    "DequantizeLinear": {
        "graph_name": "DequantizeLinear_sample",
        "inputs": {
            "x": {"data": np.array([0, 3, 128, 255]).astype(np.uint8), "type": "uint8"},
            "x_scale": {"data": np.array([2]).astype(np.float32)},
            "x_zero_point": {"data": np.array([128]).astype(np.uint8), "type": "uint8"}
        },
        "outputs": {
            "Y": {"data": np.empty(shape=(1), dtype=np.float32)}
        },
        "DequantizeLinear_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "DequantizeLinear",
            "inputs": {
                "x": "x",
                "x_scale": "x_scale",
                "x_zero_point": "x_zero_point"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "Det": {
        "graph_name": "Det_sample",
        "inputs": {
            "X": {"data": np.array([[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]).astype(np.float32)},
        },
        "outputs": {
            "Y": None
        },
        "Det_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Det",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "Div": {
        "graph_name": "Div_sample",
        "inputs": {
            "X1": {"data": np.array([3, 4]).astype(np.float32)},
            "X2": {"data": np.array([1, 2]).astype(np.float32)}
        },
        "outputs": {
            "Y": None
        },
        "Div_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Div",
            "inputs": {
                "x1": "X1",
                "x2": "X2"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "Dropout": {
        "graph_name": "Dropout_sample",
        "inputs": {
            "x": {"data": np.random.randn(3, 4, 5).astype(np.float32)},
            "r": {"data": np.array(0.75).astype(np.float32)},
            "t": {"data": np.array(True).astype(np.bool_), "type": "bool_"}
        },
        "outputs": {
            "y": None
        },
        "Dropout_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Dropout",
            "inputs": {
                "x": "x",
                "r": "r",
                "t": "t"
            },
            "outputs": {
                "y": "y"
            },
            "seed": 0
        }
    },
    "DynamicQuantizeLinear": {
        "graph_name": "DynamicQuantizeLinear_sample",
        "inputs": {
            "x": {"data": np.array([0, 2, -3, -2.5, 1.34, 0.5]).astype(np.float32)}
        },
        "outputs": {
            "y": {"data": np.empty(shape=(1), dtype=np.uint8), "type": "uint8"},
            "y_scale": None,
            "y_zero_point": {"data": np.empty(shape=(1), dtype=np.uint8), "type": "uint8"},
        },
        "DynamicQuantizeLinear_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "DynamicQuantizeLinear",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y",
                "y_scale": "y_scale",
                "y_zero_point": "y_zero_point"
            }
        }
    },
    "Einsum": {
        "graph_name": "Einsum_sample",
        "inputs": {
            "X": {"data": np.random.randn(5).astype(np.float32)},
            "Y": {"data": np.random.randn(5).astype(np.float32)}
        },
        "outputs": {
            "Z": None
        },
        "Einsum_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Einsum",
            "inputs": {
                "x": "X",
                "y": "Y"
            },
            "outputs": {
                "z": "Z"
            },
            "equation": "i,i"
        }
    },
    "Elu": {
        "graph_name": "Elu_sample",
        "inputs": {
            "X": {"data": np.random.randn(5).astype(np.float32)}
        },
        "outputs": {
            "Y": None
        },
        "Elu_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Elu",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            },
            "alpha": 2.0
        }
    },
    "Equal": {
        "graph_name": "Equal_sample",
        "inputs": {
            "X": {"data": (np.random.randn(3, 4, 5) * 10).astype(np.int32), "type": "int32"},
            "Y": {"data": (np.random.randn(3, 4, 5) * 10).astype(np.int32), "type": "int32"}
        },
        "outputs": {
            "Z": {"data": np.empty(shape=(3, 4, 5), dtype=bool), "type": "bool"}
        },
        "Equal_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Equal",
            "inputs": {
                "x": "X",
                "y": "Y"
            },
            "outputs": {
                "z": "Z"
            }
        }
    },
    "Erf": {
        "graph_name": "Erf_sample",
        "inputs": {
            "X": {"data":np.random.randn(1, 3, 32, 32).astype(np.float32)}
        },
        "outputs": {
            "Y": None
        },
        "Erf_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Erf",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "Exp": {
        "graph_name": "Exp_sample",
        "inputs": {
            "X": {"data": np.array([-1, 0, 1]).astype(np.float32)}
        },
        "outputs": {
            "Y": None
        },
        "Exp_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Exp",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "Expand": {
        "graph_name": "Expand_sample",
        "inputs": {
            "X": {"data": np.reshape(np.arange(1, np.prod([3, 1]) + 1, dtype=np.float32), [3, 1])},
            "Y": {"data": np.array([2, 1, 6], dtype=np.int64), "type": "int64"}
        },
        "outputs": {
            "Z": None
        },
        "Expand_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Expand",
            "inputs": {
                "x": "X",
                "y": "Y"
            },
            "outputs": {
                "z": "Z"
            }
        }
    },
    "EyeLike": {
        "graph_name": "EyeLike_sample",
        "inputs": {
            "X": {"data": np.random.randint(0, 100, size=(4, 5), dtype=np.int32), "type": "int32"}
        },
        "outputs": {
            "Y": None
        },
        "EyeLike_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "EyeLike",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            },
            "k": 1,
            "dtype": onnx.TensorProto.INT32,
        }
    },
    "Flatten": {
        "graph_name": "Flatten_sample",
        "inputs": {
            "X": {"data": np.random.random_sample((2, 3, 4)).astype(np.float32)}
        },
        "outputs": {
            "Y": None
        },
        "Flatten_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Flatten",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            },
            "axis": 2
        }
    },
    "Floor": {
        "graph_name": "Floor_sample",
        "inputs": {
            "X": {"data":np.array([-1.5, 1.2, 2]).astype(np.float32)}
        },
        "outputs": {
            "Y": None
        },
        "Floor_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Floor",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "GRU": { },
    "Gather": {
        "graph_name": "Gather_sample",
        "inputs": {
            "data": {"data": np.random.randn(3, 3, 4).astype(np.float32)},
            "indices": {"data": np.array([1, 2]), "type": "int32"}
        },
        "outputs": {
            "y": None
        },
        "Gather_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Gather",
            "inputs": {
                "data": "data",
                "indices": "indices"
            },
            "outputs": {
                "y": "y"
            },
            "axis": 0
        }
    },
    "GatherElements": {
        "graph_name": "GatherElements_sample",
        "inputs": {
            "data": {"data":np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)},
            "indices": {"data":np.array([[1, 2, 0], [2, 0, 0]], dtype=np.int32), "type": "int32"}
        },
        "outputs": {
            "y": None
        },
        "GatherElements_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "GatherElements",
            "inputs": {
                "data": "data",
                "indices": "indices"
            },
            "outputs": {
                "y": "y"
            },
            "axis": 0
        }
    },
    "GatherND": {
        "graph_name": "GatherND_sample",
        "inputs": {
            "data": {"data": np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int32), "type": "int32"},
            "indices": {"data": np.array([[1], [0]], dtype=np.int64), "type": "int64"}
        },
        "outputs": {
            "y": None
        },
        "GatherND_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "GatherND",
            "inputs": {
                "data": "data",
                "indices": "indices"
            },
            "outputs": {
                "y": "y"
            },
            "batch_dims": 1
        }
    },
    "Gelu": {
        "graph_name": "Gelu_sample",
        "inputs": {
            "X": {"data": np.array([-1, 0, 1]).astype(np.float32)}
        },
        "outputs": {
            "Y": None
        },
        "Gelu_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Gelu",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            }
        }

    },
    "Gemm": {
        "graph_name": "Gemm_sample",
        "inputs": {
            "a": {"data": np.random.ranf([3, 5]).astype(np.float32)},
            "b": {"data": np.random.ranf([5, 4]).astype(np.float32)},
            "c": {"data": np.zeros([1, 4]).astype(np.float32)}
        },
        "outputs": {
            "Y": None
        },
        "Gemm_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Gemm",
            "inputs": {
                "a": "a",
                "b": "b",
                "c": "c"
            },
            "outputs": {
                "y": "Y"
            },
            "alpha": 0.5
        }
    },
    "GlobalAveragePool": {
        "graph_name": "GlobalAveragePool_sample",
        "inputs": {
            "X": {"data": np.random.randn(1, 3, 2, 2).astype(np.float32)}
        },
        "outputs": {
            "Y": None
        },
        "GlobalAveragePool_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "GlobalAveragePool",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "GlobalLpPool": { },
    "GlobalMaxPool": {
        "graph_name": "GlobalMaxPool_sample",
        "inputs": {
            "X": {"data": np.random.randn(1, 3, 5, 5).astype(np.float32)}
        },
        "outputs": {
            "Y": None
        },
        "GlobalMaxPool_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "GlobalMaxPool",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "Greater": {
        "graph_name": "Greater_sample",
        "inputs": {
            "X": {"data": np.random.randn(3, 4, 5).astype(np.float32)},
            "Y": {"data": np.random.randn(3, 4, 5).astype(np.float32)}
        },
        "outputs": {
            "greater": {"data": np.empty(shape=(3, 4, 5), dtype=bool), "type": "bool"}
        },
        "Greater_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Greater",
            "inputs": {
                "x": "X",
                "y": "Y"
            },
            "outputs": {
                "greater": "greater"
            }
        }
    },
    "GreaterOrEqual": {
        "graph_name": "GreaterOrEqual_sample",
        "inputs": {
            "X": {"data": np.random.randn(3, 4, 5).astype(np.float32)},
            "Y": {"data": np.random.randn(3, 4, 5).astype(np.float32)}
        },
        "outputs": {
            "greater": {"data": np.empty(shape=(3, 4, 5), dtype=bool), "type": "bool"}
        },
        "GreaterOrEqual_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "GreaterOrEqual",
            "inputs": {
                "x": "X",
                "y": "Y"
            },
            "outputs": {
                "greater": "greater"
            }
        }
    },
    "GridSample": {
        "graph_name": "GridSample_sample",
        "inputs": {
            "X": {"data": np.array([[[
                    [0.0, 1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [8.0, 9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0, 15.0],
                ]]],dtype=np.float32,)},
            "Grid": {"data": np.array([[
                    [[-1.0000, -1.0000], [-0.6000, -1.0000], [-0.2000, -1.0000], [0.2000, -1.0000], [0.6000, -1.0000], [1.0000, -1.0000],],
                    [[-1.0000, -0.6000], [-0.6000, -0.6000], [-0.2000, -0.6000], [0.2000, -0.6000], [0.6000, -0.6000], [1.0000, -0.6000],],
                    [[-1.0000, -0.2000], [-0.6000, -0.2000], [-0.2000, -0.2000], [0.2000, -0.2000], [0.6000, -0.2000], [1.0000, -0.2000],],
                    [[-1.0000, 0.2000], [-0.6000, 0.2000], [-0.2000, 0.2000], [0.2000, 0.2000], [0.6000, 0.2000], [1.0000, 0.2000],],
                    [[-1.0000, 0.6000], [-0.6000, 0.6000], [-0.2000, 0.6000], [0.2000, 0.6000], [0.6000, 0.6000], [1.0000, 0.6000],],
                    [[-1.0000, 1.0000], [-0.6000, 1.0000], [-0.2000, 1.0000], [0.2000, 1.0000], [0.6000, 1.0000], [1.0000, 1.0000],],]],dtype=np.float32,)}
        },
        "outputs": {
            "Y": None
        },
        "GridSample_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "GridSample",
            "inputs": {
                "x": "X",
                "Grid": "Grid"
            },
            "outputs": {
                "y": "Y"
            },
            "mode": "linear",
            "padding_mode": "zeros",
            "align_corners": 0,
        }
    },
    "GroupNormalization": {
        "graph_name": "GroupNormalization_sample",
        "inputs": {
            "x": {"data": np.random.randn(3, 4, 2, 2).astype(np.float32)},
            "scale": {"data": np.random.randn(4).astype(np.float32)},
            "bias": {"data": np.random.randn(4).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "GroupNormalization_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "GroupNormalization",
            "inputs": {
                "x": "x",
                "scale": "scale",
                "bias": "bias"
            },
            "outputs": {
                "y": "y"
            },
            "epsilon": 1e-2,
            "num_groups": 2,
        }
    },
    "HammingWindow": {
        "graph_name": "HammingWindow_sample",
        "inputs": {
            "size": {"data": np.array([10]).astype(np.int32), "type": "int32"}
        },
        "outputs": {
            "Y": {"data": np.empty(shape=(6), dtype=np.float32)}
        },
        "HammingWindow_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "HammingWindow",
            "inputs": {
                "size": "size"
            },
            "outputs": {
                "y": "Y"
            },
            "periodic": 0
        }
    },
    "HannWindow": {
        "graph_name": "HannWindow_sample",
        "inputs": {
            "size": {"data": np.array([10]).astype(np.int32), "type": "int32"}
        },
        "outputs": {
            "Y": {"data": np.empty(shape=(6), dtype=np.float32)}
        },
        "HannWindow_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "HannWindow",
            "inputs": {
                "size": "size"
            },
            "outputs": {
                "y": "Y"
            },
            "periodic": 0
        }
    },
    "HardSigmoid": {
        "graph_name": "HardSigmoid_sample",
        "inputs": {
            "X": {"data": np.array([-1, 0, 1]).astype(np.float32)}
        },
        "outputs": {
            "Y": None
        },
        "HardSigmoid_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "HardSigmoid",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            },
            "alpha": 0.5, 
            "beta": 0.6
        }
    },
    "HardSwish": {
        "graph_name": "HardSwish_sample",
        "inputs": {
            "X": {"data": np.random.randn(3, 4, 5).astype(np.float32)}
        },
        "outputs": {
            "Y": None
        },
        "HardSwish_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "HardSwish",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "Hardmax": {
        "graph_name": "Hardmax_sample",
        "inputs": {
            "X": {"data": np.array([[3, 0, 1, 2], [2, 5, 1, 0], [0, 1, 3, 2], [0, 1, 2, 3]]).astype(np.float32)}
        },
        "outputs": {
            "Y": None
        },
        "Hardmax_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Hardmax",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "Identity": {
        "graph_name": "Identity_sample",
        "inputs": {
            "X": {"data": np.array([[[
                [1, 2],
                [3, 4],]]],dtype=np.float32,)}
        },
        "outputs": {
            "Y": None
        },
        "Identity_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Identity",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "If": {
        "graph_name": "If_sample",
        "inputs": {
            "cond": {"data": np.array(1).astype(bool), "type": "bool"}
        },
        "outputs": {
            "y": {"data": np.empty(shape=(1), dtype=np.float32)}
        },
        "If_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "If",
            "inputs": {
                "cond": "cond"
            },
            "outputs": {
                "y": "y"
            },
            "then_branch": then_body,
            "else_branch": else_body,
        }
    },
    "ImageDecoder": { },
    "InstanceNormalization": {
        "graph_name": "InstanceNormalization_sample",
        "inputs": {
            "x": {"data": np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(np.float32)},
            "s": {"data": np.array([1.0, 1.5]).astype(np.float32)},
            "bias": {"data": np.array([0, 1]).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "InstanceNormalization_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "InstanceNormalization",
            "inputs": {
                "x": "x",
                "s": "s",
                "bias": "bias"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "IsInf": {
        "graph_name": "IsInf_sample",
        "inputs": {
            "x": {"data": np.array([-1.7, np.nan, np.inf, -3.6, -np.inf, np.inf], dtype=np.float32)}
        },
        "outputs": {
            "y": {"data": np.empty(shape=(6), dtype=bool), "type": "bool"}
        },
        "IsInf_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "IsInf",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            },
            "detect_negative": 0
        }
    },
    "IsNaN": {
        "graph_name": "IsNaN_sample",
        "inputs": {
            "x": {"data":np.array([-1.2, np.nan, np.inf, 2.8, -np.inf, np.inf], dtype=np.float32)}
        },
        "outputs": {
            "y": {"data":np.empty(shape=(6), dtype=bool), "type": "bool"}
        },
        "IsNaN_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "IsNaN",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "LRN": {
        "graph_name": "LRN_sample",
        "inputs": {
            "x": {"data":np.random.randn(5, 5, 5, 5).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "LRN_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "LRN",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            },
            "alpha": 0.0002,
            "beta": 0.5,
            "bias": 2.0,
            "size": 3
        }
    },
    "LSTM": {
        "graph_name": "LSTM_sample",
        "inputs": {
            "X": {"data": np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]).astype(np.float32)},
            "W": {"data": 0.3 * np.ones((1, 4 * 7, 2)).astype(np.float32)},
            "R": {"data": 0.3 * np.ones((1, 4 * 7, 7)).astype(np.float32)}
        },
        "outputs": {
            "Y": None,
            "Y_h": None
        },
        "LSTM_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "LSTM",
            "inputs": {
                "X": "X",
                "W": "W",
                "R": "R"
            },
            "outputs": {
                "y": "Y",
                "Y_h": "Y_h"
            },
            "hidden_size": 7,
        }
    },
    "LayerNormalization": {
        "graph_name": "LayerNormalization_sample",
        "inputs": {
            "X": {"data":np.random.randn(2, 3, 4, 5).astype(np.float32)},
            "W": {"data":np.random.randn(*calculate_normalized_shape((2, 3, 4, 5), -1)).astype(np.float32)},
            "B": {"data":np.random.randn(*calculate_normalized_shape((2, 3, 4, 5), -1)).astype(np.float32)}
        },
        "outputs": {
            "Y": None,
            "Mean": None,
            "InvStdDev": None
        },
        "LayerNormalization_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "LayerNormalization",
            "inputs": {
                "X": "X",
                "W": "W",
                "B": "B"
            },
            "outputs": {
                "Y": "Y",
                "Mean": "Mean",
                "InvStdDev": "InvStdDev"
            }
        }
    },
    "LeakyRelu": {
        "graph_name": "LeakyRelu_sample",
        "inputs": {
            "x": {"data": np.array([-1, 0, 1]).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "LeakyRelu_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "LeakyRelu",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            },
            "alpha": 0.1
        }
    },
    "Less": {
        "graph_name": "Less_sample",
        "inputs": {
            "x": {"data": np.random.randn(3, 4, 5).astype(np.float32)},
            "y": {"data": np.random.randn(5).astype(np.float32)}
        },
        "outputs": {
            "less": {"data": np.empty(shape=(3, 4, 5), dtype=bool), "type": "bool"}
        },
        "Less_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Less",
            "inputs": {
                "x": "x",
                "y": "y"
            },
            "outputs": {
                "less": "less"
            }
        }
    },
    "LessOrEqual": {
        "graph_name": "LessOrEqual_sample",
        "inputs": {
            "x": {"data": np.random.randn(3, 4, 5).astype(np.float32)},
            "y": {"data": np.random.randn(5).astype(np.float32)}
        },
        "outputs": {
            "less_equal": {"data": np.empty(shape=(3, 4, 5), dtype=bool), "type": "bool"}
        },
        "LessOrEqual_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "LessOrEqual",
            "inputs": {
                "x": "x",
                "y": "y"
            },
            "outputs": {
                "less_equal": "less_equal"
            }
        }
    },
    "Log": {
        "graph_name": "Log_sample",
        "inputs": {
            "x": {"data": np.array([1, 10]).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "Log_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Log",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "LogSoftmax": {
        "graph_name": "LogSoftmax_sample",
        "inputs": {
            "x": {"data":np.array([[-1, 0, 1]]).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "LogSoftmax_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "LogSoftmax",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "Loop": { },
    "LpNormalization": { },
    "LpPool": {
        "graph_name": "LpPool_sample",
        "inputs": {
            "x": {"data": np.random.randn(1, 3, 32, 32).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "LpPool_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "LpPool",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            },
            "kernel_shape": [2, 2],
            "auto_pad": "SAME_UPPER",
            "p": 2,
        }
    },
    "MatMul": {
        "graph_name": "MatMul_sample",
        "inputs": {
            "a": {"data": np.random.randn(3, 4).astype(np.float32)},
            "b": {"data": np.random.randn(4, 3).astype(np.float32)}
        },
        "outputs": {
            "c": None
        },
        "MatMul_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "MatMul",
            "inputs": {
                "a": "a",
                "b": "b"
            },
            "outputs": {
                "c": "c"
            }
        }
    },
    "MatMulInteger": {
        "graph_name": "MatMulInteger_sample",
        "inputs": {
            "A": {"data": np.array([
                        [11, 7, 3],
                        [10, 6, 2],
                        [9, 5, 1],
                        [8, 4, 0],], dtype=np.uint8), "type": "uint8"},
            "B": {"data": np.array([
                        [1, 4],
                        [2, 5],
                        [3, 6],], dtype=np.uint8), "type": "uint8"},
            "a_zero_point": {"data": np.array([12], dtype=np.uint8), "type": "uint8"},
            "b_zero_point": {"data": np.array([0], dtype=np.uint8), "type": "uint8"}
        },
        "outputs": {
            "Y": {"data": np.empty(shape=(1), dtype=np.int32), "type": "int32"}
        },
        "MatMulInteger_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "MatMulInteger",
            "inputs": {
                "A": "A",
                "B": "B",
                "a_zero_point": "a_zero_point",
                "b_zero_point": "b_zero_point"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "Max": {
        "graph_name": "Max_sample",
        "inputs": {
            "data_0": {"data": np.array([3, 2, 1]).astype(np.float32)},
            "data_1": {"data": np.array([1, 4, 4]).astype(np.float32)},
            "data_2": {"data": np.array([2, 5, 3]).astype(np.float32)}
        },
        "outputs": {
            "result": None
        },
        "Max_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Max",
            "inputs": {
                "data_0": "data_0",
                "data_1": "data_1",
                "data_2": "data_2"
            },
            "outputs": {
                "result": "result"
            }
        }
    },
    "MaxPool": {
        "graph_name": "MaxPool_sample",
        "inputs": {
            "x": {"data":np.array([[[
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]]]).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "MaxPool_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "MaxPool",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            },
            "kernel_shape": [3, 3],
            "strides": [2, 2],
            "ceil_mode": True,
        }
    },
    "MaxRoiPool": { },
    "MaxUnpool": {
        "graph_name": "MaxUnpool_sample",
        "inputs": {
            "xT": {"data": np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)},
            "xI": {"data": np.array([[[[5, 7], [13, 15]]]], dtype=np.int64), "type": "int64"}
        },
        "outputs": {
            "y": None
        },
        "MaxUnpool_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "MaxUnpool",
            "inputs": {
                "xT": "xT",
                "xI": "xI"
            },
            "outputs": {
                "y": "y"
            },
            "kernel_shape": [2, 2],
            "strides": [2, 2]
        }
    },
    "Mean": {
        "graph_name": "Mean_sample",
        "inputs": {
            "data_0": {"data": np.array([3, 0, 2]).astype(np.float32)},
            "data_1": {"data": np.array([1, 3, 4]).astype(np.float32)},
            "data_2": {"data": np.array([2, 6, 6]).astype(np.float32)}
        },
        "outputs": {
            "result": None
        },
        "Mean_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Mean",
            "inputs": {
                "data_0": "data_0",
                "data_1": "data_1",
                "data_2": "data_2"
            },
            "outputs": {
                "result": "result"
            }
        }
    },
    "MeanVarianceNormalization": {
        "graph_name": "MeanVarianceNormalization_sample",
        "inputs": {
            "X": {"data": np.array(
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
                ],dtype=np.float32,)},
        },
        "outputs": {
            "Y": None
        },
        "MeanVarianceNormalization_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "MeanVarianceNormalization",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "MelWeightMatrix": {
        "graph_name": "MelWeightMatrix_sample",
        "inputs": {
            "num_mel_bins": {"data": np.array([8]).astype(np.int32), "type": "int32"},
            "dft_length": {"data": np.array([16]).astype(np.int32), "type": "int32"},
            "sample_rate": {"data": np.array([8192]).astype(np.int32), "type": "int32"},
            "lower_edge_hertz": {"data": np.array([0]).astype(np.float32)},
            "upper_edge_hertz": {"data": np.array([8192 / 2]).astype(np.float32)},
        },
        "outputs": {
            "output": {"data": np.empty(shape=(3, 4, 5), dtype=np.float32)}
        },
        "MelWeightMatrix_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "MelWeightMatrix",
            "inputs": {
                "num_mel_bins": "num_mel_bins",
                "dft_length": "dft_length",
                "sample_rate": "sample_rate",
                "lower_edge_hertz": "lower_edge_hertz",
                "upper_edge_hertz": "upper_edge_hertz"
            },
            "outputs": {
                "output": "output"
            }
        }
    },
    "Min": {
        "graph_name": "Min_sample",
        "inputs": {
            "data_0": {"data": np.array([3, 2, 1]).astype(np.float32)},
            "data_1": {"data": np.array([1, 4, 4]).astype(np.float32)},
            "data_2": {"data": np.array([2, 5, 3]).astype(np.float32)}
        },
        "outputs": {
            "result": None
        },
        "Min_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Min",
            "inputs": {
                "data_0": "data_0",
                "data_1": "data_1",
                "data_2": "data_2"
            },
            "outputs": {
                "result": "result"
            }
        }
    },
    "Mish": {
        "graph_name": "Mish_sample",
        "inputs": {
            "X": {"data": np.linspace(-10, 10, 10000, dtype=np.float32)}
        },
        "outputs": {
            "Y": None
        },
        "Mish_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Mish",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            }
        }
    },
    "Mod": {
        "graph_name": "Mod_sample",
        "inputs": {
            "x": {"data": np.array([-4.3, 7.2, 5.0, 4.3, -7.2, 8.0]).astype(np.float32)},
            "y": {"data": np.array([2.1, -3.4, 8.0, -2.1, 3.4, 5.0]).astype(np.float32)}
        },
        "outputs": {
            "Y": None
        },
        "Mod_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Mod",
            "inputs": {
                "x": "x",
                "y": "y"
            },
            "outputs": {
                "Y": "Y"
            },
            "fmod": 1
        }
    },
    "Mul": {
        "graph_name": "Mul_sample",
        "inputs": {
            "x": {"data": np.array([1, 2, 3]).astype(np.float32)},
            "y": {"data": np.array([4, 5, 6]).astype(np.float32)}
        },
        "outputs": {
            "z": None
        },
        "Mul_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Mul",
            "inputs": {
                "x": "x",
                "y": "y"
            },
            "outputs": {
                "z": "z"
            }
        }
    },
    "Multinomial": { },
    "Neg": {
        "graph_name": "Neg_sample",
        "inputs": {
            "x": {"data":np.array([-4, 2]).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "Neg_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Neg",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "NegativeLogLikelihoodLoss": {
        "graph_name": "NegativeLogLikelihoodLoss_sample",
        "inputs": {
            "input": {"data":np.random.rand(3, 5, 6, 6).astype(np.float32)},
            "target": {"data":np.random.randint(0, high=5, size=(3, 6, 6)).astype(np.int64), "type": "int64"},
            "weight": {"data":np.random.rand(5).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "NegativeLogLikelihoodLoss_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "NegativeLogLikelihoodLoss",
            "inputs": {
                "input": "input",
                "target": "target",
                "weight": "weight"
            },
            "outputs": {
                "y": "y"
            },
            "reduction": "sum"
        }
    },
    "NonMaxSuppression": {
        "graph_name": "LpPool_sample",
        "inputs": {
            "boxes": {"data": np.array([[
                        [0.5, 0.5, 1.0, 1.0],
                        [0.5, 0.6, 1.0, 1.0],
                        [0.5, 0.4, 1.0, 1.0],
                        [0.5, 10.5, 1.0, 1.0],
                        [0.5, 10.6, 1.0, 1.0],
                        [0.5, 100.5, 1.0, 1.0],]]).astype(np.float32)},
            "scores": {"data": np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)},
            "max_output_boxes_per_class": {"data": np.array([3]).astype(np.int64), "type": "int64"},
            "iou_threshold": {"data": np.array([0.5]).astype(np.float32)},
            "score_threshold": {"data": np.array([0.0]).astype(np.float32)},
        },
        "outputs": {
            "y": {"data": np.empty(shape=(1), dtype=np.int64), "type": "int64"}
        },
        "NonMaxSuppression_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "NonMaxSuppression",
            "inputs": {
                "boxes": "boxes",
                "scores": "scores",
                "max_output_boxes_per_class": "max_output_boxes_per_class",
                "iou_threshold": "iou_threshold",
                "score_threshold": "score_threshold"
            },
            "outputs": {
                "y": "y"
            },
            "center_point_box": 1,
        }
    },
    "NonZero": {
        "graph_name": "NonZero_sample",
        "inputs": {
            "condition": {"data": np.array([[1, 0], [1, 1]], dtype=bool), "type": "bool"}
        },
        "outputs": {
            "result": {"data": np.empty(shape=(1), dtype=np.int64), "type": "int64"}
        },
        "NonZero_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "NonZero",
            "inputs": {
                "condition": "condition"
            },
            "outputs": {
                "result": "result"
            }
        }
    },
    "Not": {
        "graph_name": "Not_sample",
        "inputs": {
            "x": {"data": (np.random.randn(3, 4, 5) > 0).astype(bool), "type": "bool"}
        },
        "outputs": {
            "not": None
        },
        "Not_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Not",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "not": "not"
            }
        }
    },
    "OneHot": {
        "graph_name": "OneHot_sample",
        "inputs": {
            "indices": {"data": np.array([[1, 9], [2, 4]], dtype=np.float32)},
            "depth": {"data": np.array([10], dtype=np.float32)},
            "values": {"data": np.array([1, 3], dtype=np.float32)}
        },
        "outputs": {
            "y": None
        },
        "OneHot_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "OneHot",
            "inputs": {
                "indices": "indices",
                "depth": "depth",
                "values": "values"
            },
            "outputs": {
                "y": "y"
            },
            "axis": 1,
        }
    },
    "Optional": { },
    "OptionalGetElement": { },
    "OptionalHasElement": { },
    "Or": {
        "graph_name": "Or_sample",
        "inputs": {
            "x": {"data": (np.random.randn(3, 4, 5) > 0).astype(bool), "type": "bool"},
            "y": {"data": (np.random.randn(3, 4, 5) > 0).astype(bool), "type": "bool"}
        },
        "outputs": {
            "z": None
        },
        "Or_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Or",
            "inputs": {
                "x": "x",
                "y": "y"
            },
            "outputs": {
                "z": "z"
            }
        }
    },
    "PRelu": {
        "graph_name": "PRelu_sample",
        "inputs": {
            "x": {"data": np.random.randn(3, 4, 5).astype(np.float32)},
            "slope": {"data": np.random.randn(3, 4, 5).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "PRelu_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "PRelu",
            "inputs": {
                "x": "x",
                "slope": "slope"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "Pad": {
        "graph_name": "Pad_sample",
        "inputs": {
            "x": {"data": np.random.randn(1, 3, 4, 5).astype(np.float32)},
            "pads": {"data": np.array([0, 3, 0, 4]).astype(np.int64), "type": "int64"},
            "value": {"data": np.array(1.2, dtype=np.float32)},
            "axes": {"data": np.array([1, 3], dtype=np.int64), "type": "int64"}
        },
        "outputs": {
            "y": None
        },
        "Pad_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Pad",
            "inputs": {
                "x": "x",
                "pads": "pads",
                "value": "value",
                "axes": "axes"
            },
            "outputs": {
                "y": "y"
            },
            "mode": "constant"
        }
    },
    "Pow": {
        "graph_name": "Pow_sample",
        "inputs": {
            "x": {"data": np.array([1, 2, 3]).astype(np.float32)},
            "y": {"data": np.array([4, 5, 6]).astype(np.float32)}
        },
        "outputs": {
            "z": None
        },
        "Pow_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Pow",
            "inputs": {
                "x": "x",
                "y": "y"
            },
            "outputs": {
                "z": "z"
            }
        }
    },
    "QLinearConv": {
        "graph_name": "QLinearConv_sample",
        "inputs": {
            "x": {"data": np.array([
                    [255, 174, 162, 25, 203, 168, 58],
                    [15, 59, 237, 95, 129, 0, 64],
                    [56, 242, 153, 221, 168, 12, 166],
                    [232, 178, 186, 195, 237, 162, 237],
                    [188, 39, 124, 77, 80, 102, 43],
                    [127, 230, 21, 83, 41, 40, 134],
                    [255, 154, 92, 141, 42, 148, 247],],dtype=np.uint8,).reshape((1, 1, 7, 7)), "type": "uint8"},
            "x_scale": {"data": np.array(0.00369204697, dtype=np.float32)},
            "x_zero_point": {"data": np.array(132, dtype=np.uint8), "type": "uint8"},
            "w": {"data": np.array([0], dtype=np.uint8).reshape((1, 1, 1, 1)), "type": "uint8"},
            "w_scale": {"data": np.array([0.00172794575], dtype=np.float32)},
            "w_zero_point": {"data": np.array([255], dtype=np.uint8), "type": "uint8"},
            "y_scale": {"data": np.array(0.00162681262, dtype=np.float32)},
            "y_zero_point": {"data": np.array(123, dtype=np.uint8), "type": "uint8"},
        },
        "outputs": {
            "y": None
        },
        "QLinearConv_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "QLinearConv",
            "inputs": {
                "x": "x",
                "x_scale": "x_scale",
                "x_zero_point": "x_zero_point",
                "w": "w",
                "w_scale":"w_scale",
                "w_zero_point": "w_zero_point",
                "y_scale": "y_scale",
                "y_zero_point": "y_zero_point"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "QLinearMatMul": {
        "graph_name": "QLinearMatMul_sample",
        "inputs": {
            "a": {"data": np.array([[208, 236, 0, 238], [3, 214, 255, 29]]).astype(np.uint8), "type": "uint8"},
            "a_scale": {"data": np.array([0.0066], dtype=np.float32)},
            "a_zero_point": {"data": np.array([113], dtype=np.uint8), "type": "uint8"},
            "b": {"data": np.array([[152, 51, 244], [60, 26, 255], [0, 127, 246], [127, 254, 247]]).astype(np.uint8), "type": "uint8"},
            "b_scale": {"data": np.array([0.00705], dtype=np.float32)},
            "b_zero_point": {"data": np.array([114], dtype=np.uint8), "type": "uint8"},
            "y_scale": {"data": np.array([0.0107], dtype=np.float32)},
            "y_zero_point": {"data": np.array([118], dtype=np.uint8), "type": "uint8"}
        },
        "outputs": {
            "y": None
        },
        "QLinearMatMul_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "QLinearMatMul",
            "inputs": {
                "a": "a",
                "a_scale": "a_scale",
                "a_zero_point": "a_zero_point",
                "b": "b",
                "b_scale": "b_scale",
                "b_zero_point": "b_zero_point",
                "y_scale": "y_scale",
                "y_zero_point": "y_zero_point"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "QuantizeLinear": {
        "graph_name": "QuantizeLinear_sample",
        "inputs": {
            "x": {"data":np.array([
                        [6.0, -8, -10, 5.0],
                        [1.0, 8.0, 4.0, 5.0],
                        [0.0, 20.0, 10.0, 4.0],],dtype=np.float32)},
            "y_scale": {"data":np.array([
                        [1.5, 2.5],
                        [3.0, 4.9],
                        [5.1, 6.9],],dtype=np.float32)}
        },
        "outputs": {
            "y": {"data":np.empty(shape=(1), dtype=np.uint8), "type": "uint8"}
        },
        "QuantizeLinear_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "QuantizeLinear",
            "inputs": {
                "x": "x",
                "y_scale": "y_scale"
            },
            "outputs": {
                "y": "y"
            },
            "axis": 1,
            "block_size": 2,
            "output_dtype": onnx.TensorProto.UINT8,
        }
    },
    "RNN": {
        "graph_name": "RNN_sample",
        "inputs": {
            "X": {"data": np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]).astype(np.float32)},
            "W": {"data": 0.5 * np.ones((1, 4, 2)).astype(np.float32)},
            "R": {"data": 0.5 * np.ones((1, 4, 4)).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "RNN_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "RNN",
            "inputs": {
                "X": "X",
                "W": "W",
                "R": "R"
            },
            "outputs": {
                "y": "y"
            },
            "hidden_size": 4
        }
    },
    "RandomNormal": { },
    "RandomNormalLike": { },
    "RandomUniform": { },
    "RandomUniformLike": { },
    "Range": {
        "graph_name": "Range_sample",
        "inputs": {
            "start": {"data": np.array([1]).astype(np.float32)},
            "limit": {"data": np.array([5]).astype(np.float32)},
            "delta": {"data": np.array([2]).astype(np.float32)}
        },
        "outputs": {
            "output": None
        },
        "Range_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Range",
            "inputs": {
                "start": "start",
                "limit": "limit",
                "delta": "delta"
            },
            "outputs": {
                "output": "output"
            }
        }
    },
    "Reciprocal": {
        "graph_name": "Reciprocal_sample",
        "inputs": {
            "x": {"data": np.array([-4, 2]).astype(np.float32)},
        },
        "outputs": {
            "y": None
        },
        "Reciprocal_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Reciprocal",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "ReduceL1": {
        "graph_name": "ReduceL1_sample",
        "inputs": {
            "data": {"data": np.reshape(np.arange(1, np.prod([3, 2, 2]) + 1, dtype=np.float32), [3, 2, 2])},
            "axes": {"data": np.array([], dtype=np.int64), "type": "int64"}
        },
        "outputs": {
            "reduced": None
        },
        "ReduceL1_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "ReduceL1",
            "inputs": {
                "data": "data",
                "axes": "axes"
            },
            "outputs": {
                "reduced": "reduced"
            },
            "keepdims": 1
        }
    },
    "ReduceL2": {
        "graph_name": "ReduceL2_sample",
        "inputs": {
            "data": {"data": np.reshape(np.arange(1, np.prod([3, 2, 2]) + 1, dtype=np.float32), [3, 2, 2])},
            "axes": {"data": np.array([], dtype=np.int64), "type": "int64"}
        },
        "outputs": {
            "reduced": None
        },
        "ReduceL2_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "ReduceL2",
            "inputs": {
                "data": "data",
                "axes": "axes"
            },
            "outputs": {
                "reduced": "reduced"
            },
            "keepdims": 1
        }
    },
    "ReduceLogSum": {
        "graph_name": "ReduceLogSum_sample",
        "inputs": {
            "data": {"data":np.array([], dtype=np.float32).reshape([2, 0, 4])},
            "axes": {"data":np.array([1], dtype=np.int64), "type": "int64"}
        },
        "outputs": {
            "reduced": None
        },
        "ReduceLogSum_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "ReduceLogSum",
            "inputs": {
                "data": "data",
                "axes": "axes"
            },
            "outputs": {
                "reduced": "reduced"
            },
            "keepdims": 1
        }
    },
    "ReduceLogSumExp": {
        "graph_name": "ReduceLogSumExp_sample",
        "inputs": {
            "data": {"data": np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)},
            "axes": {"data": np.array([], dtype=np.int64), "type": "int64"}
        },
        "outputs": {
            "reduced": None
        },
        "ReduceLogSumExp_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "ReduceLogSumExp",
            "inputs": {
                "data": "data",
                "axes": "axes"
            },
            "outputs": {
                "reduced": "reduced"
            },
            "keepdims": 1
        }
    },
    "ReduceMax": {
        "graph_name": "ReduceMax_sample",
        "inputs": {
            "data": {"data": np.array([[True, True], [True, False], [False, True], [False, False]],), "type": "bool"},
            "axes": {"data": np.array([1], dtype=np.int64), "type": "int64"}
        },
        "outputs": {
            "reduced": None
        },
        "ReduceMax_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "ReduceMax",
            "inputs": {
                "data": "data",
                "axes": "axes"
            },
            "outputs": {
                "reduced": "reduced"
            },
            "keepdims": 1
        }
    },
    "ReduceMean": {
        "graph_name": "ReduceMean_sample",
        "inputs": {
            "data": {"data": np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],dtype=np.float32,)},
            "axes": {"data": np.array([], dtype=np.int64), "type": "int64"}
        },
        "outputs": {
            "reduced": None
        },
        "ReduceMean_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "ReduceMean",
            "inputs": {
                "data": "data",
                "axes": "axes"
            },
            "outputs": {
                "reduced": "reduced"
            },
            "keepdims": 1
        }
    },
    "ReduceMin": {
        "graph_name": "ReduceMin_sample",
        "inputs": {
            "data": {"data": np.array([[True, True], [True, False], [False, True], [False, False]],), "type": "bool"},
            "axes": {"data": np.array([1], dtype=np.int64), "type": "int64"}
        },
        "outputs": {
            "reduced": None
        },
        "ReduceMin_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "ReduceMin",
            "inputs": {
                "data": "data",
                "axes": "axes"
            },
            "outputs": {
                "reduced": "reduced"
            },
            "keepdims": 1
        }
    },
    "ReduceProd": {
        "graph_name": "ReduceProd_sample",
        "inputs": {
            "data": {"data":np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)}
        },
        "outputs": {
            "reduced": None
        },
        "ReduceProd_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "ReduceProd",
            "inputs": {
                "data": "data"
            },
            "outputs": {
                "reduced": "reduced"
            },
            "keepdims": 1
        }
    },
    "ReduceSum": {
        "graph_name": "ReduceSum_sample",
        "inputs": {
            "data": {"data": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)},
            "axes": {"data": np.array([], dtype=np.int64), "type": "int64"}
        },
        "outputs": {
            "reduced": None
        },
        "ReduceSum_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "ReduceSum",
            "inputs": {
                "data": "data",
                "axes": "axes"
            },
            "outputs": {
                "reduced": "reduced"
            },
            "keepdims": 1
        }
    },
    "ReduceSumSquare": {
        "graph_name": "ReduceSumSquare_sample",
        "inputs": {
            "data": {"data": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)},
            "axes": {"data": np.array([], dtype=np.int64), "type": "int64"}
        },
        "outputs": {
            "reduced": None
        },
        "ReduceSumSquare_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "ReduceSumSquare",
            "inputs": {
                "data": "data",
                "axes": "axes"
            },
            "outputs": {
                "reduced": "reduced"
            },
            "keepdims": 1
        }
    },
    "RegexFullMatch": {
        "graph_name": "RegexFullMatch_sample",
        "inputs": {
            "X": {"data": np.array(["www.google.com", "www.facebook.com", "www.bbc.co.uk"]).astype(object), "type": "object"}
        },
        "outputs": {
            "Y": {"data": np.empty(shape=(1), dtype=bool), "type": "bool"}
        },
        "RegexFullMatch_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "RegexFullMatch",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            },
            "pattern": r"www\.[\w.-]+\.\bcom\b"
        }
    },
    "Relu": {
        "graph_name": "Relu_sample",
        "inputs": {
            "x": {"data": np.random.randn(3, 4, 5).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "Relu_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Relu",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "Reshape": {
        "graph_name": "Reshape_sample",
        "inputs": {
            "X1": {"data": np.random.random_sample([2, 3, 4]).astype(np.float32)},
            "X2": {"data": np.array([4, 2, 3], dtype=np.int64), "type": "int64"}
        },
        "outputs": {
            "Y": None
        },
        "Reshape_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Reshape",
            "inputs": {
                "X1": "X1",
                "X2": "X2"
            },
            "outputs": {
                "Y": "Y"
            },
            "allowzero": 1
        }
    },
    "Resize": {
        "graph_name": "Resize_sample",
        "inputs": {
            "X": {"data": np.array([[[
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]]],dtype=np.float32,)},
            "": "",
            "scales": {"data": np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)}
        },
        "outputs": {
            "Y": None
        },
        "Resize_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Resize",
            "inputs": {
                "x": "X",
                "": "",
                "scales": "scales"
            },
            "outputs": {
                "y": "Y"
            },
            "mode": "cubic"
        }
    },
    "ReverseSequence": {
        "graph_name": "ReverseSequence_sample",
        "inputs": {
            "x": {"data": np.array([
                        [0.0, 1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0, 7.0],
                        [8.0, 9.0, 10.0, 11.0],
                        [12.0, 13.0, 14.0, 15.0],],dtype=np.float32,)},
            "sequence_lens": {"data": np.array([1, 2, 3, 4], dtype=np.int64), "type": "int64"}
        },
        "outputs": {
            "y": None
        },
        "ReverseSequence_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "ReverseSequence",
            "inputs": {
                "x": "x",
                "sequence_lens": "sequence_lens"
            },
            "outputs": {
                "y": "y"
            },
            "time_axis": 1,
            "batch_axis": 0
        }
    },
    "RoiAlign": {
        "graph_name": "RoiAlign_sample",
        "inputs": {
            "X": {"data": np.array([[[
                            [0.2764, 0.7150, 0.1958, 0.3416, 0.4638, 0.0259, 0.2963, 0.6518, 0.4856, 0.7250,],
                            [0.9637, 0.0895, 0.2919, 0.6753, 0.0234, 0.6132, 0.8085, 0.5324, 0.8992, 0.4467,],
                            [0.3265, 0.8479, 0.9698, 0.2471, 0.9336, 0.1878, 0.4766, 0.4308, 0.3400, 0.2162,],
                            [0.0206, 0.1720, 0.2155, 0.4394, 0.0653, 0.3406, 0.7724, 0.3921, 0.2541, 0.5799,],
                            [0.4062, 0.2194, 0.4473, 0.4687, 0.7109, 0.9327, 0.9815, 0.6320, 0.1728, 0.6119,],
                            [0.3097, 0.1283, 0.4984, 0.5068, 0.4279, 0.0173, 0.4388, 0.0430, 0.4671, 0.7119,],
                            [0.1011, 0.8477, 0.4726, 0.1777, 0.9923, 0.4042, 0.1869, 0.7795, 0.9946, 0.9689,],
                            [0.1366, 0.3671, 0.7011, 0.6234, 0.9867, 0.5585, 0.6985, 0.5609, 0.8788, 0.9928,],
                            [0.5697, 0.8511, 0.6711, 0.9406, 0.8751, 0.7496, 0.1650, 0.1049, 0.1559, 0.2514,],
                            [0.7012, 0.4056, 0.7879, 0.3461, 0.0415, 0.2998, 0.5094, 0.3727, 0.5482, 0.0502,],]]],dtype=np.float32,)},
            "rois": {"data": np.array([[0, 0, 9, 9], [0, 5, 4, 9], [5, 5, 9, 9]], dtype=np.float32)},
            "batch_indices": {"data": np.array([0, 0, 0], dtype=np.int64), "type": "int64"},
        },
        "outputs": {
            "y": None
        },
        "RoiAlign_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "RoiAlign",
            "inputs": {
                "x": "X",
                "rois": "rois",
                "batch_indices": "batch_indices"
            },
            "outputs": {
                "y": "y"
            },
            "spatial_scale": 1.0,
            "output_height": 5,
            "output_width": 5,
            "sampling_ratio": 2,
            "coordinate_transformation_mode": "half_pixel",
        }
    },
    "Round": {
        "graph_name": "Round_sample",
        "inputs": {
            "x": {"data": np.array([0.1, 0.5, 0.9, 1.2, 1.5, 1.8, 2.3, 2.5, 2.7, -1.1, -1.5, -1.9, -2.2,  -2.5, -2.8,]).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "Round_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Round",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "STFT": { },
    "Scan": { },
    "Scatter": { },
    "ScatterElements": {
        "graph_name": "ScatterElements_sample",
        "inputs": {
            "data": {"data": np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)},
            "indices": {"data": np.array([[1, 1]], dtype=np.int64), "type": "int64"},
            "updates": {"data": np.array([[1.1, 2.1]], dtype=np.float32)}
        },
        "outputs": {
            "y": None
        },
        "ScatterElements_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "ScatterElements",
            "inputs": {
                "data": "data",
                "indices": "indices",
                "updates": "updates"
            },
            "outputs": {
                "y": "y"
            },
            "axis": 1,
            "reduction": "add",
        }
    },
    "ScatterND": {
        "graph_name": "ScatterND_sample",
        "inputs": {
            "data": {"data": np.array([
                    [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                    [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                    [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
                    [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],],dtype=np.float32,)},
            "indices": {"data": np.array([[0], [0]], dtype=np.int64), "type": "int64"},
            "updates": {"data": np.array([
                    [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                    [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],],dtype=np.float32,)}
        },
        "outputs": {
            "y": None
        },
        "ScatterND_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "ScatterND",
            "inputs": {
                "data": "data",
                "indices": "indices",
                "updates": "updates"
            },
            "outputs": {
                "y": "y"
            },
            "reduction": "mul",
        }
    },
    "Selu": {
        "graph_name": "Selu_sample",
        "inputs": {
            "x": {"data": np.array([-1, 0, 1]).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "Selu_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Selu",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            },
            "alpha": 2.0,
            "gamma": 3.0
        }
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
            "x": {"data": np.random.randn(3, 4, 5).astype(np.float32)}
        },
        "outputs": {
            "y": {"data": np.empty(shape=(1), dtype=np.int64), "type": "int64"}
        },
        "Shape_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Shape",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            },
            "start": 1,
            "end": 2,
        }
    },
    "Shrink": {
        "graph_name": "Shrink_sample",
        "inputs": {
            "x": {"data": np.arange(-2.0, 2.1, dtype=np.float32)}
        },
        "outputs": {
            "y": None
        },
        "Shrink_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Shrink",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            },
            "lambd": 1.5,
            "bias": 1.5
        }
    },
    "Sigmoid": {
        "graph_name": "Sigmoid_sample",
        "inputs": {
            "x": {"data": np.array([-1, 0, 1]).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "Sigmoid_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Sigmoid",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "Sign": {
        "graph_name": "Sign_sample",
        "inputs": {
            "x": {"data": np.array(range(-5, 6)).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "Sign_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Sign",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "Sin": {
        "graph_name": "Sin_sample",
        "inputs": {
            "x": {"data": np.array([-1, 0, 1]).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "Sin_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Sin",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "Sinh": {
        "graph_name": "Sinh_sample",
        "inputs": {
            "x": {"data": np.array([-1, 0, 1]).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "Sinh_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Sinh",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "Size": {
        "graph_name": "Size_sample",
        "inputs": {
            "x": {"data": np.array([
                        [1, 2, 3],
                        [4, 5, 6],]).astype(np.float32)}
        },
        "outputs": {
            "y": {"data": np.empty(shape=(1), dtype=np.int64), "type": "int64"}
        },
        "Size_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Size",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "Slice": {
        "graph_name": "Slice_sample",
        "inputs": {
            "x": {"data": np.random.randn(20, 10, 5).astype(np.float32)},
            "starts": {"data": np.array([0, 0], dtype=np.int64), "type": "int64"},
            "ends": {"data": np.array([3, 10], dtype=np.int64), "type": "int64"},
            "axes": {"data": np.array([0, 1], dtype=np.int64), "type": "int64"},
            "steps": {"data": np.array([1, 1], dtype=np.int64), "type": "int64"}
        },
        "outputs": {
            "y": None
        },
        "Slice_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Slice",
            "inputs": {
                "x": "x",
                "starts": "starts",
                "ends": "ends",
                "axes": "axes",
                "steps": "steps"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "Softmax": {
        "graph_name": "Softmax_sample",
        "inputs": {
            "x": {"data": np.abs(np.random.randn(3, 4, 5).astype(np.float32))}
        },
        "outputs": {
            "y": None
        },
        "Softmax_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Softmax",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            },
            "axis": 0
        }
    },
    "SoftmaxCrossEntropyLoss": {
        "graph_name": "SoftmaxCrossEntropyLoss_sample",
        "inputs": {
            "x": {"data": np.random.rand(3, 5).astype(np.float32)},
            "labels": {"data": np.random.randint(0, high=5, size=(3,)).astype(np.int64), "type": "int64"}
        },
        "outputs": {
            "y": None
        },
        "SoftmaxCrossEntropyLoss_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "SoftmaxCrossEntropyLoss",
            "inputs": {
                "x": "x",
                "labels": "labels"
            },
            "outputs": {
                "y": "y"
            },
            "reduction": "sum"
        }
    },
    "Softplus": {
        "graph_name": "Softplus_sample",
        "inputs": {
            "x": {"data": np.array([-1, 0, 1]).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "Softplus_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Softplus",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "Softsign": {
        "graph_name": "Softsign_sample",
        "inputs": {
            "x": {"data": np.array([-1, 0, 1]).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "Softsign_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Softsign",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "SpaceToDepth": {
        "graph_name": "SpaceToDepth_sample",
        "inputs": {
            "x": {"data": np.array([[[
                    [0, 6, 1, 7, 2, 8],
                    [12, 18, 13, 19, 14, 20],
                    [3, 9, 4, 10, 5, 11],
                    [15, 21, 16, 22, 17, 23],
                ]]]).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "SpaceToDepth_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "SpaceToDepth",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            },
            "blocksize": 2
        }
    },
    "Split": {
        "graph_name": "Split_sample",
        "inputs": {
            "x": {"data":np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32)},
            "split": {"data":np.array([2, 4]).astype(np.int64), "type": "int64"}
        },
        "outputs": {
            "output_1": None,
            "output_2": None
        },
        "Split_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Split",
            "inputs": {
                "x": "x",
                "split": "split"
            },
            "outputs": {
                "output_1": "output_1",
                "output_2": "output_2"
            },
            "axis": 0
        }
    },
    "SplitToSequence": { },
    "Sqrt": {
        "graph_name": "Sqrt_sample",
        "inputs": {
            "x": {"data": np.array([1, 4, 9]).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "Sqrt_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Sqrt",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "Squeeze": {
        "graph_name": "Squeeze_sample",
        "inputs": {
            "x": {"data":np.random.randn(1, 3, 4, 5).astype(np.float32)},
            "axes": {"data":np.array([0], dtype=np.int64), "type": "int64"}
        },
        "outputs": {
            "y": None
        },
        "Squeeze_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Squeeze",
            "inputs": {
                "x": "x",
                "axes": "axes"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "StringConcat": {
        "graph_name": "StringConcat_sample",
        "inputs": {
            "x": {"data": np.array(["abc", "def"]).astype("object"), "type": "object"},
            "y": {"data": np.array([".com", ".net"]).astype("object"), "type": "object"}
        },
        "outputs": {
            "result": None
        },
        "StringConcat_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "StringConcat",
            "inputs": {
                "x": "x",
                "y": "y"
            },
            "outputs": {
                "result": "result"
            }
        }
    },
    "StringNormalizer": {
        "graph_name": "StringNormalizer_sample",
        "inputs": {
            "x": {"data": np.array(["monday", "tuesday", "wednesday", "thursday"]).astype(object), "type": "object"}
        },
        "outputs": {
            "y": None
        },
        "StringNormalizer_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "StringNormalizer",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            },
            "case_change_action": "LOWER",
            "is_case_sensitive": 1,
            "stopwords": ["monday"],
        }
    },
    "StringSplit": { },
    "Sub": {
        "graph_name": "Sub_sample",
        "inputs": {
            "x": {"data": np.array([1, 2, 3]).astype(np.float32)},
            "y": {"data": np.array([3, 2, 1]).astype(np.float32)}
        },
        "outputs": {
            "z": None
        },
        "Sub_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Sub",
            "inputs": {
                "x": "x",
                "y": "y"
            },
            "outputs": {
                "z": "z"
            }
        }
    },
    "Sum": {
        "graph_name": "Sum_sample",
        "inputs": {
            "x1": {"data": np.array([3, 0, 2]).astype(np.float32)},
            "x2": {"data": np.array([1, 3, 4]).astype(np.float32)},
            "x3": {"data": np.array([2, 6, 6]).astype(np.float32)}
        },
        "outputs": {
            "result": None
        },
        "Sum_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Sum",
            "inputs": {
                "x1": "x1",
                "x2": "x2",
                "x3": "x3"
            },
            "outputs": {
                "result": "result"
            }
        }
    },
    "Tan": {
        "graph_name": "Tan_sample",
        "inputs": {
            "x": {"data": np.array([-1, 0, 1]).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "Tan_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Tan",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "Tanh": {
        "graph_name": "Tanh_sample",
        "inputs": {
            "x": {"data": np.array([-1, 0, 1]).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "Tanh_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Tanh",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "TfIdfVectorizer": { },
    "ThresholdedRelu": {
        "graph_name": "ThresholdedRelu_sample",
        "inputs": {
            "x": {"data": np.array([-1.5, 0.0, 1.2, 2.0, 2.2]).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "ThresholdedRelu_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "ThresholdedRelu",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            },
            "alpha": 2.0
        }
    },
    "Tile": {
        "graph_name": "Tile_sample",
        "inputs": {
            "x": {"data":np.array([[0, 1], [2, 3]], dtype=np.float32)},
            "repeat": {"data":np.array([1, 2], dtype=np.int64), "type": "int64"}
        },
        "outputs": {
            "z": None
        },
        "Tile_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Tile",
            "inputs": {
                "x": "x",
                "repeat": "repeat"
            },
            "outputs": {
                "z": "z"
            }
        }
    },
    "TopK": {
        "graph_name": "TopK_sample",
        "inputs": {
            "x": {"data": np.array([
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [11, 10, 9, 8],],dtype=np.float32,)},
            "K": {"data": np.array([3], dtype=np.int64), "type": "int64"}
        },
        "outputs": {
            "values": None,
            "indices": {"data": np.empty(shape=(1), dtype=np.int64), "type": "int64"}
        },
        "TopK_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "TopK",
            "inputs": {
                "x": "x",
                "K": "K"
            },
            "outputs": {
                "values": "values",
                "indices": "indices"
            },
            "axis": 1,
            "largest": 0,
            "sorted": 1,
        }
    },
    "Transpose": {
        "graph_name": "Transpose_sample",
        "inputs": {
            "x": {"data": np.random.random_sample((2, 3, 4)).astype(np.float32)}
        },
        "outputs": {
            "y": None
        },
        "Transpose_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Transpose",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "y"
            },
            "perm": (0, 2, 1)
        }
    },
    "Trilu": {
        "graph_name": "Trilu_sample",
        "inputs": {
            "x": {"data":np.random.randint(10, size=(2, 3, 3)).astype(np.int64), "type": "int64"},
            "k": {"data":np.array(-1).astype(np.int64), "type": "int64"}
        },
        "outputs": {
            "y": None
        },
        "Trilu_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Trilu",
            "inputs": {
                "x": "x",
                "k": "k"
            },
            "outputs": {
                "y": "y"
            },
            "upper": 0
        }
    },
    "Unique": {
        "graph_name": "Unique_sample",
        "inputs": {
            "x": {"data": np.array([[1, 0, 0], [1, 0, 0], [2, 3, 3]], dtype=np.float32)}
        },
        "outputs": {
            "Y": None,
            "indices": {"data": np.empty(shape=(1), dtype=np.int64), "type": "int64"},
            "inverse_indices": {"data": np.empty(shape=(1), dtype=np.int64), "type": "int64"},
            "counts": {"data": np.empty(shape=(1), dtype=np.int64), "type": "int64"},
        },
        "Unique_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Unique",
            "inputs": {
                "x": "x"
            },
            "outputs": {
                "y": "Y",
                "indices": "indices",
                "inverse_indices": "inverse_indices",
                "counts": "counts"
            },
            "sorted": 1,
            "axis": -1,
        }
    },
    "Unsqueeze": {
        "graph_name": "Unsqueeze_sample",
        "inputs": {
            "x": {"data": np.random.randn(1, 3, 1, 5).astype(np.float32)},
            "axes": {"data": np.array([-2]).astype(np.int64), "type": "int64"}
        },
        "outputs": {
            "y": None
        },
        "Unsqueeze_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Unsqueeze",
            "inputs": {
                "x": "x",
                "axes": "axes"
            },
            "outputs": {
                "y": "y"
            }
        }
    },
    "Upsample": { },
    "Where": {
        "graph_name": "Where_sample",
        "inputs": {
            "condition": {"data": np.array([[1, 0], [1, 1]], dtype=bool), "type": "bool"},
            "x": {"data": np.array([[1, 2], [3, 4]], dtype=np.int64), "type": "int64"},
            "y": {"data": np.array([[9, 8], [7, 6]], dtype=np.int64), "type": "int64"}
        },
        "outputs": {
            "z": {"data": np.empty(shape=(1), dtype=np.int64), "type": "int64"}
        },
        "Where_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Where",
            "inputs": {
                "condition": "condition",
                "x": "x",
                "y": "y"
            },
            "outputs": {
                "z": "z"
            }
        }
    },
    "Xor": {
        "graph_name": "Xor_sample",
        "inputs": {
            "x": {"data": (np.random.randn(3, 4, 5) > 0).astype(bool), "type": "bool"},
            "y": {"data": (np.random.randn(3, 4, 5) > 0).astype(bool), "type": "bool"}
        },
        "outputs": {
            "z": None
        },
        "Xor_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Xor",
            "inputs": {
                "x": "x",
                "y": "y"
            },
            "outputs": {
                "z": "z"
            }
        },
    },
}

left_operators = \
{
    "Abs": {
        "graph_name": "Reshape_sample",
        "inputs": {
            "X": {
                "data": [[0.06329948, -1.0832994 , 0.37930292], [0.71035045, -1.6637981 , 1.0044696]],
                "type": "float64"
            }
        },
        "outputs": {
            "Y": None
        },
        "Abs_Node":
        {
            "Action": "Add",
            "Category": "Node",
            "Type": "Abs",
            "inputs": {
                "x": "X"
            },
            "outputs": {
                "y": "Y"
            }
        }
    }
}

right_operators = \
{
    "Abs": 
    {
        "list":[
            {
                "graph_name": "Reshape_sample",
                "inputs": {
                    "x": {"data": [[0.06329948, -1.0832994 , 0.37930292], [0.71035045, -1.6637981 , 1.0044696]],},
                    "y": {"data": [-1]}
                },
                "outputs": {
                    "z": None
                },
                "Mul_Node":
                {
                    "Action": "Add",
                    "Category": "Node",
                    "Type": "Mul",
                    "inputs": {
                        "x": "x",
                        "y": "y"
                    },
                    "outputs": {
                        "output": "output"
                    }
                },
                "Max_Node":
                {
                    "Action": "Add",
                    "Category": "Node",
                    "Type": "Max",
                    "inputs": {
                        "data_0": "x",
                        "data_1": "output"
                    },
                    "outputs": {
                        "z": "z"
                    }
                }
            },
            {
                "graph_name": "Reshape_sample",
                "inputs": {
                    "x": {"data": [[0.06329948, -1.0832994 , 0.37930292], [0.71035045, -1.6637981 , 1.0044696]],},
                    "y": {"data": [-1]}
                },
                "outputs": {
                    "z": None
                },
                "Mul_Node":
                {
                    "Action": "Add",
                    "Category": "Node",
                    "Type": "Mul",
                    "inputs": {
                        "x": "x",
                        "y": "y"
                    },
                    "outputs": {
                        "output": "Mul_Node_output"
                    }
                },
                "Relu_Node_1":
                {
                    "Action": "Add",
                    "Category": "Node",
                    "Type": "Relu",
                    "inputs": {
                        "a": "Mul_Node_output"
                    },
                    "outputs": {
                        "output": "Relu_Node_1_output"
                    }
                },
                "Relu_Node_2":
                {
                    "Action": "Add",
                    "Category": "Node",
                    "Type": "Relu",
                    "inputs": {
                        "a": "x"
                    },
                    "outputs": {
                        "output": "Relu_Node_2_output"
                    }
                },
                "Add_Node":
                {
                    "Action": "Add",
                    "Category": "Node",
                    "Type": "Add",
                    "inputs": {
                        "x1": "Relu_Node_1_output",
                        "x2": "Relu_Node_2_output"
                    },
                    "outputs": {
                        "y": "z"
                    }
                }
            }
        ]
    }
}

def main():
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", "-op", help="Operator name", default="")
    parser.add_argument("--modify", "-md", help="Setting to modify network", default="")
    parser.add_argument("--input_onnx", "-io", help="Input ONNX file path")
    parser.add_argument("--output_onnx", "-oo", help="Output ONNX file path")
    args = parser.parse_args()

    # if args.operator != "":
    #     model = ONMAModel()
    #     inf1 = model.ONMAModel_CreateNetworkFromGraph(default_input[args.operator])
    #     model.ONMAModel_DisplayInformation(inf1, **default_input[args.operator])

    # if args.modify != "":
    #     with open(args.modify) as user_file:
    #         file_contents = user_file.read()
    #     json_contents = json.loads(file_contents)

    #     model = ONMAModel()
    #     model.ONMAModel_LoadModel(args.input_onnx)
    #     model.ONMAModel_UpdateModel(json_contents, args.output_onnx)

    model = ONMAModel()
    inf1 = model.ONMAModel_CreateNetworkFromGraph(left_operators["Abs"])
    inf2 = model.ONMAModel_CreateNetworkFromGraph(right_operators["Abs"]["list"][1])

    print(inf1)
    print(inf2)

if main() == False:
    sys.exit(-1)
