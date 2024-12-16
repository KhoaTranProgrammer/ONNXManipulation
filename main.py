import sys
import onnx
import argparse
import numpy as np
from ONMAOperators import ONMAOperators

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
    "AffineGrid": { },
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
    "ArgMax": { },
    "ArgMin": { },
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
    "AveragePool": { },
    "BatchNormalization": { },
    "Bernoulli": { },
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
    "BlackmanWindow": { },
    "Cast": { },
    "CastLike": { },
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
    "ConstantOfShape": { },
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
    "ConvInteger": { },
    "ConvTranspose": { },
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
    "DFT": { },
    "DeformConv": { },
    "DepthToSpace": { },
    "DequantizeLinear": { },
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
    "Dropout": { },
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
    "Equal": { },
    "Erf": { },
    "Exp": { },
    "Expand": { },
    "EyeLike": { },
    "Flatten": { },
    "Floor": { },
    "GRU": { },
    "Gather": { },
    "GatherElements": { },
    "GatherND": { },
    "Gelu": { },
    "Gemm": { },
    "GlobalAveragePool": { },
    "GlobalLpPool": { },
    "GlobalMaxPool": { },
    "Greater": { },
    "GreaterOrEqual": { },
    "GridSample": { },
    "GroupNormalization": { },
    "HammingWindow": { },
    "HannWindow": { },
    "HardSigmoid": { },
    "HardSwish": { },
    "Hardmax": { },
    "Identity": { },
    "If": { },
    "ImageDecoder": { },
    "InstanceNormalization": { },
    "IsInf": { },
    "IsNaN": { },
    "LRN": { },
    "LSTM": { },
    "LayerNormalization": { },
    "LeakyRelu": { },
    "Less": { },
    "LessOrEqual": { },
    "Log": { },
    "LogSoftmax": { },
    "Loop": { },
    "LpNormalization": { },
    "LpPool": { },
    "MatMul": { },
    "MatMulInteger": { },
    "Max": { },
    "MaxPool": { },
    "MaxRoiPool": { },
    "MaxUnpool": { },
    "Mean": { },
    "MeanVarianceNormalization": { },
    "MelWeightMatrix": { },
    "Min": { },
    "Mish": { },
    "Mod": { },
    "Mul": { },
    "Multinomial": { },
    "Neg": { },
    "NegativeLogLikelihoodLoss": { },
    "NonMaxSuppression": { },
    "NonZero": { },
    "Not": { },
    "OneHot": { },
    "Optional": { },
    "OptionalGetElement": { },
    "OptionalHasElement": { },
    "Or": { },
    "PRelu": { },
    "Pad": { },
    "Pow": { },
    "QLinearConv": { },
    "QLinearMatMul": { },
    "QuantizeLinear": { },
    "RNN": { },
    "RandomNormal": { },
    "RandomNormalLike": { },
    "RandomUniform": { },
    "RandomUniformLike": { },
    "Range": { },
    "Reciprocal": { },
    "ReduceL1": { },
    "ReduceL2": { },
    "ReduceLogSum": { },
    "ReduceLogSumExp": { },
    "ReduceMax": { },
    "ReduceMean": { },
    "ReduceMin": { },
    "ReduceProd": { },
    "ReduceSum": { },
    "ReduceSumSquare": { },
    "RegexFullMatch": { },
    "Relu": { },
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
    "Resize": { },
    "ReverseSequence": { },
    "RoiAlign": { },
    "Round": { },
    "STFT": { },
    "Scan": { },
    "Scatter": { },
    "ScatterElements": { },
    "ScatterND": { },
    "Selu": { },
    "SequenceAt": { },
    "SequenceConstruct": { },
    "SequenceEmpty": { },
    "SequenceErase": { },
    "SequenceInsert": { },
    "SequenceLength": { },
    "SequenceMap": { },
    "Shape": { },
    "Shrink": { },
    "Sigmoid": { },
    "Sign": { },
    "Sin": { },
    "Sinh": { },
    "Size": { },
    "Slice": { },
    "Softmax": { },
    "SoftmaxCrossEntropyLoss": { },
    "Softplus": { },
    "Softsign": { },
    "SpaceToDepth": { },
    "Split": { },
    "SplitToSequence": { },
    "Sqrt": { },
    "Squeeze": { },
    "StringConcat": { },
    "StringNormalizer": { },
    "StringSplit": { },
    "Sub": { },
    "Sum": { },
    "Tan": { },
    "Tanh": { },
    "TfIdfVectorizer": { },
    "ThresholdedRelu": { },
    "Tile": { },
    "TopK": { },
    "Transpose": { },
    "Trilu": { },
    "Unique": { },
    "Unsqueeze": { },
    "Upsample": { },
    "Where": { },
    "Xor": { },
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
