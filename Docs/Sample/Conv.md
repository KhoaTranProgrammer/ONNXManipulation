# Explanation of Conv.json

This sample describes an ONMA graph containing its input tensors, constant initializers, computation nodes, and outputs.

## Graph overview

1. Provide input tensor `X` with shape `[1, 1, 4, 4]`.
2. Apply the `Conv` node `Conv_Node` using inputs `["X", "W"]`.
3. Produce output tensor `Y` with shape `[1, 1, 3, 3]`.

## Top-level model

| Field | Meaning | Sample value |
| --- | --- | --- |
| name | Human-readable model name. | Conv_Sample |
| graph | Inputs, outputs, initializers, and computation nodes. | Object |

## Input and output tensors

Four-dimensional tensor shapes use ONNX NCHW order: batch, channels, height, width.

| Tensor | Shape | Interpretation | Data type |
| --- | --- | --- | --- |
| X | [1, 1, 4, 4] | One batch, 1 channel(s), 4 × 4 spatial dimensions (NCHW). | float32 |
| Y | [1, 1, 3, 3] | One batch, 1 channel(s), 3 × 3 spatial dimensions (NCHW). | float32 |

Input entries declare tensor metadata, not runtime input values. Values must be supplied when the generated model runs.

## Weight initializers

| Name | Shape | Data type | Purpose |
| --- | --- | --- | --- |
| W | [1, 1, 2, 2] | float32 | Convolution weight tensor. |

`W` has shape `[1, 1, 2, 2]`: 1 output channel(s), 1 input channel(s), and a `[2, 2]` kernel.

```text
W[0, 0]:
0.25 0.25
0.25 0.25
```

At each output position, the kernel is multiplied element by element with an input window and the products are summed.

## Convolution node

### Conv_Node — Conv

| Field | Value | Meaning |
| --- | --- | --- |
| name | Conv_Node | Identifies this graph node. |
| op_type | Conv | ONNX convolution operator. |
| inputs | ["X", "W"] | Input activation, weights, and optional bias, in that order. |
| outputs | ["Y"] | Names of the tensors produced by this node. |
| kernel_shape | [2, 2] | Height and width of the kernel. |
| strides | [1, 1] | Step size vertically and horizontally. |
| pads | [0, 0, 0, 0] | Padding at the beginning of each spatial axis, followed by padding at the end of each axis. |

This node applies an ONNX Conv operation to `X` using `W` as its weight input. The kernel shape is `[2, 2]`. The strides are `[1, 1]`. Padding is `[0, 0, 0, 0]` (ONNX order: beginning pads followed by ending pads for each spatial axis). For the declared input shape `[1, 1, 4, 4]`, the calculated spatial output size is `3 × 3` using the Conv kernel, stride, and padding settings. The `W` weights are all `0.25` and sum to 1, so each output is the average of its input window. No bias input is specified.

> **Expected behavior:** The weights are a normalized `[2, 2]` kernel, so the 9 output values are overlapping input-window averages.

### Why is the output this size?

For input `[1, 1, 4, 4]`, kernel `[2, 2]`, stride `[1, 1]`, and padding `[0, 0, 0, 0]`, the spatial output dimensions are calculated as `floor((4 + 0 + 0 - 1 × (2 - 1) - 1) / 1) + 1 = 3` for height and `floor((4 + 0 + 0 - 1 × (2 - 1) - 1) / 1) + 1 = 3` for width. The calculated output shape is `[1, 1, 3, 3]`.

> **Expected behavior:** The declared output shape `[1, 1, 3, 3]` matches the calculated shape.

## Source file

- [Conv.json](../../Sample/Conv.json)
