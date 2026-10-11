# Replace_Add_Scalar_By_Conv

## Introduction

### Description

Replace Add Scalar by Conv Operation

This transformation pattern replaces an Add operation that combines a tensor with a scalar value by an equivalent Conv-based computation. It is useful when the target backend or optimization flow prefers Conv over elementwise addition for scalar bias handling.

### Mathematical Equivalence

Add(x, s) = Conv(x, W, b) where W is an identity kernel and b is the expanded scalar value, applied with a 1x1 convolution and no padding or stride.

### Transformation Steps

1. Detect an Add node that combines a tensor with a scalar initializer
2. Expand the scalar into a broadcastable shape matching the output channels
3. Create a 1x1 Conv with identity weights and the expanded bias
4. Remove the original Add node
### Benefits

- Enables backend support for Conv-based scalar addition
- Preserves semantic equivalence using a standard convolution primitive
- Supports decomposition into a more uniform graph representation
### Preconditions

- The Add node must involve a tensor and a scalar initializer
- The input tensor must be compatible with Conv semantics
- The scalar must be broadcastable to the output shape
### Example

**Input tensor:** X with shape [N, C, H, W] **Scalar:** s = 3.0 Original Add computation:

```text
Y = X + s
```

Equivalent Conv-based form:

```text
1. Expand s to a bias tensor of shape [C, 1, 1]
2. Use identity weights W = I_C so that Conv(X, W, b) computes X + b
3. Apply a 1x1 convolution with no padding and stride 1
4. The result is equivalent to the original Add operation
```

### Result

A Conv-based replacement that preserves the Add semantics

## SearchBy

![SearchBy](Replace_Add_Scalar_By_Conv_SearchBy.png)

## ReplaceBy

![ReplaceBy](Replace_Add_Scalar_By_Conv_ReplaceBy.png)
