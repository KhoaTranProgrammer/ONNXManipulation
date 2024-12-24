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

    def ONMAMakeNode(
            self,
            name,
            inputs,
            outputs,
            direction=None,
            alpha=None,
            axes=None,
            axis=None,
            values=None,
            kernel_shape=None,
            pads=None,
            allowzero=None,
            exclusive=None,
            reverse=None,
            equation=None,
            beta=None,
            detect_positive=None,
            detect_negative=None,
            bias=None,
            size=None,
            fmod=None,
            lambd=None,
            align_corners=None,
            keepdims=None,
            select_last_index=None,
            strides=None,
            ceil_mode=None,
            dilations=None,
            count_include_pad=None,
            auto_pad=None,
            epsilon=None,
            training_mode=None,
            seed=None,
            periodic=None,
            pattern=None,
            mode=None,
            cubic_coeff_a=None,
            exclude_outside=None,
            coordinate_transformation_mode=None,
            antialias=None,
            keep_aspect_ratio_policy=None,
            extrapolation_value=None,
            nearest_mode=None,
            time_axis=None,
            batch_axis=None,
            spatial_scale=None,
            output_height=None,
            output_width=None,
            sampling_ratio=None,
            to=None,
            block_size=None,
            ratio=None,
            k=None,
            dtype=None,
            batch_dims=None,
            transA=None,
            transB=None,
            padding_mode=None,
            num_groups=None,
            p=None,
            reduction=None,
            ignore_index=None,
            center_point_box=None,
            output_dtype=None,
            gamma=None,
            start=None,
            end=None,
            blocksize=None,
        ):

        isCreated = False
        try:
            if values == None:
                self._node = onnx.helper.make_node(
                    name,
                    inputs=inputs,
                    outputs=outputs,
                    direction=direction,
                    alpha=alpha,
                    axes=axes,
                    axis=axis,
                    kernel_shape=kernel_shape,
                    pads=pads,
                    allowzero=allowzero,
                    exclusive=exclusive,
                    reverse=reverse,
                    equation=equation,
                    beta=beta,
                    detect_positive=detect_positive,
                    detect_negative=detect_negative,
                    bias=bias,
                    size=size,
                    fmod=fmod,
                    lambd=lambd,
                    align_corners=align_corners,
                    keepdims=keepdims,
                    select_last_index=select_last_index,
                    strides=strides,
                    ceil_mode=ceil_mode,
                    dilations=dilations,
                    count_include_pad=count_include_pad,
                    auto_pad=auto_pad,
                    epsilon=epsilon,
                    training_mode=training_mode,
                    seed=seed,
                    periodic=periodic,
                    pattern=pattern,
                    mode=mode,
                    cubic_coeff_a=cubic_coeff_a,
                    exclude_outside=exclude_outside,
                    coordinate_transformation_mode=coordinate_transformation_mode,
                    antialias=antialias,
                    keep_aspect_ratio_policy=keep_aspect_ratio_policy,
                    extrapolation_value=extrapolation_value,
                    nearest_mode=nearest_mode,
                    time_axis=time_axis,
                    batch_axis=batch_axis,
                    spatial_scale=spatial_scale,
                    output_height=output_height,
                    output_width=output_width,
                    sampling_ratio=sampling_ratio,
                    to=to,
                    block_size=block_size,
                    ratio=ratio,
                    k=k,
                    dtype=dtype,
                    batch_dims=batch_dims,
                    transA=transA,
                    transB=transB,
                    padding_mode=padding_mode,
                    num_groups=num_groups,
                    p=p,
                    reduction=reduction,
                    ignore_index=ignore_index,
                    center_point_box=center_point_box,
                    output_dtype=output_dtype,
                    gamma=gamma,
                    start=start,
                    end=end,
                    blocksize=blocksize,
                )
                isCreated = True
        except:
            pass

        try:
            if values.all() and isCreated == False:
                self._node = onnx.helper.make_node(
                    name,
                    inputs=inputs,
                    outputs=outputs,
                    value=onnx.helper.make_tensor(
                        name="const_tensor",
                        data_type=onnx.TensorProto.FLOAT,
                        dims=values.shape,
                        vals=values.flatten().astype(float),
                    )
                )
                isCreated = True
        except:
            pass

        try:
            if isCreated == False:
                self._node = onnx.helper.make_node(
                        name,
                        inputs=inputs,
                        outputs=outputs,
                        value=values
                    )
                isCreated = True
        except:
            pass

    def ONMAGetNode(self):
        return self._node
    
    def ONMACreateInput(self, name, type, dimension):
        return make_tensor_value_info(name, type, dimension)
