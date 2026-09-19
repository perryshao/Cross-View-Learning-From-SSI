"""Lambda-layer helpers that turn a skeleton sequence into Self-Similarity Images.

Shared by `ntu-latefusion-spp-metric.py`, `ntu-earlyfusion-spp-metric.py` and
`ntu-earlyfusion-spp-metric-c3d.py`.

The pairwise joint graph of Eq. (1) is built by broadcasting the joint tensor
twice and subtracting:

    (batch, T, N, 3)  --repeat_x_onejoint-->   (batch, T, N*N, 3)   [p_i tiled]
    (batch, T, N, 3)  --repeat_x_groupjoint--> (batch, T, N*N, 3)   [p_j tiled]
    subtract                                -> (batch, T, N*N, 3)   [p_i - p_j]

`MetricLayer` then maps that to the SSI, and `flatten_ssm` reshapes the flat
N*N axis back into an N x N image so the 3D CNN branch can consume it.
"""

import numpy as np
from keras import backend as K


## ---------------------------------------------------------------------------
## pairwise joint graph building (Eq. 1)
## ---------------------------------------------------------------------------

def repeat_x_onejoint(x):
    shape = x.shape.as_list()
    x = K.repeat_elements(x, shape[-2], axis=-2)
    return x


def repeat_x_onejoint_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4  # only valid for 4D tensors
    shape = [input_shape[0], input_shape[1], input_shape[2] * input_shape[2], input_shape[-1]]
    return tuple(shape)


def repeat_x_groupjoint(x):
    shape = x.shape.as_list()
    x = K.repeat_elements(x, shape[-2], axis=-3)
    x = K.reshape(x, [-1, shape[1], shape[-2] * shape[-2], shape[-1]])
    return x


def repeat_x_groupjoint_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4  # only valid for 4D tensors
    shape = [input_shape[0], input_shape[1], input_shape[2] * input_shape[2], input_shape[-1]]
    return tuple(shape)


## ---------------------------------------------------------------------------
## reshaping between the flat (N*N) SSI axis and the N x N image
## ---------------------------------------------------------------------------

def make_flatten_ssm(channels=1):
    """Return the `(fn, output_shape_fn)` pair for a Lambda that reshapes
    `(batch, T, N*N, c)` into `(batch, T, N, N, c)`.

    `channels` is 1 for the light-weight CNN branches and 3 for the Sports-1M
    pretrained C3D branch (see `MetricLayer_ForC3D`).
    """

    def flatten_ssm(x):
        shape = x.shape.as_list()
        return K.reshape(x, [-1, shape[1],
                             int(np.sqrt(shape[2])), int(np.sqrt(shape[2])),
                             channels])

    def flatten_ssm_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 4  # only valid for 4D tensors
        return (input_shape[0], input_shape[1],
                int(np.sqrt(input_shape[2])), int(np.sqrt(input_shape[2])),
                channels)

    return flatten_ssm, flatten_ssm_output_shape


#: single-channel SSI -- used by the two light-weight-CNN models
flattenSSM, flattenSSM_output_shape = make_flatten_ssm(channels=1)

#: 3-channel SSI -- used by the pretrained C3D model
flattenSSM3, flattenSSM3_output_shape = make_flatten_ssm(channels=3)


## ---------------------------------------------------------------------------
## flattening convolutional feature maps for the LSTM / fusion stages
## ---------------------------------------------------------------------------

def flattenConv(x):
    shape = K.shape(x)
    dim = K.prod(shape[2:])
    return K.reshape(x, [shape[0], shape[1], dim])


def flattenConv_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4 or len(shape) == 5  # only valid for 4D or 5D tensors
    dim = np.prod(shape[2:])
    return (input_shape[0], input_shape[1], dim)
