"""Learnable Mahalanobis metric layer used to build Self-Similarity Images (SSIs).

This implements Eq. (1)-(3) of

    Z. Shao, Y. Li, H. Zhang, "Learning Representations From Skeletal
    Self-Similarities for Cross-View Action Recognition",
    IEEE TCSVT, 31(1):160-174, 2021.

The layer receives the pairwise joint differences (p_i - p_j) and applies the
learned linear transform L, so that the Mahalanobis metric M = L'L is obtained
by learning L directly (Eq. 3).  Wrapping this as a custom Keras layer is what
lets the SSI computation live *inside* the network instead of being a
hand-crafted preprocessing step.

Note on the exact quantity produced: `call()` returns the *squared* transformed
norm ||L'(p_i - p_j)||^2, which is then min-max normalised to [0, 1] over each
sample so the SSI can be consumed as an image by the 3D CNN branch.  This is
the behaviour the published experiments were run with; it is kept verbatim.
"""

from keras import backend as K
from keras.layers import Layer
from keras import regularizers


class MetricLayer(Layer):
    """Build a single-channel SSI from pairwise joint differences.

    # Arguments
        output_dim: number of joints N. Kept for API compatibility and
            model deserialisation; the output spatial size is taken from
            the input shape.
        kernel_regularizer: regulariser applied to L. Per Eq. (8) of the
            paper the weights of L take part in the global l2 term ||W||_2,
            so the callers pass the same l2(1e-6) regulariser used elsewhere.

    # Input shape
        4D tensor `(batch, T, N*N, 3)` -- the (p_i - p_j) differences.

    # Output shape
        4D tensor `(batch, T, N*N, 1)`.
    """

    # : number of channels replicated on the output (see MetricLayer_ForC3D)
    channels = 1

    def __init__(self, output_dim, kernel_regularizer=None, **kwargs):
        self.output_dim = output_dim
        super(MetricLayer, self).__init__(**kwargs)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        # L in Eq. (3): a (3, 3) linear transform on the xyz joint offsets.
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='uniform',
            regularizer=self.kernel_regularizer,
            trainable=True,
        )
        super(MetricLayer, self).build(input_shape)

    def call(self, inputs):
        # ||L'(p_i - p_j)||^2  -- Eq. (2)/(3) without the outer square root.
        x = K.sum(K.square(K.dot(inputs, self.kernel)), axis=-1, keepdims=True)

        # Min-max normalise over the whole (T, N*N) volume of each sample so
        # the resulting SSI has a stable dynamic range across sequences.
        x_st = K.reshape(x, [-1, x.shape[1] * x.shape[2], x.shape[-1]])
        x_st_max = K.reshape(K.max(x_st, axis=-2, keepdims=True), [-1, 1, 1, x.shape[-1]])
        x_st_min = K.reshape(K.min(x_st, axis=-2, keepdims=True), [-1, 1, 1, x.shape[-1]])
        x = (x - x_st_min) / (x_st_max - x_st_min + K.epsilon())

        if self.channels > 1:
            x = K.repeat_elements(x, self.channels, axis=-1)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.channels)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        }
        base_config = super(MetricLayer, self).get_config()
        base_config.update(config)
        return base_config


class MetricLayer_ForC3D(MetricLayer):
    """SSI layer for the MSNN_early-C3D model.

    Identical to `MetricLayer` except that the single-channel SSI is replicated
    into 3 channels, because the Sports-1M pretrained C3D backbone used as the
    3D CNN branch expects 3-channel input.
    """

    channels = 3
