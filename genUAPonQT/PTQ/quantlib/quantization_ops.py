#!/usr/bin/env python
# ******************************************************************************
# Copyright 2019 Brainchip Holdings Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

# -*- coding: utf-8 -*-
from __future__ import absolute_import
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Layer


class BaseWeightQuantizer(Layer):
    """The base class for all Quantizers.

    All quantizers derived from this class are supposed to be symmetric uniform
    quantizers.

    This base class must be overloaded as well as the two functions `quantize`
    and `scale_factor`.

    The scale factor is the value used to obtain an integer from a float
    quantized weight.

    In other words, given a set of float weights:

     quantize(w) * scale_factor(w) is a set of integer weights.

    Note that since all these values are still expressed as float, a rounding
    operation will always be required to obtain actual integer values.

    """

    def __init__(self, bitwidth):
        """Creates a Weights quantizer for the specified bitwidth.

        Args:
            bitwidth (integer): the quantizer bitwidth defining the number of
                quantized values.

        """
        self.bitwidth_ = int(bitwidth)
        super().__init__()

    def quantize(self, w):
        """Quantizes the specified weights Tensor.

        Args:
            w (:obj:`tensorflow.Tensor`): the weights Tensor to quantize.

        Returns:
            :obj:`tensorflow.Tensor`: a Tensor of quantized weights.

        """
        raise NotImplementedError()

    def scale_factor(self, w):
        """Evaluates the scale factor for the specified weights Tensor.

        Args:
          w (:obj:`tensorflow.Tensor`): the weights Tensor to quantize.

        Returns:
          :obj:`tensorflow.Tensor`: a Tensor containing a single scalar value.

        """
        raise NotImplementedError()

    @property
    def bitwidth(self):
        return self.bitwidth_

    def get_config(self):
        return {'bitwidth': self.bitwidth_}


class WeightQuantizer(BaseWeightQuantizer):
    """A uniform quantizer.

    Quantizes the specified weights into 2^bitwidth-1 values centered on zero.
    E.g. with bitwidth = 4, 15 quantization levels: from -7 * qstep to 7 * qstep
    with qstep being the quantization step. The quantization step is defined by:

     qstep = threshold * std(W) / max_value

    with max_value being 2^(bitwidth-1) - 1. E.g with bitwidth = 4, max_value = 7.

    All values below or above threshold * std(W) are automatically assigned to
    the min (resp max) value.

    """

    def __init__(self, threshold=3, bitwidth=4):
        """Creates a Weights quantizer for the specified bitwidth.

        Args:
            threshold (integer): the standard deviation multiplier used to exclude
                outliers.
            bitwidth (integer): the quantizer bitwidth defining the number of
                quantized values.

        """
        # Having a cast guarantees a check when the parameters are not numbers
        # (e.g.: None)
        self.threshold_ = float(threshold)
        self.kmax_ = (2.**(bitwidth - 1) - 1)
        # Initialize parent to store the bitwidth
        super().__init__(bitwidth)

    def sigma_scaled_(self, w):
        return K.std(w) * self.threshold_

    def scale_factor(self, w):
        return self.kmax_ / self.sigma_scaled_(w)

    def quantize(self, w):
        delta = self.scale_factor(w)
        return K.clip(round_through(w * delta), -self.kmax_, self.kmax_) / delta

    def get_config(self):
        config = super().get_config()
        config.update({'threshold': self.threshold_})
        return config


class TrainableWeightQuantizer(BaseWeightQuantizer):
    """A trainable weight quantizer.

    Quantizes the specified weights into 2^bitwidth-1 values centered on zero.
    E.g. with bitwidth = 4, 15 quantization levels: from -7 * qstep to 7 * qstep
    with qstep being the quantization step. The quantization step is defined by:

     qstep = threshold * std(W) / max_value

    with:

     - max_value being 2^(bitwidth-1) - 1. E.g with bitwidth = 4, max_value = 7.
     - threshold a trainable parameter whose initial value can be specified.

    All values below or above threshold * std(W) are automatically assigned to
    the min (resp max) value.

    This is the trainable version of the WeightQuantizer class.

    """

    def __init__(self, threshold=3, bitwidth=4, **kwargs):
        """Creates a trainable weights quantizer for the specified bitwidth.

        Args:
            threshold (integer): the initial value of the standard deviation
                multiplier used to exclude outliers.
            bitwidth (integer): the quantizer bitwidth defining the number of
                quantized values.

        """
        super().__init__(bitwidth, **kwargs)
        self.threshold_ = tf.Variable(name=f"{self.name}/threshold",
                                      initial_value=float(threshold),
                                      dtype=tf.float32)
        self.kmax_ = (2.**(bitwidth - 1) - 1)

    def sigma_scaled_(self, w):
        return tf.math.reduce_std(w) * self.threshold_

    def scale_factor(self, w):
        return self.kmax_ / self.sigma_scaled_(w)

    def quantize(self, w):
        delta = self.scale_factor(w)
        return tf.clip_by_value(round_through(w * delta), -self.kmax_,
                                self.kmax_) / delta

    def get_config(self):
        config = super().get_config()
        config.update({'threshold': self.threshold_.numpy()})
        return config


class MaxQuantizer(BaseWeightQuantizer):
    """A quantizer that relies on maximum range.

    Quantizes the specified weights into 2^bitwidth-1 values centered on zero.
    E.g. with bitwidth = 4, 15 quantization levels: from -7 * qstep to 7 * qstep
    with qstep being the quantization step. The quantization step is defined by:

     qstep = max_range / max_value

    with:

     - max_range = max(abs(W))
     - max_value = 2^(bitwidth-1) - 1. E.g with bitwidth = 4, max_value = 7.

    """

    def __init__(self, bitwidth=4):
        """Creates a Max quantizer for the specified bitwidth.

        Args:
            bitwidth (integer): the quantizer bitwidth defining the number of
                quantized values.

        """
        self.kmax_ = (2.**(bitwidth - 1) - 1)
        # Initialize parent to store the bitwidth
        super().__init__(bitwidth)

    def max_range_(self, w):
        return tf.math.reduce_max(tf.math.abs(w))

    def scale_factor(self, w):
        return self.kmax_ / self.max_range_(w)

    def quantize(self, w):
        delta = self.scale_factor(w)
        return K.clip(round_through(w * delta), -self.kmax_, self.kmax_) / delta

    def get_config(self):
        config = super().get_config()
        return config


class MaxPerAxisQuantizer(MaxQuantizer):
    """A quantizer that relies on maximum range per axis.

    Quantizes the specified weights into 2^bitwidth-1 values centered on zero.
    E.g. with bitwidth = 4, 15 quantization levels: from -7 * qstep to 7 * qstep
    with qstep being the quantization step. The quantization step is defined by:

     qstep = max_range / max_value

    with:

     - max_range = max(abs(W))
     - max_value = 2^(bitwidth-1) - 1. E.g with bitwidth = 4, max_value = 7.

    This is an evolution of the MaxQuantizer that defines the max_range per
    axis.

    The last dimension is used as axis, meaning that the scaling factor is a
    vector with as many values as "filters", or "neurons".

    Note: for a DepthwiseConv2D layer that has a single filter, this
    quantizer is strictly equivalent to the MaxQuantizer.

    """

    def max_range_(self, w):
        red_range = tf.range(tf.rank(w) - 1)
        return tf.math.reduce_max(tf.math.abs(w), red_range)


class WeightFloat(BaseWeightQuantizer):
    """This quantizer actually does not perform any quantization, and it might
    be used for training.

    """

    def __init__(self):
        super().__init__(bitwidth=0)

    def scale_factor(self, w):
        return tf.constant(1.0)

    def quantize(self, w):
        return w

    def get_config(self):
        return {}


def get(identifier):
    """Returns the weight quantizer corresponding to the identifier.

    The 'identifier' input can take two types: either a weight quantizer
    instance, or a dictionary corresponding to the config serialization
    of a weight quantizer.

    Args:
        identifier (BaseWeightQuantizer or dict): either a BaseWeightQuantizer
            instance or a configuration dictionary to deserialize.

    Returns:
        :obj:`cnn2snn.BaseWeightQuantizer`: a weight quantizer
    """
    if isinstance(identifier, BaseWeightQuantizer):
        return identifier
    elif isinstance(identifier, dict):
        return tf.keras.utils.deserialize_keras_object(
            identifier,
            custom_objects={
                'WeightFloat': WeightFloat,
                'WeightQuantizer': WeightQuantizer,
                'TrainableWeightQuantizer': TrainableWeightQuantizer,
                'MaxQuantizer': MaxQuantizer,
                'MaxPerAxisQuantizer': MaxPerAxisQuantizer
            })
    else:
        raise ValueError(f"Could not interpret identifier {identifier} "
                         "for a weight quantizer object")


def round_through(x):
    """Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182).

    """
    rounded = K.round(x)
    return x + K.stop_gradient(rounded - x)


def ceil_through(x):
    """Element-wise ceiling operation (to the closest greater integer) with
    full gradient propagation.

    """
    ceiling_value = tf.math.ceil(x)
    return x + K.stop_gradient(ceiling_value - x)
