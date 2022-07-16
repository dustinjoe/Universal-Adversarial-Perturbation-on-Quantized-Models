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
"""
A set of functions to convert a Keras (tf.keras) model to a new
equivalent model with different characteristics. Then, the new model
can be quantized.

"""
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import (InputLayer, Conv2D, SeparableConv2D, Dense,
                                     MaxPool2D, GlobalAveragePooling2D,
                                     BatchNormalization)
from tensorflow.keras.models import Model, load_model, model_from_json
from .quantization_ops import (WeightFloat, WeightQuantizer,
                               TrainableWeightQuantizer)
from .quantization_layers import (QuantizedConv2D, QuantizedDepthwiseConv2D,
                                  QuantizedSeparableConv2D, QuantizedDense,
                                  ActivationDiscreteRelu, QuantizedReLU)

cnn2snn_objects = {
    'WeightFloat': WeightFloat,
    'WeightQuantizer': WeightQuantizer,
    'TrainableWeightQuantizer': TrainableWeightQuantizer,
    'QuantizedConv2D': QuantizedConv2D,
    'QuantizedSeparableConv2D': QuantizedSeparableConv2D,
    'QuantizedDense': QuantizedDense,
    'ActivationDiscreteRelu': ActivationDiscreteRelu,
    'QuantizedReLU': QuantizedReLU
}


def invert_batchnorm_pooling(model):
    """Returns a new model where pooling and batchnorm layers are inverted.
    From a Keras model where pooling layers precede batch normalization
    layers, this function places the BN layers before pooling layers. This
    is the first step before folding BN layers into neural layers.
    Note: inversion of layers is equivalent only if the gammas of BN layers
    are positive. The function raises an error if not.
    Args:
        model (:obj:`tf.keras.Model`): a tf.keras model.
    Returns:
        :obj:`tf.keras.Model`: a keras.Model.
    """

    first_layer = 0
    if isinstance(model.layers[0], InputLayer):
        first_layer = 1

    inp = Input(model.layers[first_layer].input_shape[1:])
    x = inp
    i = first_layer
    while i < len(model.layers) - 1:
        layer = model.layers[i]
        next_layer = model.layers[i + 1]
        supported_pool = (MaxPool2D, GlobalAveragePooling2D)
        is_pool = isinstance(layer, supported_pool)
        if is_pool and isinstance(next_layer, BatchNormalization):
            gammas = next_layer.get_weights()[0]
            if np.any(gammas <= 0):
                # NB: negative gammas are only a problem for max pooling, not
                # for avg pooling
                raise RuntimeError(f"There are {np.sum(gammas <= 0)} negative "
                                   "gammas in the batch norm layer "
                                   f"{next_layer.name}. Negative gammas are "
                                   "not supported.")
            next_layer_config = next_layer.get_config()
            # GlobalAveragePooling2D brings a change on axis for the batch norm.
            if isinstance(layer, GlobalAveragePooling2D):
                next_layer_config['axis'] = [-1]
            bn_layer = BatchNormalization.from_config(next_layer_config)
            x = bn_layer(x)
            x = chain_cloned_layer(x, layer)
            bn_layer.set_weights(next_layer.get_weights())
            i = i + 2
        else:
            x = chain_cloned_layer(x, layer)
            i = i + 1

    x = model.layers[-1](x)
    return Model(inp, x, name=model.name)


def fold_batch_norms(model):
    """Returns a new model where batchnorm layers are folded into
    previous neural layers.
    From a Keras model where BN layers follow neural layers, this
    function removes the BN layers and updates weights and bias
    accordingly of the preceding neural layers. The new model is
    strictly equivalent to the previous one.
    Args:
        model (:obj:`tf.keras.Model`): a Keras model.
    Returns:
        :obj:`tf.keras.Model`: a tf.keras.Model.
    """
    first_layer = 0
    if isinstance(model.layers[0], InputLayer):
        first_layer = 1

    inp = Input(model.layers[first_layer].input_shape[1:])
    x = inp

    i = first_layer
    supported_layers = (Conv2D, SeparableConv2D, Dense, QuantizedConv2D,
                        QuantizedSeparableConv2D, QuantizedDense)
    while i < len(model.layers) - 1:
        layer = model.layers[i]
        next_layer = model.layers[i + 1]

        if isinstance(next_layer, BatchNormalization):
            if not isinstance(layer, supported_layers):
                raise AttributeError(
                    "The layer preceding a batch norm layer must be "
                    "(Quantized)Conv2D, (Quantized)SeparableConv2D or "
                    "(Quantized)SeparableConv2D.")

            # Get weights and BN parameters
            gamma, beta, mean, var = next_layer.get_weights()
            epsilon = next_layer.epsilon
            if isinstance(layer, (SeparableConv2D, QuantizedSeparableConv2D)):
                weights = layer.get_weights()[1]
                bias = layer.get_weights()[2] if len(
                    layer.get_weights()) > 2 else 0
            else:
                weights = layer.get_weights()[0]
                bias = layer.get_weights()[1] if len(
                    layer.get_weights()) > 1 else 0

            # Compute new weights for folded layer
            scale_BN = gamma / np.sqrt(var + epsilon)
            new_weights = []
            if isinstance(layer, (SeparableConv2D, QuantizedSeparableConv2D)):
                new_weights.append(layer.get_weights()[0])
            new_weights.append(weights * scale_BN)
            new_weights.append(beta + (bias - mean) * scale_BN)

            # Create new layer
            config = layer.get_config()
            config['use_bias'] = True
            if isinstance(layer, QuantizedSeparableConv2D):
                new_layer = QuantizedSeparableConv2D.from_config(config)
            elif isinstance(layer, SeparableConv2D):
                new_layer = SeparableConv2D.from_config(config)
            elif isinstance(layer, QuantizedConv2D):
                new_layer = QuantizedConv2D.from_config(config)
            elif isinstance(layer, Conv2D):
                new_layer = Conv2D.from_config(config)
            elif isinstance(layer, QuantizedDense):
                new_layer = QuantizedDense.from_config(config)
            elif isinstance(layer, Dense):
                new_layer = Dense.from_config(config)

            x = new_layer(x)
            new_layer.set_weights(new_weights)
            i = i + 2

        else:
            x = chain_cloned_layer(x, layer)
            i = i + 1

    if not isinstance(model.layers[-1], BatchNormalization):
        x = chain_cloned_layer(x, model.layers[-1])

    return Model(inp, x, name=model.name)


def merge_separable_conv(model):
    """Returns a new model where all depthwise conv2d layers followed by conv2d
    layers are merged into single separable conv layers.

    The new model is strictly equivalent to the previous one.

    Args:
        model (:obj:`tf.keras.Model`): a Keras model.

    Returns:
        :obj:`tf.keras.Model`: a tf.keras.Model.

    """
    # If no layers are Depthwise, there is nothing to be done, return.
    if not any([isinstance(l, QuantizedDepthwiseConv2D) for l in model.layers]):
        return model

    if isinstance(model.layers[0], InputLayer):
        x = model.layers[0].output
        i = 1
    else:
        x = model.layers[0].input
        i = 0
    while i < len(model.layers) - 1:
        layer = model.layers[i]
        next_layer = model.layers[i + 1]

        if isinstance(layer, QuantizedDepthwiseConv2D):
            # Check layers expected order
            if not isinstance(next_layer, QuantizedConv2D):
                raise AttributeError(f"Layer {layer.name} "
                                     "QuantizedDepthwiseConv2D should be "
                                     "followed by QuantizedConv2D layers.")

            if layer.bias is not None:
                raise AttributeError(f"Unsupported bias in "
                                     "QuantizedDepthwiseConv2D Layer "
                                     "{layer.name} ")

            # Get weights and prepare new ones
            dw_weights = layer.get_weights()[0]
            pw_weights = next_layer.get_weights()[0]
            new_weights = [dw_weights, pw_weights]
            if next_layer.use_bias:
                bias = next_layer.get_weights()[1]
                new_weights.append(bias)

            # Create new layer
            new_name = f'{layer.name}_{next_layer.name}'
            new_layer = QuantizedSeparableConv2D(next_layer.filters,
                                                 layer.kernel_size,
                                                 quantizer_dw=layer.quantizer,
                                                 quantizer=layer.quantizer,
                                                 padding=layer.padding,
                                                 use_bias=next_layer.use_bias,
                                                 name=new_name)
            x = new_layer(x)
            new_layer.set_weights(new_weights)
            i = i + 2

        else:
            x = chain_cloned_layer(x, layer)
            i = i + 1

    # Add last layer if not done already
    if i == (len(model.layers) - 1):
        if isinstance(model.layers[-1], QuantizedDepthwiseConv2D):
            raise AttributeError(f"Layer {layer.name} "
                                 "QuantizedDepthwiseConv2D should be followed "
                                 "by QuantizedConv2D layers.")
        x = model.layers[-1](x)

    return Model(inputs=model.input, outputs=[x], name=model.name)


def load_quantized_model(filepath, compile=True):
    """Loads a quantized model saved in TF or HDF5 format.

    If the model was compiled and trained before saving, its training state
    will be loaded as well.
    This function is a wrapper of `tf.keras.models.load_model`.

    Args:
        filepath (string): path to the saved model.
        compile (bool): whether to compile the model after loading.

    Returns:
        :obj:`tensorflow.keras.Model`: a Keras model instance.
    """
    return load_model(filepath, cnn2snn_objects, compile)


def load_partial_weights(dest_model, src_model):
    """Loads a subset of weights from one Keras model to another

    This goes through each layers of the source model, looking for a matching
    layer in the destination model.
    If a layer with the same name is found, then this method assumes that one
    of the two layer has the same set of weights as the other plus some extra
    weights at the end, and loads only the first common weights into the
    destination layer.

    Args:
        dest_model(:obj:`tensorflow.keras.Model`): the destination Model
        src_model(:obj:`tensorflow.keras.Model`): the source Model

    """
    for src_layer in src_model.layers:
        src_weights = src_layer.get_weights()
        dest_layer = dest_model.get_layer(src_layer.name)
        dest_weights = dest_layer.get_weights()
        # Take the minimum of the two lists of weights
        n_weights = min(len(src_weights), len(dest_weights))
        # Only replace the first weights
        dest_weights[0:n_weights] = src_weights[0:n_weights]
        dest_layer.set_weights(dest_weights)


def create_trainable_quantizer_model(quantized_model):
    """Converts a legacy quantized model to a model using trainable quantizers.

    Legacy cnn2snn models have fixed quantization schemes. This method converts
    such a model to an equivalent model using trainable quantizers.

    Args:
        quantized_model(str, :obj:`tensorflow.keras.Model`): a keras Model or a
        path to a keras Model file

    Returns:
        :obj:`tensorflow.keras.Model`: a Keras Model instance.
    """
    if isinstance(quantized_model, str):
        # Load the model at the specified path
        quantized_model = load_quantized_model(quantized_model)
    # Dump model configuration in a JSON string
    json_string = quantized_model.to_json()
    # Edit the model configuration to replace static quantizers by trainable
    # ones
    json_string = json_string.replace("WeightQuantizer",
                                      "TrainableWeightQuantizer")
    json_string = json_string.replace("ActivationDiscreteRelu", "QuantizedReLU")
    # Create a new model from the modified configuration
    new_model = model_from_json(json_string, custom_objects=cnn2snn_objects)
    # Transfer weights to the new model
    load_partial_weights(new_model, quantized_model)
    return new_model


def chain_cloned_layer(x, layer):
    """This function creates a hard copy of a layer and apply to it the
    tf.Tensor x.

    """

    config = layer.get_config()
    w = layer.get_weights()
    new_layer = type(layer).from_config(config)
    out = new_layer(x)
    new_layer.set_weights(w)
    return out