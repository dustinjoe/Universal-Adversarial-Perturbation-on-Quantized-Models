from tensorflow.keras import Model, Input
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import (InputLayer, Conv2D, SeparableConv2D, Dense,
                                     ReLU, Layer)

from .quantization_ops import MaxQuantizer, MaxPerAxisQuantizer, WeightFloat
from .quantization_layers import (QuantizedConv2D, QuantizedSeparableConv2D,
                                  QuantizedDense, ActivationDiscreteRelu)
from .utils import invert_batchnorm_pooling, fold_batch_norms, chain_cloned_layer


def quantize(model,
             weight_quantization=0,
             activ_quantization=0,
             input_weight_quantization=None):
    """Converts a standard sequential Keras model to a CNN2SNN Keras quantized
    model, compatible for Akida conversion.

    This function returns a Keras model where the standard neural layers
    (Conv2D, SeparableConv2D, Dense) and the ReLU activations are replaced with
    CNN2SNN quantized layers (QuantizedConv2D, QuantizedSeparableConv2D,
    QuantizedDense, ActivationDiscreteRelu).

    Several transformations are applied to the model:
    - the order of MaxPool and BatchNormalization layers are inverted so that
    BatchNormalization always happens first,
    - the batch normalization layers are folded into the previous layers.

    This new model can be either directly converted to akida, or first
    retrained for a few epochs to recover any accuracy loss.

    Args:
        model (tf.keras.Model): a standard Keras model
        weight_quantization (int): sets all weights in the model to have
            a particular quantization bitwidth except for the weights in the
            first layer.

            * '0' implements floating point 32-bit weights.
            * '2' through '8' implements n-bit weights where n is from 2-8 bits.
        activ_quantization (int): sets all activations in the model to have a
            particular activation quantization bitwidth.

            * '0' implements floating point 32-bit activations.
            * '1' through '8' implements n-bit weights where n is from 1-8 bits.
        input_weight_quantization (int): sets weight quantization in the first
            layer. Defaults to weight_quantization value.

            * 'None' implements the same bitwidth as the other weights.
            * '0' implements floating point 32-bit weights.
            * '2' through '8' implements n-bit weights where n is from 2-8 bits.

    Returns:
        tf.keras.Model: a quantized Keras model
    """

    # Overrides input weight quantization if None
    if input_weight_quantization is None:
        input_weight_quantization = weight_quantization

    # Apply Model transformations, obtaining strictly equivalent models at
    # inference time.

    # Invert batch normalization and pooling
    model_t = invert_batchnorm_pooling(model)
    # Fold batch norm layers with corresponding neural layers
    model_t = fold_batch_norms(model_t)

    # Convert neural layers and ReLU to CNN2SNN quantized layers
    first_layer = 0
    if isinstance(model_t.layers[0], InputLayer):
        first_layer = 1

    inp = Input(model_t.layers[first_layer].input_shape[1:])
    x = inp
    first_neural_layer = True
    for layer in model_t.layers[first_layer:]:
        if isinstance(layer, (ReLU, ActivationDiscreteRelu)):
            if activ_quantization > 0:
                x = ActivationDiscreteRelu(activ_quantization,
                                           name=layer.name)(x)
            else:
                x = chain_cloned_layer(x, layer)
        elif isinstance(layer, (Conv2D, SeparableConv2D, Dense)):
            if first_neural_layer:
                bitwidth = input_weight_quantization
                first_neural_layer = False
            else:
                bitwidth = weight_quantization
            layer_q = _convert_to_quantized_layer(layer, bitwidth)
            x = layer_q(x)
            layer_q.set_weights(layer.get_weights())

        else:
            x = chain_cloned_layer(x, layer)

    return Model(inp, x)


def quantize_layer(model, target_layer, bitwidth):
    """Converts a specific layer to a quantized version with the given bitwidth.

    This function returns a Keras model where the target layer is converted to
    an equivalent quantized layer. All other layers are preserved.

    Args:
        model (tf.keras.Model): a standard Keras model
        target_layer: a standard or quantized Keras layer to be
            converted, or the index or name of the target layer.
        bitwidth (int): the desired quantization bitwidth. If zero, no
            quantization will be applied.

    Returns:
        tf.keras.Model: a quantized Keras model
    """

    if isinstance(target_layer, int):
        layer_to_quantize = model.layers[target_layer]
    elif isinstance(target_layer, str):
        layer_to_quantize = model.get_layer(target_layer)
    elif isinstance(target_layer, Layer):
        layer_to_quantize = target_layer
    else:
        raise ValueError(f"Target layer argument is not recognized")

    def replace_layer(layer):
        if layer == layer_to_quantize:
            if isinstance(layer, (ReLU, ActivationDiscreteRelu)):
                if bitwidth > 0:
                    return ActivationDiscreteRelu(bitwidth, name=layer.name)
                else:
                    return ReLU(name=layer.name)
            elif isinstance(layer, (Conv2D, SeparableConv2D, Dense)):
                return _convert_to_quantized_layer(layer, bitwidth)
            else:
                return layer
        else:
            return layer

    new_model = clone_model(model, clone_function=replace_layer)
    new_model.set_weights(model.get_weights())

    return new_model


def _convert_to_quantized_layer(layer, bitwidth):
    """Converts a standard Keras layer (Conv2D, SeparableConv2D, Dense) to
    a CNN2SNN quantized Keras layer.

    This function returns a quantized Keras layer (QuantizedConv2D,
    QuantizedSeparableConv2D or QuantizedDense layer) with a weight quantizer
    using the bitwidth value to quantize weights. The original weights are
    loaded in the new quantized layer. If bitwidth is zero, a WeightFloat
    quantizer is used (no quantization is performed).

    Args:
        layer (tf.keras.Layer): a standard Keras layer (Conv2D, SeparableConv2D,
            or Dense)
        bitwidth (int): the desired weight quantization bitwidth. If zero, no
            quantization will be applied.

    Returns:
        :obj:`tensorflow.keras.Layer`: a CNN2SNN quantized Keras layer
    """

    config = layer.get_config()
    if bitwidth > 0:
        config['quantizer'] = MaxPerAxisQuantizer(bitwidth=bitwidth)
    else:
        config['quantizer'] = WeightFloat()

    if isinstance(layer, Conv2D):
        for arg in QuantizedConv2D.unsupported_args:
            if arg in config:
                del config[arg]
        return QuantizedConv2D.from_config(config)
    elif isinstance(layer, SeparableConv2D):
        if bitwidth > 0:
            config['quantizer_dw'] = MaxQuantizer(bitwidth=bitwidth)
        for arg in QuantizedSeparableConv2D.unsupported_args:
            if arg in config:
                del config[arg]
        return QuantizedSeparableConv2D.from_config(config)
    elif isinstance(layer, Dense):
        for arg in QuantizedDense.unsupported_args:
            if arg in config:
                del config[arg]
        return QuantizedDense.from_config(config)
    else:
        return None
