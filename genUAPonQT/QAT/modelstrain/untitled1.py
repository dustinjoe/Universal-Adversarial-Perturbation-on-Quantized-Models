#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 18:36:53 2020

@author: xyzhou
"""
from keras.layers import *
from qkeras import *

x = x_in = Input(shape)
x = QConv2D(18, (3, 3),
        kernel_quantizer="stochastic_ternary",
        bias_quantizer="ternary", name="first_conv2d")(x)
x = QActivation("quantized_relu(3)")(x)
x = QSeparableConv2D(32, (3, 3),
        depthwise_quantizer=quantized_bits(4, 0, 1),
        pointwise_quantizer=quantized_bits(3, 0, 1),
        bias_quantizer=quantized_bits(3),
        depthwise_activation=quantized_tanh(6, 2, 1))(x)
x = QActivation("quantized_relu(3)")(x)
x = Flatten()(x)
x = QDense(NB_CLASSES,
        kernel_quantizer=quantized_bits(3),
        bias_quantizer=quantized_bits(3))(x)
x = QActivation("quantized_bits(20, 5)")(x)
x = Activation("softmax")(x)


config = {
  "conv2d_1": {
      "kernel_quantizer": "stochastic_binary",
      "bias_quantizer": "quantized_po2(4)"
  },
  "QConv2D": {
      "kernel_quantizer": "stochastic_ternary",
      "bias_quantizer": "quantized_po2(4)"
  },
  "QDense": {
      "kernel_quantizer": "quantized_bits(4,0,1)",
      "bias_quantizer": "quantized_bits(4)"
  },
  "QActivation": { "relu": "binary" },
  "act_2": "quantized_relu(3)",
}

from qkeras.utils import model_quantize

qmodel = model_quantize(model, config, 4, transfer_weights=True)

for layer in qmodel.layers:
    if hasattr(layer, "kernel_quantizer"):
        print(layer.name, "kernel:", str(layer.kernel_quantizer_internal), "bias:", str(layer.bias_quantizer_internal))
    elif hasattr(layer, "quantizer"):
        print(layer.name, "quantizer:", str(layer.quantizer))

print()
qmodel.summary()