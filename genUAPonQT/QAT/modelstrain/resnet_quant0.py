#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 20:17:41 2020

@author: xyzhou
"""
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import keras

from keras.models import Sequential,load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np

from art.attacks.evasion import FastGradientMethod,ProjectedGradientDescent
from art.estimators.classification import KerasClassifier
from art.utils import load_cifar10

from keras.datasets import cifar10
from keras.utils import np_utils
# Step 1: Load the  dataset

(x_train, y_train), (x_test, y_test)  = cifar10.load_data()
# normalize inputs from 0-255 to 0.0-1.0
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
min_pixel_value = 0.0
max_pixel_value = 1.0

x_test = x_test[:1000]
y_test = y_test[:1000]


modelname = 'resnet_v1.h5'
#modelname = 'multi7convConcat1.h5'
model = load_model(modelname)
print(modelname)


config = {
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

#%%
from keras.optimizers import Adam
from qkeras import *

qmodel.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(0.001),
    metrics=["accuracy"])
scores = qmodel.evaluate(x_test,
                        y_test,
                        batch_size=batch_size,
                        verbose=0)
print('Current Test loss:', scores[0])
print('Current Test accuracy:', scores[1])

print_qstats(qmodel)

#%%
print('Fine Tuning:')
qmodel.fit(x_train, y_train, epochs=50, batch_size=128, validation_data=(x_test, y_test), verbose=True)

scores = qmodel.evaluate(x_test,
                        y_test,
                        batch_size=batch_size,
                        verbose=0)
print('Current Test loss:', scores[0])
print('Current Test accuracy:', scores[1])

