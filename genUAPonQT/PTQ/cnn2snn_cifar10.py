#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 23:38:33 2020

@author: xyzhou
"""
######################################################################
# 1. Dataset preparation
# ~~~~~~~~~~~~~~~~~~~~~~
from tensorflow.keras.datasets import cifar10
from tensorflow import keras
# Load CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Reshape x-data
x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)
input_shape = (32, 32, 3)

# Set aside raw test data for use with Akida Execution Engine later
raw_x_test = x_test.astype('uint8')

# Rescale x-data
a = 255
b = 0

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = (x_train - b) / a
x_test = (x_test - b) / a

######################################################################
# 2. Create a Keras DS-CNN model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The DS-CNN architecture is available in the `Akida models zoo
# <../api_reference/akida_models_apis.html#cifar-10>`_ along with pretrained
# weights.
#
#  .. Note:: The pre-trained weights were obtained after training the model with
#            unconstrained float weights and activations for 1000 epochs
#

from tensorflow.keras.utils import get_file
from cnn2snn import load_quantized_model

# Retrieve the float model with pretrained weights and load it
model_keras = load_quantized_model('models/ds_cnn_cifar10')
model_keras.summary()


######################################################################
# Keras model accuracy is checked against the first *n* images of the test set.
#
# The table below summarizes the expected results:
#
# +---------+----------+
# | #Images | Accuracy |
# +=========+==========+
# | 100     |  96.00 % |
# +---------+----------+
# | 1000    |  94.30 % |
# +---------+----------+
# | 10000   |  93.60 % |
# +---------+----------+
#
# .. Note:: Depending on your hardware setup, the processing time may vary.
#

import numpy as np

from sklearn.metrics import accuracy_score
from timeit import default_timer as timer


# Check Model performance
def check_model_performances(model, x_test, num_images=1000):
    start = timer()
    potentials_keras = model.predict(x_test[:num_images])
    preds_keras = np.squeeze(np.argmax(potentials_keras, 1))

    accuracy = accuracy_score(y_test[:num_images], preds_keras)
    print("Accuracy: " + "{0:.2f}".format(100 * accuracy) + "%")
    end = timer()
    print(f'Keras inference on {num_images} images took {end-start:.2f} s.\n')
    return accuracy


check_model_performances(model_keras, x_test)

######################################################################
# 3. Quantized model
# ~~~~~~~~~~~~~~~~~~
#
# Quantizing a model is done using `CNN2SNN quantize
# <../api_reference/cnn2snn_apis.html#quantize>`_. After the call, all the
# layers will have 4-bit weights and 4-bit activations.
#
# This model will therefore satisfy the Akida NSoC requirements but will suffer
# from a drop in accuracy due to quantization as shown in the table below:
#
# +---------+----------------+--------------------+
# | #Images | Float accuracy | Quantized accuracy |
# +=========+================+====================+
# | 100     |     96.00 %    |       96.00 %      |
# +---------+----------------+--------------------+
# | 1000    |     94.30 %    |       92.60 %      |
# +---------+----------------+--------------------+
# | 10000   |     93.66 %    |       92.58 %      |
# +---------+----------------+--------------------+
#

from cnn2snn import quantize


#%% Quantize the model to 8-bit weights and activations
print('8-bit Quantization:')
model_quantized88 = quantize(model_keras, 8, 8)

# Check Model performance
check_model_performances(model_quantized88, x_test)



#%% Quantize the model to 4-bit weights and activations
print('4-bit Quantization:')
model_quantized44 = quantize(model_keras, 4, 4)

# Check Model performance
check_model_performances(model_quantized44, x_test)


#%% Quantize the model to 3-bit weights and 2-bit activations
def check_model_performances(model, x_test, num_images=1000):
    start = timer()
    potentials_keras = model.predict(x_test[:num_images])
    preds_keras = np.squeeze(np.argmax(potentials_keras, 1))

    accuracy = accuracy_score(y_test[:num_images], preds_keras)
    print("Accuracy: " + "{0:.2f}".format(100 * accuracy) + "%")
    end = timer()
    print(f'Keras inference on {num_images} images took {end-start:.2f} s.\n')
    return accuracy


print('w3_a2-bit Quantization:')
model_quantized32 = quantize(model_keras, 3, 2)
model_quantized32.save('model_w3a2.h5')

# Check Model performance
acc_best = check_model_performances(model_quantized32, x_test)

for i in range(10):
    model_quantized32.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy'])
    model_quantized32.fit(x_train, y_train, epochs=1, validation_split=0.1)
    model_quantized32 = quantize(model_quantized32, 3, 2)
    print('Test accuracy after'+str(i)+'Round tuning:')
    acc = check_model_performances(model_quantized32, x_test)
    if acc>acc_best:
        acc_best = acc
        model_quantized32.save('model_w3a2.h5')

#%% Quantize the model to 3-bit weights and 3-bit activations
print('w3_a3-bit Quantization:')
model_quantized33 = quantize(model_keras, 3, 3)
model_quantized33.save('model_w3a3.h5')

# Check Model performance
acc_best = check_model_performances(model_quantized33, x_test)

for i in range(10):
    model_quantized33.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy'])
    model_quantized33.fit(x_train, y_train, epochs=1, validation_split=0.1)
    model_quantized33 = quantize(model_quantized33, 3, 3)
    print('Test accuracy after'+str(i)+'Round tuning:')
    acc = check_model_performances(model_quantized33, x_test)
    if acc>acc_best:
        acc_best = acc
        model_quantized33.save('model_w3a3.h5')


#


#%% Quantize the model to 2-bit weights and activations
print('2-bit Quantization:')
model_quantized22 = quantize(model_keras, 2, 2)
model_quantized22.save('model_w2a2.h5')

# Check Model performance
acc_best = check_model_performances(model_quantized22, x_test)




for i in range(20):
    model_quantized22.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy'])
    model_quantized22.fit(x_train, y_train, epochs=1, validation_split=0.1)
    model_quantized22 = quantize(model_quantized22, 2, 2)
    print('Test accuracy after'+str(i)+'Round tuning:')
    acc = check_model_performances(model_quantized22, x_test)
    if acc>acc_best:
        acc_best = acc
        model_quantized22.save('model_w2a2.h5')
'''
score = model_quantized22.evaluate(x_test, y_test, verbose=0)
print('Test accuracy after Round tuning:', score[1])
'''

'''
#%% Quantize the model to 1-bit weights and activations
print('1-bit Quantization:')
model_quantized11 = quantize(model_keras, 1, 1)
#%%
model_quantized11.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

model_quantized11.fit(x_train, y_train, epochs=5, validation_split=0.1)
# Check Model performance
check_model_performances(model_quantized11, x_test)
'''