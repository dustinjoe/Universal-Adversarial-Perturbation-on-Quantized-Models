#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 08:21:45 2020

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

#from tensorflow.keras.utils import get_file
from tensorflow.keras.models import Model, load_model, model_from_json

from quantlib.quantization_ops import (WeightFloat, WeightQuantizer,
                               TrainableWeightQuantizer)
from quantlib.quantization_layers import (QuantizedConv2D, QuantizedDepthwiseConv2D,
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

# Retrieve the float model with pretrained weights and load it
#model_keras = load_quantized_model('models/ds_cnn_cifar10.h5')
#model_keras = load_model('models/resnet_v1.h5')
model_keras = load_quantized_model('../akida/vgg/vgg_cifar10.h5')
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
def check_model_performances(model, x_test,y_test, num_images=1000):
    start = timer()
    potentials_keras = model.predict(x_test[:num_images])
    preds_keras = np.squeeze(np.argmax(potentials_keras, 1))

    accuracy = accuracy_score(y_test[:num_images], preds_keras)
    print("Accuracy: " + "{0:.2f}".format(100 * accuracy) + "%")
    end = timer()
    print(f'Keras inference on {num_images} images took {end-start:.2f} s.\n')
    return accuracy

check_model_performances(model_keras, x_test,y_test)


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

from quantlib import quantize

#%% Quantize the model to 8-bit weights and activations
print('8-bit Quantization:')
model_quantized88 = quantize(model_keras, 8, 8)

# Check Model performance
check_model_performances(model_quantized88, x_test,y_test)


#%% 
def tunedQuantize(model_keras,x_train, y_train,x_test,y_test,Wb=8,Ab=8,tuneIter=10):
    print('Quantization with '+str(Wb)+' Bits Weights+ '+str(Ab)+' Bits Activations.')
    model_quantized = quantize(model_keras, Wb, Ab)
    model_filename = 'vgg_w'+str(Wb)+'a'+str(Ab)+'.h5'
    model_quantized.save(model_filename)
    
    # Check Model performance
    acc_best = check_model_performances(model_quantized, x_test,y_test)
    
    
    if tuneIter>0:    
        for i in range(tuneIter):
            model_quantized.compile(
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer='adam',
                metrics=['accuracy'])
            model_quantized.fit(x_train, y_train, epochs=1, validation_split=0.1)
            model_quantized = quantize(model_quantized, Wb, Ab)
            print('Test accuracy after'+str(i)+'Round tuning:')
            acc = check_model_performances(model_quantized, x_test,y_test)
            if acc>acc_best:
                acc_best = acc
                model_quantized.save(model_filename)
    print('Best Acc After Tuning:', acc_best)
    return model_quantized
'''
Wb = 4
Ab = 4
tuneIter = 30
model_quantized11 =tunedQuantize(model_keras,x_train, y_train,x_test,y_test,Wb,Ab,tuneIter)
'''
tuneIter = 30
qtModelList = [ [2,2],[3,2],[3,3],[4,4],[8,8] ]
for modelCFG in qtModelList:
    Wb = modelCFG[0]
    Ab = modelCFG[1]
    modelname = 'vgg_w'+str(Wb)+'a'+str(Ab)+'.h5'
    model_quantized11 =tunedQuantize(model_keras,x_train, y_train,x_test,y_test,Wb,Ab,tuneIter)

