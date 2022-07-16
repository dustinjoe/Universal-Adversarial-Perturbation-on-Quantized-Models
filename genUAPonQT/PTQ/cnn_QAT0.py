#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow import keras

from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

import numpy as np
import random

from art.attacks.evasion import FastGradientMethod,ProjectedGradientDescent
from art.estimators.classification import KerasClassifier
from art.utils import load_cifar10

from tensorflow.keras.datasets import cifar10

# Step 1: Load the  dataset
# Load CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Reshape x-data
x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)
input_shape = (32, 32, 3)

# Set aside raw test data for use with Akida Execution Engine later
raw_x_test = x_test.astype('uint8')

# Rescale x-data
a = 255.0
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
#modelname = 'models/ds_cnn_cifar10.h5'
#modelname = 'models/model_w4a4.h5'
#print(modelname)
#model = load_quantized_model(modelname)

qtModelList = [ [8,8],[4,4],[3,3],[3,2],[2,2] ]
qatarchList = []
path = 'models/'
for modelCFG in qtModelList:
    wb = modelCFG[0]
    ab = modelCFG[1]
    modelname = 'model_w'+str(wb)+'a'+str(ab)+'.h5'
    modelarchname = 'modelArch_w'+str(wb)+'a'+str(ab)+'.json'
    model0 = load_quantized_model(path+modelname)        
    model0.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy'])
    json_config = model0.to_json()
    qatarchList.append(json_config)
    '''
    with open(path+modelarchname, "w") as json_file:
        json_file.write(json_config)
    '''


#%% Step 5: Generate adversarial test examples
from quantlib import quantize

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

#check_model_performances(model_keras, x_test,y_test)
qatidx = 0
nb_epoch = 40
acc_best = 0.0
opt = tf.keras.optimizers.Adam(lr=0.001, decay=0.0001)
for modelCFG in qtModelList:
    wb = modelCFG[0]
    ab = modelCFG[1]
    model_filename = 'qat_w'+str(wb)+'a'+str(ab)+'.h5'
    
    
    # load json and create model
    modelarchname = 'modelArch_w'+str(wb)+'a'+str(ab)+'.json'

    model_qat = model_from_json(qatarchList[qatidx],cnn2snn_objects)
    qatidx = qatidx+1
    
    for i in range(nb_epoch):
        model_qat.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=opt,
            metrics=['accuracy'])
        model_qat.fit(x_train, y_train, epochs=5, validation_split=0.1)
        model_qat = quantize(model_qat, wb, ab)
        print('Test accuracy after'+str(i)+'Round QAT:')
        acc = check_model_performances(model_qat, x_test,y_test)
        if acc>acc_best:
            acc_best = acc
            model_qat.save(path+model_filename)
    print(model_filename)
    print('Best Acc After QAT:', acc_best)
    
    
    