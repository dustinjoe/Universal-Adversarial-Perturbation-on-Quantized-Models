#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 16:40:46 2020

@author: xyzhou
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow import keras

from tensorflow.keras.models import load_model

import numpy as np

from art.attacks.evasion import FastGradientMethod,ProjectedGradientDescent
from art.estimators.classification import KerasClassifier
from art.utils import load_cifar10

from keras.datasets import cifar10
from keras.utils import np_utils
# Step 1: Load the  dataset
(_, _), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()

x_test = x_test.astype('float32')#[:1000]
y_test = y_test.astype('float32')#[:1000]

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

qtModelList = [ [2,2],[3,2],[3,3],[4,4],[8,8] ]
#qtModelList = [ [2,2],[4,4],[8,8] ]
def loadModels(path,qtModelList):
    models = []
    modelname = 'ds_cnn_cifar10.h5'
    model0 = load_quantized_model(path+modelname)
    #model0.summary()
    model0.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy'])
    # Step 3: Create the ART classifier
    classifier = KerasClassifier(model=model0, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)
    models.append(classifier)
    
    for modelCFG in qtModelList:
        wb = modelCFG[0]
        ab = modelCFG[1]
        modelname = 'model_w'+str(wb)+'a'+str(ab)+'.h5'
        model0 = load_quantized_model(path+modelname)        
        model0.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer='adam',
            metrics=['accuracy'])
        # Step 3: Create the ART classifier
        classifier = KerasClassifier(model=model0, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)
        models.append(classifier)
    return models
path = 'models/'
classifierSet = loadModels(path,qtModelList)

# Step 4: Evaluate the ART classifier on benign test examples
classifierFP = classifierSet[0]
predictions = classifierFP.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

#%% Step 5: Generate adversarial test examples
#from transfer_uap import TranferUniversalPerturbation
from art.attacks.evasion import UniversalPerturbation

modelidx = 0
N= 500 #len(x_test) 
for epsDeg in [2,4,8]:
    advEps = epsDeg/255
    for classifier in classifierSet:
        if modelidx==0:
            modeltag = 'w32a32'
        else:
            modelCFG = qtModelList[modelidx-1]
            wb = modelCFG[0]
            ab = modelCFG[1]
            modeltag = 'w'+str(wb)+'a'+str(ab)
        print('Working on Model('+str(modelidx)+'):'+modeltag)
        
        
        uap_attack = UniversalPerturbation(classifier=classifier, attacker='deepfool',eps=advEps, max_iter=20)
        
        x_test_uap0 = uap_attack.generate(x=x_test[:N], y=y_test[:N])
        x_test_uap0 = x_test_uap0.astype('float32')
        
        datname = 'x_uapraw'+str(N)+'_'+modeltag+'_'+str(epsDeg)+'.npy'
        np.save(datname,x_test_uap0)
        
        predictions = classifier.predict(x_test_uap0)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test[:N], axis=1)) / len(y_test)
        print(datname)
        print("Accuracy on first N adv examples: {}%".format(accuracy * 100))
        
        modelidx =modelidx +1

