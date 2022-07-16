#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow import keras

from tensorflow.keras.models import load_model

import numpy as np
import random

from art.attacks.evasion import FastGradientMethod,ProjectedGradientDescent
from art.estimators.classification import KerasClassifier
from art.utils import load_cifar10

from keras.datasets import cifar10
from keras.utils import np_utils
# Step 1: Load the  dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
'''
# randomly selecting 1000 images for each class in 10 classes 
trainIdx = []
num_each_class = 10
for class_i in range(10):
    np.random.seed(111)
    idx = np.random.choice( np.where(y_train[:, class_i]==1)[0], num_each_class, replace=False ).tolist()
    trainIdx = trainIdx + idx
random.shuffle(trainIdx)
x_train, y_train = x_train[trainIdx], y_train[trainIdx]
'''
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
from transfer_uap import TranferUniversalPerturbation

#N= len(x_test) 
N= 10000 #500
advMode = 'tr'
if advMode=='tr':
    x_dat = x_train[:N]
    y_dat = y_train[:N]    
elif advMode=='tst':
    x_dat = x_test[:N]
    y_dat = y_test[:N] 

N_eps = 5
N_model = 6
accResults = np.zeros((N_eps,N_model))
    
epsidx = 0


for epsDeg in [12,20]:#[4,8,16,26,32]:

    advEps = epsDeg/255.0
#uap_attack = TranferUniversalPerturbation(estimatorSet=classifierSet, attacker='deepfool',eps=advEps, max_iter=10)
   
    uap_attack = TranferUniversalPerturbation(
        estimatorSet=classifierSet,
        attacker='fgsm',
        delta=0.000001,
        attacker_params={'eps':advEps/5},
        max_iter=10,
        eps=advEps)
    
    print('Generating adv on N='+str(N)+advMode+' data samples.')
    x_test_uap0,x_uap_noise = uap_attack.generate(x=x_dat, y=y_dat)
    np.save('multiqmodel/uap'+str(N)+advMode+'_'+str(epsDeg)+'.npy',x_uap_noise) 


    #### Batch eval results on all models under together
    modelidx = 0
    
    x_test_uap0 = x_test_uap0.astype('float32')
    #np.save('x_uap'+str(N)+advMode+'_'+str(epsDeg)+'.npy',x_test_uap0)
    predictions = classifierFP.predict(x_test_uap0)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_dat, axis=1)) / N
    print("FP32 Accuracy on first N adv examples: {}%".format(accuracy * 100))
    accResults[epsidx,modelidx] = accuracy
    modelidx = modelidx+1
    
    numQtModel = len(qtModelList)
    for i in range(numQtModel):
        modelCFG = qtModelList[i]
        classifierQt = classifierSet[i+1]
        predictions = classifierQt.predict(x_test_uap0)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_dat, axis=1)) / N
    
        #accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test[:N], axis=1)) / len(y_test)
        
        wb = modelCFG[0]
        ab = modelCFG[1] 
        modelid = 'QModel_W'+str(wb)+'A'+str(ab)
        print(modelid+" Accuracy on first N adv examples: {}%".format(accuracy * 100))
        
        accResults[epsidx,modelidx] = accuracy
        modelidx = modelidx+1
    epsidx = epsidx+1  
    
datname = 'multiqmodel/uap'+advMode+'Results.npy'
np.save(datname,accResults)


#%%
#N= len(x_test) 
N= 10000 #500
advMode = 'tst'
if advMode=='tr':
    x_dat = x_train[:N]
    y_dat = y_train[:N]    
elif advMode=='tst':
    x_dat = x_test[:N]
    y_dat = y_test[:N] 

N_eps = 5
N_model = 6
accResults = np.zeros((N_eps,N_model))
    
epsidx = 0


for epsDeg in [12,20]:#[4,8,16,26,32]:

    advEps = epsDeg/255.0
#uap_attack = TranferUniversalPerturbation(estimatorSet=classifierSet, attacker='deepfool',eps=advEps, max_iter=10)
   
    uap_attack = TranferUniversalPerturbation(
        estimatorSet=classifierSet,
        attacker='fgsm',
        delta=0.000001,
        attacker_params={'eps':advEps/5},
        max_iter=10,
        eps=advEps)
    
    print('Generating adv on N='+str(N)+advMode+' data samples.')
    x_test_uap0,x_uap_noise = uap_attack.generate(x=x_dat, y=y_dat)
    np.save('multiqmodel/uap'+str(N)+advMode+'_'+str(epsDeg)+'.npy',x_uap_noise) 


    #### Batch eval results on all models under together
    modelidx = 0
    
    x_test_uap0 = x_test_uap0.astype('float32')
    #np.save('x_uap'+str(N)+advMode+'_'+str(epsDeg)+'.npy',x_test_uap0)
    predictions = classifierFP.predict(x_test_uap0)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_dat, axis=1)) / N
    print("FP32 Accuracy on first N adv examples: {}%".format(accuracy * 100))
    accResults[epsidx,modelidx] = accuracy
    modelidx = modelidx+1
    
    numQtModel = len(qtModelList)
    for i in range(numQtModel):
        modelCFG = qtModelList[i]
        classifierQt = classifierSet[i+1]
        predictions = classifierQt.predict(x_test_uap0)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_dat, axis=1)) / N
    
        #accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test[:N], axis=1)) / len(y_test)
        
        wb = modelCFG[0]
        ab = modelCFG[1] 
        modelid = 'QModel_W'+str(wb)+'A'+str(ab)
        print(modelid+" Accuracy on first N adv examples: {}%".format(accuracy * 100))
        
        accResults[epsidx,modelidx] = accuracy
        modelidx = modelidx+1
    epsidx = epsidx+1  
    
datname = 'multiqmodel/uap'+advMode+'Results.npy'
np.save(datname,accResults)



#%%  Test on the whole test set
x_test_uapall = x_test.copy()
# Apply attack and clip
x_test_uapall = x_test_uapall + x_uap_noise
x_test_uapall = np.clip(x_test_uapall, 0.0, 1.0)

print('Results on more test data:')

predictions = classifierFP.predict(x_test_uapall)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("FP32 Accuracy on adv examples: {}%".format(accuracy * 100))

numQtModel = len(qtModelList)
for i in range(numQtModel):
    modelCFG = qtModelList[i]
    classifierQt = classifierSet[i+1]
    predictions = classifierQt.predict(x_test_uapall)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    
    wb = modelCFG[0]
    ab = modelCFG[1] 
    modelid = 'QModel_W'+str(wb)+'A'+str(ab)
    print(modelid+" Accuracy on adv examples: {}%".format(accuracy * 100))
