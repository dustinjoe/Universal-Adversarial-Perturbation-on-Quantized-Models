#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 00:30:55 2020

@author: xyzhou
"""
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow import keras

from tensorflow.keras.models import load_model

import numpy as np

#from tensorflow.keras.utils import get_file
from tensorflow.keras.models import Model, load_model, model_from_json

from ptqlib.quantization_ops import (WeightFloat, WeightQuantizer,
                               TrainableWeightQuantizer)
from ptqlib.quantization_layers import (QuantizedConv2D, QuantizedDepthwiseConv2D,
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

def load_keras_qtmodel(filepath, compile=True):
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

from art.estimators.classification import KerasClassifier
def loadPTQKerasModel(wb,ab):
    path = 'ptq_models/'
    min_pixel_value = 0.0
    max_pixel_value = 1.0
    if wb==32 and ab==32:
        modelname = 'ds_cnn_cifar10.h5'
        model0 = load_keras_qtmodel(path+modelname)
        #model0.summary()
        model0.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer='adam',
            metrics=['accuracy'])
        # Step 3: Create the ART classifier
        classifier = KerasClassifier(model=model0, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)
    
    else:
        modelname = 'model_w'+str(wb)+'a'+str(ab)+'.h5'
        model0 = load_keras_qtmodel(path+modelname)        
        model0.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer='adam',
            metrics=['accuracy'])
        # Step 3: Create the ART classifier
        classifier = KerasClassifier(model=model0, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)

    return classifier




import torch
import torch.nn as nn
from uqatlib.utils import model_with_cfg
from uqatlib.CNV import CNV
from art.estimators.classification import PyTorchClassifier
#from CNVfp32 import CNVfp32
def loadQATBrevitasModel(Wb,Ab):
    in_bit_width = 8
    num_classes = 10 #cfg.getint('MODEL', 'NUM_CLASSES')
    in_channels = 3  #cfg.getint('MODEL', 'IN_CHANNELS')
    min_pixel_value = 0.0
    max_pixel_value = 1.0
    
    path = 'uqat_models/'
    
    if [Wb,Ab] in [ [1,1],[1,2],[2,2] ]:
        cfgstr = 'cnv_'+str(Wb)+'w'+str(Ab)+'a'
        print(cfgstr)
        model, _ = model_with_cfg(cfgstr, pretrained=True)
    else:
        model = CNV(weight_bit_width=Wb,
                  act_bit_width=Ab,
                  in_bit_width=in_bit_width,
                  num_classes=num_classes,
                  in_ch=in_channels)
        model_filename = 'cnv_w'+str(Wb)+'a'+str(Ab)+'.tar'
        print(model_filename)
        checkpoint = torch.load(path+model_filename)
        model.load_state_dict(checkpoint['state_dict'],strict=False)

    criterion = nn.CrossEntropyLoss()
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        #optimizer=optimizer,
        input_shape=(3, 28, 28),
        nb_classes=10,
    )
    return classifier


#%%% other fp32 models worth exploring
# models from adversarial training
import robustbench
def loadRobustAdvModel(modelname):
    #from robustbench.utils import load_model
    #model = load_model(model_name='Carmon2019Unlabeled', norm='Linf')
    
    model = robustbench.utils.load_model(model_name=modelname,norm='Linf')
    
    # Step 3: Create the ART classifier
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    min_pixel_value = 0.0
    max_pixel_value = 1.0
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
    )
    
    return classifier

def loadQtRobustAdvModel(modeldef):
    #from robustbench.utils import load_model
    #model = load_model(model_name='Carmon2019Unlabeled', norm='Linf')
    modelclass = modeldef[1]
    modelname = modeldef[0]
    path = 'models/Linf/'
    model = robustbench.utils.load_model(model_name=modelclass,norm='Linf')
    model = torch.load(path+modelname)
    
    # Step 3: Create the ART classifier
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    min_pixel_value = 0.0
    max_pixel_value = 1.0
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
    )
    
    return classifier


# several classic models
def loadKerasModel(modelname):
    # 'resnet_v1.h5','resnet_v2.h5','vgg_cifar10.h5'
    modelpath = 'models/'
    model = load_model(modelpath+modelname)
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy'])
    # Step 3: Create the ART classifier
    min_pixel_value = 0.0
    max_pixel_value = 1.0
    classifier = KerasClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)
    return classifier



#%%
def getAcc(model,x_dat,y_dat):
    predictions = model.predict(x_dat)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_dat, axis=1)) / len(y_dat)
    print("Accuracy on test examples: {}%".format(accuracy * 100)) 
    return accuracy


import matplotlib.pyplot as plt
def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()
    