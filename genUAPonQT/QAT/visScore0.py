#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 21:32:24 2021

@author: xyzhou
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from art.attacks.evasion import FastGradientMethod,ProjectedGradientDescent
#from art.estimators.classification import KerasClassifier
from art.estimators.classification import PyTorchClassifier
from art.utils import load_cifar10




    
                      
                      
# Step 1: Load the  dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
# Step 1a: Swap axes to PyTorch's NCHW format
x_train = x_train.transpose((0, 3, 1, 2)).astype(np.float32)
x_test = x_test.transpose((0, 3, 1, 2)).astype(np.float32)
x_test = x_test.astype('float32')#[:1000]
y_test = y_test.astype('float32')#[:1000]

# Step 2: Load the pretrained model
from robustbench.utils import load_model
robPthModelList = [ 'Wu2020Adversarial_extra','Carmon2019Unlabeled','Wang2020Improving' ]

# Step 3: Create the ART classifier
model = load_model(model_name='Standard', norm='Linf')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
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


def get2MaxPredScore(model,x_test):
    predictions = model.predict(x_test)
    #accuracy = np.sum(np.argmax(predictions, axis=1)
    # sorted array 
    #sorted_index_array = np.argsort(predictions,axis=0) 
      
    # sorted array 
    #sorted_pred = predictions[sorted_index_array]
    
    sorted_pred = np.sort(predictions,axis=1)
    
    # we want 2 largest class confidence values    
    # we are using negative indexing concept 
    max1_pred = sorted_pred[:,-1]
    max2_pred = sorted_pred[:,-2] 
    return max1_pred,max2_pred

max1_pred,max2_pred = get2MaxPredScore(classifier, x_test)

bins = np.linspace(0.0, 1.0, 30)
histogram, bins = np.histogram(max1_pred, bins=bins, density=True)

bin_centers = 0.5*(bins[1:] + bins[:-1])

# Compute the PDF on the bin centers from scipy distribution object
from scipy import stats
pdf = stats.norm.pdf(bin_centers)

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 4))
plt.plot(bin_centers, histogram, label="Histogram of samples")
plt.plot(bin_centers, pdf, label="PDF")
plt.legend()
plt.show()
