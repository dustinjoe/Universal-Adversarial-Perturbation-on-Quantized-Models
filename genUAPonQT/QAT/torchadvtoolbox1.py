#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 09:39:42 2020

@author: xyzhou
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.utils import load_cifar10

from utils import model_with_cfg
#from losses import SqrHingeLoss

# Step 1: Load the cifar10 dataset

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()

# Step 1a: Swap axes to PyTorch's NCHW format

x_train = x_train.transpose((0, 3, 1, 2)).astype(np.float32)
x_test = x_test.transpose((0, 3, 1, 2)).astype(np.float32)

# Step 2: Load the pretrained model

from CNV import CNV
from CNVfp32 import CNVfp32
def loadBrevitasModel(Wb,Ab):
    in_bit_width = 8
    num_classes = 10 #cfg.getint('MODEL', 'NUM_CLASSES')
    in_channels = 3  #cfg.getint('MODEL', 'IN_CHANNELS')
    
    if Wb==32 and Ab==32:
        model = CNVfp32(weight_bit_width=Wb,
                  act_bit_width=Ab,
                  in_bit_width=in_bit_width,
                  num_classes=num_classes,
                  in_ch=in_channels)
        model_filename = 'bnn_models/cnv_w'+str(Wb)+'a'+str(Ab)+'.tar'
        print(model_filename)
        checkpoint = torch.load(model_filename)
        model.load_state_dict(checkpoint['state_dict'],strict=False)
    # qtModelList = [ [2,2],[3,2],[3,3],[4,4],[8,8] ]
    elif [Wb,Ab] in [ [1,1],[1,2],[2,2] ]:
        cfgstr = 'cnv_'+str(Wb)+'w'+str(Ab)+'a'
        model, _ = model_with_cfg(cfgstr, pretrained=True)
    else:
        model = CNV(weight_bit_width=Wb,
                  act_bit_width=Ab,
                  in_bit_width=in_bit_width,
                  num_classes=num_classes,
                  in_ch=in_channels)
        model_filename = 'bnn_models/cnv_w'+str(Wb)+'a'+str(Ab)+'.tar'
        print(model_filename)
        checkpoint = torch.load(model_filename)
        model.load_state_dict(checkpoint['state_dict'],strict=False)
        
    return model
#optimizer.load_state_dict(checkpoint['optimizer'])

#model = load_checkpoint('./bnn_models/cnv_1w1a-758c8fef.pth')


#model, _ = model_with_cfg('cnv_1w1a', pretrained=True)
#model = torch.load("./bnn_models/cnv_1w1a-758c8fef.pth")
#model = model.eval().cuda()
Wb = 32
Ab = 32
model = loadBrevitasModel(Wb,Ab)

criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=0.01)
classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    #optimizer=optimizer,
    input_shape=(3, 28, 28),
    nb_classes=10,
)


# Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples
attack = FastGradientMethod(estimator=classifier, eps=0.2)
x_test_adv = attack.generate(x=x_test)

# Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))