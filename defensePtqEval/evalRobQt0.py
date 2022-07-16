#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 15:59:33 2020

@author: xyzhou
"""




import numpy as np

#from art.attacks.evasion import FastGradientMethod,ProjectedGradientDescent

#from art.utils import load_cifar10


# Step 1: Load the  dataset
from robustbench.data import load_cifar10

x_test, y_test = load_cifar10(n_examples=2000)

## Step 1a: Swap axes to PyTorch's NCHW format
#x_trainPth = x_train.transpose((0, 3, 1, 2)).astype(np.float32)
#x_testPth = x_test.transpose((0, 3, 1, 2)).astype(np.float32)


#%%
#from art.estimators.classification import KerasClassifier
#from art.estimators.classification import PyTorchClassifier


#%%
#from utils import loadRobustAdvModel
#from utils import loadKerasModel
#from utils import loadQtRobustAdvModel
#from utils import getAcc


# autoattack is installed as a dependency of robustbench so there is not need to install it separately
from autoattack import AutoAttack


#%%% other fp32 models worth exploring
# models from adversarial training
import robustbench
import torch
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier
from robustbench.eval import benchmark
    


print('Eval FP32 robust model:')
model = robustbench.utils.load_model(model_name='Wu2020Adversarial_extra',norm='Linf')
model.cuda()
adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
adversary.apgd.n_restarts = 1
x_adv = adversary.run_standard_evaluation(x_test, y_test)
# clean_acc, robust_acc = benchmark(model,
#                                   dataset='cifar10',
#                                   n_examples = 2000,
#                                   threat_model='Linf',
#                                   batch_size = 128,
#                                   eps = 8.0/255)
#print('Clean Acc:',clean_acc)
#print('Robust Acc:',robust_acc)    
# https://github.com/RobustBench/robustbench
#robPthModelList = [ 'Standard','Wu2020Adversarial_extra','Carmon2019Unlabeled','Wang2020Improving' ]
robQtPthModelList = [ ['linear','w8a8','Wu2020Adversarial_extra'],
                     ['linear','w7a7','Wu2020Adversarial_extra'], 
                     ['linear','w6a6','Wu2020Adversarial_extra'] ]
path = 'models/cifar10/Linf/'
print('Eval Quantized robust model:')
for robQtPthModel in robQtPthModelList:
    #model = loadQtRobustAdvModel(robQtPthModel) 
    model = robustbench.utils.load_model(model_name='Wu2020Adversarial_extra',norm='Linf')
    modelname = robQtPthModel[2]+'_'+robQtPthModel[0]+'_'+robQtPthModel[1]+'.pth'
    print('Quantization Setting:',robQtPthModel[1])
    model = torch.load(path+modelname)
    #model.load_state_dict(torch.load(path+modelname))
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
    adversary.apgd.n_restarts = 1
    x_adv = adversary.run_standard_evaluation(x_test, y_test)
    #Evaluate the Linf robustness of the model using AutoAttack
    
    # clean_acc, robust_acc = benchmark(model,
    #                                   dataset='cifar10',
    #                                   n_examples = 2000,
    #                                   threat_model='Linf')
    # print('Clean Acc:',clean_acc)
    # print('Robust Acc:',robust_acc)    



'''
Eval FP32 robust model:
using custom version including apgd-ce, apgd-dlr
initial accuracy: 88.10%
apgd-ce - 1/8 - 77 out of 250 successfully perturbed
apgd-ce - 2/8 - 70 out of 250 successfully perturbed
apgd-ce - 3/8 - 75 out of 250 successfully perturbed
apgd-ce - 4/8 - 54 out of 250 successfully perturbed
apgd-ce - 5/8 - 69 out of 250 successfully perturbed
apgd-ce - 6/8 - 79 out of 250 successfully perturbed
apgd-ce - 7/8 - 73 out of 250 successfully perturbed
apgd-ce - 8/8 - 5 out of 12 successfully perturbed
robust accuracy after APGD-CE: 63.00% (total time 1024.9 s)
apgd-dlr - 1/6 - 11 out of 250 successfully perturbed
apgd-dlr - 2/6 - 12 out of 250 successfully perturbed
apgd-dlr - 3/6 - 13 out of 250 successfully perturbed
apgd-dlr - 4/6 - 12 out of 250 successfully perturbed
apgd-dlr - 5/6 - 15 out of 250 successfully perturbed
apgd-dlr - 6/6 - 0 out of 10 successfully perturbed
robust accuracy after APGD-DLR: 59.85% (total time 1761.3 s)
max Linf perturbation: 0.03137, nan in tensor: 0, max: 1.00000, min: 0.00000
robust accuracy: 59.85%



Eval Quantized robust model:
    
Quantization Setting: w8a8
using custom version including apgd-ce, apgd-dlr
initial accuracy: 88.25%
apgd-ce - 1/8 - 77 out of 250 successfully perturbed
apgd-ce - 2/8 - 74 out of 250 successfully perturbed
apgd-ce - 3/8 - 77 out of 250 successfully perturbed
apgd-ce - 4/8 - 57 out of 250 successfully perturbed
apgd-ce - 5/8 - 69 out of 250 successfully perturbed
apgd-ce - 6/8 - 83 out of 250 successfully perturbed
apgd-ce - 7/8 - 73 out of 250 successfully perturbed
apgd-ce - 8/8 - 5 out of 15 successfully perturbed
robust accuracy after APGD-CE: 62.50% (total time 1023.7 s)
apgd-dlr - 1/5 - 12 out of 250 successfully perturbed
apgd-dlr - 2/5 - 11 out of 250 successfully perturbed
apgd-dlr - 3/5 - 11 out of 250 successfully perturbed
apgd-dlr - 4/5 - 13 out of 250 successfully perturbed
apgd-dlr - 5/5 - 12 out of 250 successfully perturbed
robust accuracy after APGD-DLR: 59.55% (total time 1747.7 s)
max Linf perturbation: 0.03137, nan in tensor: 0, max: 1.00000, min: 0.00000
robust accuracy: 59.55%

Quantization Setting: w7a7
using custom version including apgd-ce, apgd-dlr
initial accuracy: 88.10%
apgd-ce - 1/8 - 76 out of 250 successfully perturbed
apgd-ce - 2/8 - 75 out of 250 successfully perturbed
apgd-ce - 3/8 - 77 out of 250 successfully perturbed
apgd-ce - 4/8 - 54 out of 250 successfully perturbed
apgd-ce - 5/8 - 72 out of 250 successfully perturbed
apgd-ce - 6/8 - 82 out of 250 successfully perturbed
apgd-ce - 7/8 - 74 out of 250 successfully perturbed
apgd-ce - 8/8 - 5 out of 12 successfully perturbed
robust accuracy after APGD-CE: 62.35% (total time 1021.8 s)
apgd-dlr - 1/5 - 13 out of 250 successfully perturbed
apgd-dlr - 2/5 - 13 out of 250 successfully perturbed
apgd-dlr - 3/5 - 11 out of 250 successfully perturbed
apgd-dlr - 4/5 - 15 out of 250 successfully perturbed
apgd-dlr - 5/5 - 8 out of 247 successfully perturbed
robust accuracy after APGD-DLR: 59.35% (total time 1744.9 s)
max Linf perturbation: 0.03137, nan in tensor: 0, max: 1.00000, min: 0.00000
robust accuracy: 59.35%

Quantization Setting: w6a6
using custom version including apgd-ce, apgd-dlr
initial accuracy: 88.05%
apgd-ce - 1/8 - 77 out of 250 successfully perturbed
apgd-ce - 2/8 - 70 out of 250 successfully perturbed
apgd-ce - 3/8 - 81 out of 250 successfully perturbed
apgd-ce - 4/8 - 56 out of 250 successfully perturbed
apgd-ce - 5/8 - 76 out of 250 successfully perturbed
apgd-ce - 6/8 - 84 out of 250 successfully perturbed
apgd-ce - 7/8 - 80 out of 250 successfully perturbed
apgd-ce - 8/8 - 4 out of 11 successfully perturbed
robust accuracy after APGD-CE: 61.65% (total time 1020.5 s)
apgd-dlr - 1/5 - 14 out of 250 successfully perturbed
apgd-dlr - 2/5 - 11 out of 250 successfully perturbed
apgd-dlr - 3/5 - 9 out of 250 successfully perturbed
apgd-dlr - 4/5 - 8 out of 250 successfully perturbed
apgd-dlr - 5/5 - 9 out of 233 successfully perturbed
robust accuracy after APGD-DLR: 59.10% (total time 1736.7 s)
max Linf perturbation: 0.03137, nan in tensor: 0, max: 1.00000, min: 0.00000
robust accuracy: 59.10%
'''    


