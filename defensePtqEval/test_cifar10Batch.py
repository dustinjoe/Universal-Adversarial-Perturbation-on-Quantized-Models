#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 14:55:29 2022

@author: xyzhou
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import os, sys; 
#sys.path.append('/media/xyzhou/extDisk2t/_AAAI22/QtPreserveRob/robustbench')

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from art.attacks.evasion import FastGradientMethod,ProjectedGradientDescent
#from art.estimators.classification import KerasClassifier
from art.estimators.classification import PyTorchClassifier


from torchvision.datasets import CIFAR10
import torchvision.transforms as tt
from torch.utils.data.dataloader import DataLoader


# Step 1: Load the  dataset
stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
train_transform = tt.Compose([
    tt.RandomHorizontalFlip(),
    tt.RandomCrop(32,padding=4,padding_mode="reflect"),
    tt.ToTensor()
])
test_transform = tt.Compose([
    tt.ToTensor()
])

train_data = CIFAR10(download=True,root="./data",transform=train_transform)
test_data = CIFAR10(root="./data",train=False,transform=test_transform)
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
dataRatio = 1.0
split = int(np.floor(num_train*dataRatio))

train_idx = indices[:split]
#trainset_sel,trainset_others = random_split(train_data,[split, num_train-split])
train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)

dataset_size = len(train_data)

BATCH_SIZE=128
train_dl = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2)
#train_dl = DataLoader(train_data,BATCH_SIZE,num_workers=4,pin_memory=True,shuffle=True)
test_dl = DataLoader(test_data,BATCH_SIZE,num_workers=4,pin_memory=True)

def extractXY(dataloader,n_sample):
    
    l = [x for (x, y) in dataloader]
    x_test = torch.cat(l, 0)
    x_test = x_test[:n_sample]
    l = [y for (x, y) in dataloader]
    y_test = torch.cat(l, 0)
    y_test = y_test[:n_sample]
    x_dat = x_test.cpu().detach().numpy().astype('float32')[:n_sample]
    y_dat = y_test.cpu().detach().numpy().astype('float32')[:n_sample]
    num_classes = 10
    y_dat_1hot = np.eye(10)[y_dat.astype('int')]
    return x_dat,y_dat_1hot
n_sample = 10000
x_train,y_train = extractXY(train_dl, n_sample)
x_test,y_test = extractXY(test_dl, n_sample)
del train_data,train_dl,test_data,test_dl

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
#from svhnLib0 import getModelSvhn

#from qtlib import qtLinear
import copy
from collections import OrderedDict
from utee import quant
def qtLinear(model_raw0,modelname,param_bits,fwd_bits,bn_bits=32,overflow_rate=0.0,n_sample=20,saveqt=False):


    assert torch.cuda.is_available(), 'no cuda'  
  
    model_raw = copy.deepcopy(model_raw0)

    # quantize parameters
    if  param_bits < 32:
        state_dict = model_raw.state_dict()
        state_dict_quant = OrderedDict()
        #sf_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'running' in k:
                if  bn_bits >=32:
                    #print("Ignoring {}".format(k))
                    state_dict_quant[k] = v
                    continue
                else:
                    bits =  bn_bits
            else:
                bits =  param_bits


            sf = bits - 1. - quant.compute_integral_part(v, overflow_rate= overflow_rate)
            v_quant  = quant.linear_quantize(v, sf, bits=bits)

            state_dict_quant[k] = v_quant
            #print(k, bits)
        model_raw.load_state_dict(state_dict_quant)

    # quantize forward activation
    if  fwd_bits < 32:
        model_raw = quant.duplicate_model_with_quant(model_raw, bits= fwd_bits, overflow_rate= overflow_rate,
                                                     counter= n_sample, type= 'linear')
        #print(model_raw)
       
    # classifier = artTorchModel(model_raw)
    # acc1 = getAcc(classifier,x_test,y_test)
    # if acc1>0.80:
    #     print("QT Accuracy on benign test examples: {}%".format(acc1 * 100))
    #     #print('QT Acc>0.80~')
    if saveqt==True:
        filename = modelname+'_linear_w'+str(param_bits)+'a'+str(fwd_bits)+'.pth'
        torch.save(model_raw, filename)
    

    # print sf
    # sf decide the least precision, bits decide the range. So if sf=2 and bits=8, it means the least precision is 0.25, and range is [-127, 128] * 0.25
    #print(model_raw)
    return model_raw


# https://github.com/RobustBench/robustbench
#robPthModelList = [ 'Standard','Wu2020Adversarial_extra','Carmon2019Unlabeled','Wang2020Improving' ]
print('Eval Adv Training Cifar10 WRN34 Model.')
#robModelName = 'Wu2020Adversarial_extra'
robModelName = 'Rade2021Helper_extra'
qtModelList = [ [32,32],[8,8],[7,7],[6,6],[5,5]]
#qtModelList = [ [32,32] ]
#qtModelList = [ [2,2],[4,4],[8,8] 


from robustbench.utils import load_model

def loadModels(path,qtModelList):
    models = []
    criterion = nn.CrossEntropyLoss()
    min_pixel_value=0.0
    max_pixel_value=1.0


    #modelFP32 = getModelSvhn()
    modelFP32 = load_model(model_name=robModelName, dataset='cifar10', threat_model='Linf')
    modelname = 'Adv'
    for modelCFG in qtModelList:
        Wb = modelCFG[0]
        Ab = modelCFG[1]
        if Wb==32 and Ab==32:
            model0=modelFP32.cuda()
        else:
            model0 = qtLinear(modelFP32,modelname,Wb,Ab)
        # Step 3: Create the ART classifier
        #optimizer = optim.Adam(model.parameters(), lr=0.01)
        classifier = PyTorchClassifier(
            model=model0,
            clip_values=(min_pixel_value, max_pixel_value),
            loss=criterion,
            #optimizer=optimizer,
            input_shape=(3, 28, 28),
            nb_classes=10,
        )
        models.append(classifier)
    return models
path = 'models/'
classifierSet = loadModels(path,qtModelList)

# Step 4: Evaluate the ART classifier on benign test examples
epsList = [4,8,12,16,26,32]
N_eps = len(epsList)+1
N_model = len(qtModelList)
accResults = np.zeros((N_eps,N_model))

#%%
modelidx = 0
for classifier in classifierSet:    
    predictions = classifier.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    accResults[0,modelidx] = accuracy
    modelidx = modelidx+1
'''
# Step 4: Evaluate the ART classifier on benign test examples
classifierFP = classifierSet[0]
predictions = classifierFP.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))
'''

#%% Step 5: Generate adversarial test examples
from transfer_uap import TransferUniversalPerturbation

phase = 'eval' # or 'adv'

#N= len(x_test) 
N= 10000 #500
advMode = 'tr'  # 'tst'
modelType = 'nat' #'adv'
modelMode = 'multi'# 'single'  
if phase == 'adv' and advMode=='tr':
    x_dat = x_train[:N]
    y_dat = y_train[:N]    
else:
    x_dat = x_test[:N]
    y_dat = y_test[:N] 
'''
N_eps = 5
N_model = len(qtModelList)
accResults = np.zeros((N_eps,N_model))
'''  
epsidx = 0
#%%

for epsDeg in epsList:

    advEps = epsDeg/255.0
#uap_attack = TranferUniversalPerturbation(estimatorSet=classifierSet, attacker='deepfool',eps=advEps, max_iter=10)
    if phase=='adv':
        uap_attack = TransferUniversalPerturbation(
            estimatorSet=classifierSet,
            attacker='fgsm',
            delta=0.000001,
            attacker_params={'eps':advEps/5},
            max_iter=10,
            eps=advEps)
        
        print('Generating adv on N='+str(N)+advMode+' data samples.')
        x_test_uap0,x_uap_noise = uap_attack.generate(x=x_dat, y=y_dat)
        np.save('./UAPs/ptq_cifar100/uap'+str(N)+advMode+'_'+str(epsDeg)+'.npy',x_uap_noise) 
    else:
        print('Eval adv on N='+str(N)+' tst data samples of advEps='+str(epsDeg)+'/255')
        pertPath = './UAPs/ptq_cifar100/'+modelType+'/'+modelMode+'/'
        pertName = 'uap'+str(N)+advMode+'_'+str(epsDeg)+'.npy'
        x_uap_noise = np.load( pertPath+pertName )
        x_test_uap0 = x_dat + x_uap_noise
        x_test_uap0 = np.clip(x_test_uap0, 0.0, 1.0)

    #### Batch eval results on all models under together
   
    
    x_test_uap0 = x_test_uap0.astype('float32')
    #np.save('x_uap'+str(N)+advMode+'_'+str(epsDeg)+'.npy',x_test_uap0)
    '''
    predictions = classifierFP.predict(x_test_uap0)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_dat, axis=1)) / N
    print("FP32 Accuracy on first N adv examples: {}%".format(accuracy * 100))
    accResults[epsidx,modelidx] = accuracy
    modelidx = modelidx+1
    '''
    
    modelidx = 0
    numQtModel = len(qtModelList)
    for i in range(numQtModel):
        modelCFG = qtModelList[i]
        classifierQt = classifierSet[i]
        predictions = classifierQt.predict(x_test_uap0)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_dat, axis=1)) / N
    
        #accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test[:N], axis=1)) / len(y_test)
        
        wb = modelCFG[0]
        ab = modelCFG[1] 
        modelid = 'QModel_W'+str(wb)+'A'+str(ab)
        print(modelid+" Accuracy on first N adv examples: {}%".format(accuracy * 100))
        
        accResults[epsidx+1,modelidx] = accuracy
        modelidx = modelidx+1
    epsidx = epsidx+1
    
    
#datname = './UAPs/ptq_cifar100/uap'+advMode+'Results.npy'
#np.save(datname,accResults)

#%%  Test on the whole test set
x_test_uapall = x_test.copy()
# Apply attack and clip
x_test_uapall = x_test_uapall + x_uap_noise
x_test_uapall = np.clip(x_test_uapall, 0.0, 1.0)

print('Results on more test data:')
'''
predictions = classifierFP.predict(x_test_uapall)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("FP32 Accuracy on adv examples: {}%".format(accuracy * 100))
'''
numQtModel = len(qtModelList)
for i in range(numQtModel):
    modelCFG = qtModelList[i]
    classifierQt = classifierSet[i]
    predictions = classifierQt.predict(x_test_uapall)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    
    wb = modelCFG[0]
    ab = modelCFG[1] 
    modelid = 'QModel_W'+str(wb)+'A'+str(ab)
    print(modelid+" Accuracy on adv examples: {}%".format(accuracy * 100))
    
#%% Test UAPs on Nat Model
def getAcc(model,x_dat,y_dat):
    predictions = model.predict(x_dat)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_dat, axis=1)) / len(y_dat)
    #print("Accuracy on test examples: {}%".format(accuracy * 100)) 
    return accuracy
# load a nat model of cifar100 ( adv trained model on corruptions )
modelFP32 = load_model(model_name='Standard', dataset='cifar10')
criterion = nn.CrossEntropyLoss()
min_pixel_value=0.0
max_pixel_value=1.0
classifier = PyTorchClassifier(
    model=modelFP32,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    #optimizer=optimizer,
    input_shape=(3, 28, 28),
    nb_classes=10,
)
accAdvOnNat = np.zeros( len(epsList)+1 )
epsidx = 0
accAdvOnNat[epsidx] = getAcc(classifier,x_test,y_test)
for epsDeg in epsList:
    epsidx = epsidx+1
    advEps = epsDeg/255.0

    print('Eval adv on N='+str(N)+' tst data samples of advEps='+str(epsDeg)+'/255')

    pertPath = './UAPs/ptq_cifar100/'+modelType+'/'+modelMode+'/'
    pertName = 'uap'+str(N)+advMode+'_'+str(epsDeg)+'.npy'
    x_uap_noise = np.load( pertPath+pertName )
    x_test_uap0 = x_dat + x_uap_noise
    x_test_uap0 = np.clip(x_test_uap0, 0.0, 1.0)

    #### Batch eval results on nat model under together     
    x_test_uap0 = x_test_uap0.astype('float32')
    

    predictions = classifier.predict(x_test_uap0)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print('advEps:'+str(epsDeg)+'/255')
    print("Nat Accuracy on Adv: {}%".format(accuracy * 100))   
    accAdvOnNat[epsidx] = accuracy

#%% Plotting
import matplotlib.pyplot as plt

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (8, 5),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

datPlt = accResults.T
titlestr = 'CIFAR10 PTQ Classification under Trainingset UAP'

plt.figure()
plt.title(titlestr)
epsList = np.array([0,4,8,12,16,26,32])
epsLen = len(epsList)
x_axis = epsList/255.0

qtModelList = [ [32,32],[8,8],[7,7],[6,6],[5,5]]
plt.plot(x_axis, accAdvOnNat ,label='Nat fp32')
plt.plot(x_axis, datPlt[0,:] ,label='Adv fp32')
plt.plot(x_axis, datPlt[1,:] ,label='Adv w8a8')
plt.plot(x_axis, datPlt[2,:] ,label='Adv w7a7')
plt.plot(x_axis, datPlt[3,:] ,label='Adv w6a6')
plt.plot(x_axis, datPlt[4,:] ,label='Adv w5a5')



plt.xlabel('Increasing Adv Strength/Epsilon')
plt.ylabel('Accuracy/%')

#dropStr = 'Model Setting'
#plt.plot([], [], ' ', label=dropStr)
plt.legend(title="Model Setting")
#plt.legend(title="Model Setting",
#          loc="center left",
#          bbox_to_anchor=(1.0, 0.12, 0.5, 0.7))
    
filestr = 'accCifar10_PTQ_white_tr'+robModelName+'.png'
#plt.savefig(filestr, dpi=600)
plt.savefig(filestr, dpi=450,bbox_inches = 'tight',pad_inches = 0.01)
plt.show()
#%%
