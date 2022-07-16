#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 10:27:15 2021

@author: xyzhou
"""


import os, sys; 
sys.path.append('/media/xyzhou/extDisk2t/_AAAI22/QtPreserveRob/robustbench')



import torch
#import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
#from torchvision import models
from torchvision import datasets, transforms
import numpy as np

from torch.utils.data import ChainDataset,ConcatDataset,random_split
# check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

from torchvision.datasets import CIFAR100,CIFAR10,SVHN
import torchvision.transforms as tt
from torch.utils.data.dataloader import DataLoader

stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
train_transform = tt.Compose([
    tt.RandomHorizontalFlip(),
    tt.RandomCrop(32,padding=4,padding_mode="reflect"),
    tt.ToTensor()
])

test_transform = tt.Compose([
    tt.ToTensor()
])

train_data = SVHN(root='./data/svhn', split='train', download=True, transform=train_transform)
#train_data = CIFAR10(download=True,root="./data",transform=train_transform)
test_data = SVHN(root='./data/svhn', split='test',download=True,transform=test_transform)
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

#%%

from robustbench.data import load_cifar10

#x_test, y_test = load_cifar100(n_examples=2000)

from robustbench.utils import load_model
modelname = 'Standard'#'Carmon2019Unlabeled'
model_ft = load_model(model_name=modelname,  norm='Linf',dataset='cifar10').cuda()


# autoattack is installed as a dependency of robustbench so there is not need to install it separately
from autoattack import AutoAttack

#adversary = AutoAttack(model_ft, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
#adversary.apgd.n_restarts = 1
#x_adv = adversary.run_standard_evaluation(x_test, y_test,bs=64)



#%%
import copy
#import torch.optim as optim
from torch.optim import lr_scheduler

#from torch.autograd import Variable
#from utils_awp2 import Bar, Logger, AverageMeter, accuracy
#from utils_awp import TradesAWP

#import os

modelDir = modelname+'_C10NatftSVNH.pth'


def train_model(model,dataloader,dataset_size, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase

        model.train()  # Set model to training mode


        running_loss = 0.0
        running_corrects = 0
        phase = 'train'

        # Iterate over data.
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

        # deep copy the model
        if  epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


awp_gamma = 0.005
awp_warmup = 10
if awp_gamma <= 0.0:
    awp_warmup = np.infty

NUM_CLASSES = 10

#epsilon = 3 / 255
#step_size = 1 / 255
step_size=0.003
epsilon=0.031
#perturb_steps=10
num_steps = 10
norm = 'l_inf'
beta = 6.0
save_freq =2


def adjust_learning_rate(optimizer, epoch,lr):
    """decrease the learning rate"""
    lr = lr
    if epoch >= 100:
        lr = lr * 0.1
    if epoch >= 150:
        lr = lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



start_epoch= 1
epochs = 5
lr = 0.005
momentum = 0.9
weight_decay = 5e-4


# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_ft.fc.in_features
#model_ft.fc = nn.Linear(num_ftrs, 10)
model_ft.fc = nn.Linear(num_ftrs, 10)
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# NOTE: pytorch optimizer explicitly accepts parameter that requires grad
# see https://github.com/pytorch/pytorch/issues/679
#optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.001, momentum=0.9)


# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

criterion =  torch.nn.CrossEntropyLoss(reduction="sum")

robAccList = []        


model_ft = train_model(model_ft,train_dl,dataset_size, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=epochs)


#torch.save(model_ft.state_dict(), 'model_current.pth')
modelsavname = modelname+'_C10ftNatSVNH.pth'
torch.save(model_ft, modelsavname)


#%%
n_sample = 2000
l = [x for (x, y) in test_dl]
x_test = torch.cat(l, 0)
x_test = x_test[:n_sample]
l = [y for (x, y) in test_dl]
y_test = torch.cat(l, 0)
y_test = y_test[:n_sample]
'''
from robustbench.data import load_cifar10

x_test, y_test = load_cifar10(n_examples=2000)

print('Eval Standard Non-robust Model')
model = load_model(model_name='Standard',  norm='Linf').cuda()
adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
adversary.apgd.n_restarts = 1
x_adv = adversary.run_standard_evaluation(x_test, y_test,bs=128)


Eval Normal Model
using custom version including apgd-ce, apgd-dlr
initial accuracy: 94.50%
apgd-ce - 1/1 - 189 out of 189 successfully perturbed
robust accuracy after APGD-CE: 0.00% (total time 84.1 s)
max Linf perturbation: 0.03137, nan in tensor: 0, max: 1.00000, min: 0.00000
robust accuracy: 0.00%
'''

#%%

print('Eval Transfer Model')
adversary = AutoAttack(model_ft, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
adversary.apgd.n_restarts = 1
x_adv = adversary.run_standard_evaluation(x_test, y_test,bs=128)

'''
Eval Transfer Model
using custom version including apgd-ce, apgd-dlr
initial accuracy: 90.50%
apgd-ce - 1/1 - 101 out of 181 successfully perturbed
robust accuracy after APGD-CE: 40.00% (total time 82.2 s)
apgd-dlr - 1/1 - 1 out of 80 successfully perturbed
robust accuracy after APGD-DLR: 39.50% (total time 119.9 s)
max Linf perturbation: 0.03137, nan in tensor: 0, max: 1.00000, min: 0.00000
robust accuracy: 39.50%
'''



#%%
from qtlib import qtLinear
wbit = 8
abit = 8
model = copy.deepcopy(model_ft)
model = qtLinear(model,modelname,wbit,abit)
model = model.cuda()
print('Linear w8a8 eval.')
adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
adversary.apgd.n_restarts = 1
x_adv = adversary.run_standard_evaluation(x_test, y_test,bs=128)

wbit = 7
abit = 7
model =copy.deepcopy(model_ft)
model = qtLinear(model,modelname,wbit,abit)
model = model.cuda()
print('Linear w7a7 eval.')
adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
adversary.apgd.n_restarts = 1
x_adv = adversary.run_standard_evaluation(x_test, y_test,bs=128)



#%%
#from utils import getAcc
#from art.estimators.classification import PyTorchClassifier
import torch
import torch.nn as nn
# pertModelType: uqat,ptq
# pertPhase: single,multi
# advMode: tr,tst
# modelCFG: [Wb,Ab]
def loadQtPert(pertModelType,pertPhase,advMode,modelCFG,epsDeg):
    N=10000
    datpath = 'UAPs/'+pertModelType+'_'+pertPhase+'/'
    Wb = modelCFG[0]
    Ab = modelCFG[1]
    modeltag = 'w'+str(Wb)+'a'+str(Ab)
    if pertPhase == 'single':        
        datname = 'x_noise'+advMode+str(N)+'_'+modeltag+'_'+str(epsDeg)+'.npy'
    elif pertPhase == 'multi':
        datname = 'uap'+str(N)+advMode+'_'+str(epsDeg)+'.npy'
    print('Current Pert:',datpath+datname)
    datpert = np.load(datpath+datname)    
    
    return datpert

def getAccPth(model_ft,x_tst,y_tst,bs):
    adversary = AutoAttack(model_ft, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
    adversary.apgd.n_restarts = 1
    cleanacc = adversary.clean_accuracy(x_orig=x_tst, y_orig=y_tst,bs=bs)
    return cleanacc

# Evaluate transferability among PTQ and QAT models
def evalPertUAP(x_pert,x_dat,y_dat,pertModelType,modelPth,bs):
    # For one perturbation generated from quantized model(PTQ or QAT), 
    # evaluate performances on the other type
    #x_adv = x_dat+x_pert
    #x_adv = np.clip(x_adv,0.0,1.0)
    model = modelPth
    if pertModelType == 'ptq':
        # eval perturbation from PTQ on QAT models
        #qtModelList = [ [8,8],[4,4],[3,3],[3,2],[2,2],[1,2],[1,1] ]
        #x_adv = np.clip(x_dat+x_pert,0.0,1.0)
        #x_adv = x_adv.transpose((0, 3, 1, 2))
        x_pert = x_pert.transpose((0, 3, 1, 2))
        x_pert = torch.from_numpy(x_pert)
        x_adv = torch.clip(x_dat+x_pert,0.0,1.0)  
        accuracy = getAccPth(model_ft=model,x_tst=x_adv,y_tst=y_dat,bs=bs)
        
    elif pertModelType == 'uqat':
        #qtModelList = [ [8,8],[4,4],[3,3],[3,2],[2,2] ]
        #x_adv = torch.clip(x_dat+x_pert.transpose(0, 2, 3, 1),0.0,1.0)
        #x_pert = x_pert.transpose((0, 2, 3, 1))
        x_pert = torch.from_numpy(x_pert)        
        #print(x_dat.shape)
        #print(x_pert.shape)
        x_adv = torch.clip(x_dat+x_pert,0.0,1.0)          
        accuracy = getAccPth(model_ft=model,x_tst=x_adv,y_tst=y_dat,bs=bs)
    else:
        accuracy = 0.0

    return accuracy

ptqKerasModelList = [ [8,8],[4,4],[3,3],[3,2],[2,2] ]
#qtModelList = [ [2,2],[3,2],[3,3],[4,4],[8,8] ]
qatPthModelList = [ [8,8],[4,4],[3,3],[3,2],[2,2],[1,2],[1,1] ]
epsList = [4,8,12,16,20,26,32]
N = 10000
#epsList = [8,16,26]
    
pertModelType = 'ptq'

accPTQonMixed = np.zeros((len(epsList),2,1+len(ptqKerasModelList)))   
epsDegidx = -1
modelidx = -1
advModeidx = -1
pertPhaseidx = -1


wbit = 8
abit = 8
model = copy.deepcopy(model_ft)
model = qtLinear(model,modelname,wbit,abit)
model = model.cuda()
print('Linear w8a8 eval.')


#x_dat, y_dat = load_cifar10(n_examples=10000)
n_sample = 10000
l = [x for (x, y) in test_dl]
x_test = torch.cat(l, 0)
x_test = x_test[:n_sample]
l = [y for (x, y) in test_dl]
y_test = torch.cat(l, 0)
y_test = y_test[:n_sample]
#x_dat = x_test.cpu().detach().numpy()
#y_dat = y_test.cpu().detach().numpy()
x_dat = x_test
y_dat = y_test


#%%

bs = 128
accClean = getAccPth(model_ft=model,x_tst=x_dat,y_tst=y_dat,bs=bs)
print( 'Acc without attack:',str(accClean) )



#%%
espDegidx = -1      
for epsDeg in epsList:
    epsDegidx = (epsDegidx+1)%len(epsList)
    advModeidx=-1
    for advMode in ['tr','tst']:
        advModeidx = advModeidx+1            
        pertPhaseidx = -1  
        pertPhase ='single'
        for modelCFG in ptqKerasModelList:
            pertPhaseidx = pertPhaseidx+1                        
            x_pert = loadQtPert(pertModelType,pertPhase,advMode,modelCFG,epsDeg)
            #x_pert = torch.from_numpy(x_pert).cuda()
            accPert = evalPertUAP(x_pert,x_dat,y_dat,pertModelType,model,bs)
            accPTQonMixed[epsDegidx,advModeidx,pertPhaseidx] = accPert
            
        pertPhase = 'multi'
        x_pert = loadQtPert(pertModelType,pertPhase,advMode,modelCFG,epsDeg) 
        accPert = accPert = evalPertUAP(x_pert,x_dat,y_dat,pertModelType,model,bs)
        accPTQonMixed[epsDegidx,advModeidx,-1] = accPert
   
modelidx = -1
        

#filename = 'accPTQonRobQt.npy'
filename = 'accPTQonSvhnNatTransW8A8.npy'
np.save(filename,accPTQonMixed)       


#
pertModelType = 'uqat'
accQATonMixed = np.zeros((len(epsList),2,1+len(qatPthModelList)))   
epsDegidx = -1
modelidx = -1
advModeidx = -1
pertPhaseidx = -1


espDegidx = -1      
for epsDeg in epsList:
    epsDegidx = (epsDegidx+1)%len(epsList)
    advModeidx=-1
    for advMode in ['tr','tst']:
        advModeidx = advModeidx+1            
        pertPhaseidx = -1  
        pertPhase ='single'
        for modelCFG in qatPthModelList:
            pertPhaseidx = pertPhaseidx+1                        
            x_pert = loadQtPert(pertModelType,pertPhase,advMode,modelCFG,epsDeg) 
            #x_pert = torch.from_numpy(x_pert).cuda()
            accPert = evalPertUAP(x_pert,x_dat,y_dat,pertModelType,model,bs)
            accQATonMixed[epsDegidx,advModeidx,pertPhaseidx] = accPert
            
        pertPhase = 'multi'
        x_pert = loadQtPert(pertModelType,pertPhase,advMode,modelCFG,epsDeg) 
        accPert = evalPertUAP(x_pert,x_dat,y_dat,pertModelType,model,bs)
        accQATonMixed[epsDegidx,advModeidx,-1] = accPert
   
modelidx = -1

#filename = 'accQATonRobQt.npy'
filename = 'accQATonSvhnNatTransW8A8.npy'
np.save(filename,accQATonMixed)





#%%
'''
from svhnLib0 import getModelSvhn
print('Eval Natural Res20 SVHN Model.')
model = getModelSvhn()

accClean = getAccPth(model_ft=model,x_tst=x_dat,y_tst=y_dat,bs=bs)
print( 'Natual Model Acc without attack:',str(accClean) )


espDegidx = -1      
for epsDeg in epsList:
    epsDegidx = (epsDegidx+1)%len(epsList)
    advModeidx=-1
    for advMode in ['tr','tst']:
        advModeidx = advModeidx+1            
        pertPhaseidx = -1  
        pertPhase ='single'
        for modelCFG in ptqKerasModelList:
            pertPhaseidx = pertPhaseidx+1                        
            x_pert = loadQtPert(pertModelType,pertPhase,advMode,modelCFG,epsDeg)
            #x_pert = torch.from_numpy(x_pert).cuda()
            accPert = evalPertUAP(x_pert,x_dat,y_dat,pertModelType,model,bs)
            accPTQonMixed[epsDegidx,advModeidx,pertPhaseidx] = accPert
            
        pertPhase = 'multi'
        x_pert = loadQtPert(pertModelType,pertPhase,advMode,modelCFG,epsDeg) 
        accPert = accPert = evalPertUAP(x_pert,x_dat,y_dat,pertModelType,model,bs)
        accPTQonMixed[epsDegidx,advModeidx,-1] = accPert
   
modelidx = -1
        

#filename = 'accPTQonRobQt.npy'
filename = 'accPTQonSvhnNatW8A8.npy'
np.save(filename,accPTQonMixed)       


#
pertModelType = 'uqat'
accQATonMixed = np.zeros((len(epsList),2,1+len(qatPthModelList)))   
epsDegidx = -1
modelidx = -1
advModeidx = -1
pertPhaseidx = -1


espDegidx = -1      
for epsDeg in epsList:
    epsDegidx = (epsDegidx+1)%len(epsList)
    advModeidx=-1
    for advMode in ['tr','tst']:
        advModeidx = advModeidx+1            
        pertPhaseidx = -1  
        pertPhase ='single'
        for modelCFG in qatPthModelList:
            pertPhaseidx = pertPhaseidx+1                        
            x_pert = loadQtPert(pertModelType,pertPhase,advMode,modelCFG,epsDeg) 
            #x_pert = torch.from_numpy(x_pert).cuda()
            accPert = evalPertUAP(x_pert,x_dat,y_dat,pertModelType,model,bs)
            accQATonMixed[epsDegidx,advModeidx,pertPhaseidx] = accPert
            
        pertPhase = 'multi'
        x_pert = loadQtPert(pertModelType,pertPhase,advMode,modelCFG,epsDeg) 
        accPert = evalPertUAP(x_pert,x_dat,y_dat,pertModelType,model,bs)
        accQATonMixed[epsDegidx,advModeidx,-1] = accPert
   
modelidx = -1

#filename = 'accQATonRobQt.npy'
filename = 'accQATonSvhnNatW8A8.npy'
np.save(filename,accQATonMixed)
'''
#%%
x_dat = x_test[:2000]
y_dat = y_dat[:2000]
adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
adversary.apgd.n_restarts = 1
x_adv = adversary.run_standard_evaluation(x_dat, y_dat,bs=128)
