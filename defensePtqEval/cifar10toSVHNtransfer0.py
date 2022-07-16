#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 11:18:24 2021

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

# https://discuss.pytorch.org/t/datasets-imagefolder-targets-one-hot-encoded/102682/3
def target_to_oh(target):
    NUM_CLASS = 10  # hard code here, can do partial
    one_hot = torch.eye(NUM_CLASS)[target]
    return one_hot

# https://discuss.pytorch.org/t/datasets-imagefolder-targets-one-hot-encoded/102682/3
def target_to_oh1(target):  
    #soft label
    NUM_CLASS = 10  # hard code here, can do partial
    target_weight = 0.9
    one_hot = torch.eye(NUM_CLASS)[target]
    one_hot = one_hot*target_weight
    dif = (1-target_weight)/(NUM_CLASS-1)
    one_hot = one_hot+dif
    one_hot[target] = one_hot[target]-dif
    return one_hot







def imshow(image, ax=None, title=None, normalize=False):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax
# change this to the trainloader or testloader 
data_iter = iter(train_dl)

images, labels = next(data_iter)
fig, axes = plt.subplots(figsize=(10,4), ncols=4)
for ii in range(4):
    ax = axes[ii]
#     helper.imshow(images[ii], ax=ax, normalize=False)
    imshow(images[ii], ax=ax, normalize=False)


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
from evalLayerSensitivityLib import evalLayersAdv0,rmvNan
from robustbench.data import load_cifar10train
x_val, y_val = load_cifar10(n_examples=500)
print('=========Start layerwise adversarial sensitivity evaluation======')
batch_size = 64
MetricSensitivity = 2
modelLayerSensitivity,hookList = evalLayersAdv0(model_ft,modelname,x_val,y_val,batch_size, n_sample=64, num_batch=1,MetricSensitivity=MetricSensitivity)
modelLayerSensitivity = rmvNan(modelLayerSensitivity)
#qtThresi = np.percentile(modelLayerSensitivity, 0.7)
print('Mean Layer Sensitivity:',str( np.mean(modelLayerSensitivity) ))

#%%
import copy
import torch.optim as optim
from torch.optim import lr_scheduler

from torch.autograd import Variable
from utils_awp2 import Bar, Logger, AverageMeter, accuracy
from utils_awp import TradesAWP

import os

modelDir = modelname+'_C10ftSVNH.pth'
def perturb_input(model,
                  x_natural,
                  step_size=0.003,
                  epsilon=0.031,
                  perturb_steps=10,
                  distance='l_inf'):
    model.eval()
    batch_size = len(x_natural)
    if distance == 'l_inf':
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = F.kl_div(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(model(x_natural), dim=1),
                                   reduction='sum')
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * F.kl_div(F.log_softmax(model(adv), dim=1),
                                       F.softmax(model(x_natural), dim=1),
                                       reduction='sum')
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            # if (grad_norms == 0).any():
            #     delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv

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
        if phase == 'val' and epoch_acc > best_acc:
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
def train1epoch(model, train_loader, optimizer, epoch, awp_adversary):
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    print('epoch: {}'.format(epoch))
    bar = Bar('Processing', max=len(train_loader))

    for batch_idx, (data, target) in enumerate(train_loader):
        #target = torch.argmax(target, 1)
        x_natural, target = data.to(device), target.to(device)

        # craft adversarial examples
        x_adv = perturb_input(model=model,
                              x_natural=x_natural,
                              step_size=step_size,
                              epsilon=epsilon,
                              perturb_steps=num_steps,
                              distance=norm)

        model.train()
        # calculate adversarial weight perturbation
        if epoch >= awp_warmup:
            awp = awp_adversary.calc_awp(inputs_adv=x_adv,
                                         inputs_clean=x_natural,
                                         targets=target,
                                         beta=beta)
            awp_adversary.perturb(awp)

        optimizer.zero_grad()
        logits_adv = model(x_adv)
        loss_robust = F.kl_div(F.log_softmax(logits_adv, dim=1),
                               F.softmax(model(x_natural), dim=1),
                               reduction='batchmean')
        # calculate natural loss and backprop
        logits = model(x_natural)
        
        loss_natural = F.cross_entropy(logits, target)
        #loss_natural = F.cross_entropy(logits, target,reduction="sum")
        loss = loss_natural + beta * loss_robust

        prec1, prec5 = accuracy(logits_adv, target, topk=(1, 5))
        losses.update(loss.item(), x_natural.size(0))
        top1.update(prec1.item(), x_natural.size(0))

        # update the parameters at last
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= awp_warmup:
            awp_adversary.restore(awp)

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s| Total:{total:}| ETA:{eta:}| Loss:{loss:.4f}| top1:{top1:.2f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg)
        bar.next()
    bar.finish()
    return losses.avg, top1.avg

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

def test(model, test_loader, criterion):
    global best_acc
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(test_loader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            #targets = torch.argmax(targets, 1)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s| Total: {total:}| ETA: {eta:}| Loss:{loss:.4f}| top1: {top1:.2f}'.format(
                batch=batch_idx + 1,
                size=len(test_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg)
            bar.next()
    bar.finish()
    return losses.avg, top1.avg

start_epoch= 1
epochs = 5
lr = 0.005
momentum = 0.9
weight_decay = 5e-4

#for param in model_ft.parameters():
#    param.requires_grad = False
qtThresi = np.quantile(modelLayerSensitivity, 0.95)
layers=-1
frezList = {}#np.zeros(modelLayerSensitivity.shape)
for name, module in model_ft.named_modules():
        if 'running' not in name:
            layers = layers+1
            frezList[name]=False
            if modelLayerSensitivity[layers]<qtThresi:
                module.requires_grad = False
                frezList[name]=True
                #frezList[layers]=1

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_ft.fc.in_features
#model_ft.fc = nn.Linear(num_ftrs, 10)
model_ft.fc = nn.Linear(num_ftrs, 10)

#model_ft = torch.load('./Carmon2019Unlabeled_C10ftSVNH.pth')
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# NOTE: pytorch optimizer explicitly accepts parameter that requires grad
# see https://github.com/pytorch/pytorch/issues/679
optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.001, momentum=0.9)


# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

criterion =  torch.nn.CrossEntropyLoss(reduction="sum")
# We use a proxy model to calculate AWP, which does not affect the statistics of BN.
proxy  = copy.deepcopy(model_ft) 
#nn.DataParallel(getattr(modelarchs, arch)(num_classes=NUM_CLASSES)).to(device)
proxy_optim = optim.SGD(proxy.parameters(), lr=lr)
awp_adversary = TradesAWP(model=model_ft, proxy=proxy, proxy_optim=proxy_optim, gamma=awp_gamma)

robAccList = []        


lossArr = np.zeros((epochs+1,2))
accArr = np.zeros((epochs+1,2))
for epoch in range(start_epoch, epochs + 1):        
        dataloader = train_dl

    
        # adjust learning rate for SGD
        lr = adjust_learning_rate(optimizer_ft, epoch,lr)
        
        # adversarial training
        adv_loss, adv_acc = train1epoch(model_ft, dataloader, optimizer_ft, epoch, awp_adversary)

        # evaluation on natural examples
        print('================================================================')
        print('Iteration:',str(epoch))
        train_loss, train_acc = test(model_ft, dataloader, criterion)
        lossArr[epoch,0]=train_loss
        accArr[epoch,0]=train_acc
        print('Train loss Before Qt:',train_loss,'  Train Acc Before Qt:',train_acc)
        #val_loss, val_acc = test(model, test_loader, criterion)
        #print('================================================================')
        model_ft = model_ft#genStatedict0(qtConfig,hookList,model_ft,fwd_bits=8,bn_bits=32)
        # #logger.append([lr, adv_loss, train_loss, val_loss, adv_acc, train_acc, val_acc])
        # train_loss, train_acc = test(model_ft, dataloader, criterion)
        # print('Train loss After Qt:',train_loss,'  Train Acc After Qt:',train_acc)
        # lossArr[epoch,1]=train_loss
        # accArr[epoch,1]=train_acc
        # save checkpoint
        if epoch % save_freq == 0:
            #torch.save(model_ft.state_dict(),
            #           os.path.join(model_dir, 'ours-model-epoch{}.pt'.format(epoch)))
            #torch.save(optimizer_ft.state_dict(),
            #           os.path.join(model_dir, 'ours-opt-checkpoint_epoch{}.tar'.format(epoch)))
            modelsavname = modelname+'_C10ftSVNH_'+str(epoch)+'.pth'
            torch.save(model_ft, modelsavname)
            #model_ft = model_ft.load_state_dict(torch.load(model_dir+'/ours-model-epoch'+str(epoch)+'.pt'))
            # adversary = AutoAttack(model_ft, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
            # adversary.apgd.n_restarts = 1
            # x_adv = adversary.run_standard_evaluation(x_test, y_test,bs=128)




#torch.save(model_ft.state_dict(), 'model_current.pth')
modelsavname = modelname+'_C10ftSVNH.pth'
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
bs = 128
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
filename = 'accPTQonSvhnTransW8A8.npy'
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
filename = 'accQATonSvhnTransW8A8.npy'
np.save(filename,accQATonMixed)

#%%
accClean = getAccPth(model_ft=model,x_tst=x_dat,y_tst=y_dat,bs=bs)
print( 'Acc without attack:',str(accClean) )



#%%
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

#%%
x_dat = x_test[:2000]
y_dat = y_dat[:2000]
adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
adversary.apgd.n_restarts = 1
x_adv = adversary.run_standard_evaluation(x_dat, y_dat,bs=128)
