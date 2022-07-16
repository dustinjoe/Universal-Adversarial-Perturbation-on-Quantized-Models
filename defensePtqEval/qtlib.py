#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 09:43:03 2021

@author: xyzhou
"""
#import argparse
#from utee import misc, selector
from utee import quant
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark =True
from collections import OrderedDict

import copy
import numpy as np
from art.estimators.classification import PyTorchClassifier
from art.utils import load_cifar10
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from art.attacks.evasion import DeepFool

import random
import sys,os
sys.path.append('/media/xyzhou/extDisk2t/_AAAI22/QtPreserveRob/zeroq')
sys.path.append('/media/xyzhou/extDisk2t/_AAAI22/QtPreserveRob/robustbench')
#from zeroq.utils import *
from zeroq.distill_data import getDistilData

# # Step 1: Load the  dataset
(x_train,y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
x_train = x_train.astype('float32')[:200]
y_train = y_train.astype('float32')[:200]
# x_val0 = x_train.transpose((0, 3, 1, 2)).astype(np.float32)
# y_val0 = y_train.transpose((0, 3, 1, 2)).astype(np.float32)
# x_test = x_test.astype('float32')[:1000]
# y_test = y_test.astype('float32')[:1000]

# ## Step 1a: Swap axes to PyTorch's NCHW format
x_train = x_train.transpose((0, 3, 1, 2)).astype(np.float32)
# x_test = x_test.transpose((0, 3, 1, 2)).astype(np.float32)

from robustbench.data import load_cifar10,load_cifar10train
x_val0, y_val0 = load_cifar10train(n_examples=200)
#x_val = torch.from_numpy(x_train[:32])
#y_val = torch.from_numpy( np.argmax(y_train[:32], axis=1) ) 
#x_val = x_val.cuda()
#y_val = y_val.cuda()
def artTorchModel(model0):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model0.parameters(), lr=0.01)
    min_pixel_value = 0.0
    max_pixel_value = 1.0
    classifier = PyTorchClassifier(
            model=model0,
            clip_values=(min_pixel_value, max_pixel_value),
            loss=criterion,
            optimizer=optimizer,
            input_shape=(3, 28, 28),
            nb_classes=10,
        )
    return classifier


from robustbench.utils import load_model
def genUAP(n_sample):
    model_raw = load_model(model_name='Carmon2019Unlabeled',  norm='Linf').cuda()
    #n_sample = 200
    x_val = x_train[:n_sample]
    y_val = y_train[:n_sample] 
    classifier = artTorchModel(model_raw)   
    attackerUAP = DeepFool(classifier,epsilon=8.0/255)
    x_val_adv = attackerUAP.generate(x=x_val, y=y_val)
    return x_val_adv
#x_val_adv = genUAP(200)
#np.save('x_valUAP200.npy',x_val_adv)
x_val_adv0 = np.load('x_valDeepFool200.npy')
x_val_adv0 = torch.from_numpy(x_val_adv0)
#x_val_adv0 = torch.from_numpy(x_val_adv0[:32]).cuda()

def getAcc(classifier,x_test,y_test):
    predictions = classifier.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    #print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    return accuracy

def getAccPth(model,x_test,y_test):
    predictions = model(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    #print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    return accuracy

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

import math
def modelPredBase(model,x):
    model.eval()
    x = x.cuda()
    with torch.no_grad():
        y = model(x)
    return y


def modelPredScore(model, x_orig, bs=32):
    model.eval()
    n_batches = math.ceil(x_orig.shape[0] / bs)
    num_ele = len(x_orig)
    num_class = 10
    predictions = torch.zeros((num_ele,num_class))
    for counter in range(n_batches):
        start = counter*bs
        end = min((counter + 1) * bs, x_orig.shape[0])
        x = x_orig[start:end].clone().cuda()
        #The torch pacy = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
        #outputbs = model(x)
        outputbs = modelPredBase(model,x)
        predictions[start:end] = outputbs
    return predictions.cuda()

def modelPredCls(model, x_orig, bs=16):
    model.eval()
    n_batches = math.ceil(x_orig.shape[0] / bs)
    num_ele = len(x_orig)    
    predictions = torch.zeros((num_ele,))
    for counter in range(n_batches):
        start = counter*bs
        end = min((counter + 1) * bs, x_orig.shape[0])
        x = x_orig[start:end].clone().cuda()
        #The torch pacy = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
        #outputbs = model(x)
        outputbs = modelPredBase(model,x)
        _, predictions[start:end] = torch.max(outputbs, 1)
    return predictions.long().cuda()


def customLossTRADES(model_qtbase_tst,x_val_tst,x_val_adv_tst,y_val_tst):
    batch_size = len(x_val_tst)
    x_val_tst = x_val_tst.cuda()
    x_val_adv_tst = x_val_adv_tst.cuda()
    criterion_kl = nn.KLDivLoss(size_average=False)
    
    predictionsadv = modelPredScore(model=model_qtbase_tst,x_orig=x_val_adv_tst)
    predictionsnor = modelPredScore(model=model_qtbase_tst,x_orig=x_val_tst)    
    
    #logits = model(x_val)
    lossnor = F.cross_entropy(predictionsnor, y_val_tst)
    lossadv = (1.0 / batch_size) * criterion_kl(F.log_softmax(predictionsadv, dim=1),
                                                    F.softmax(predictionsnor, dim=1))
    
    #lossdif = (1.0 / batch_size) * criterion_kl(x_val_tst,x_val_adv_tst)#/len(x_val)
    loss_valtmp =  lossnor + lossadv
                  
    return loss_valtmp

def customLossMART(model_qtbase_tst,x_val_tst,x_val_adv_tst,y_val_tst):
    batch_size = len(x_val_tst)
    x_val_tst = x_val_tst.cuda()
    x_val_adv_tst = x_val_adv_tst.cuda()
    criterion_kl = nn.KLDivLoss(reduction='none')
    
   
    logits = modelPredScore(model=model_qtbase_tst,x_orig=x_val_adv_tst)
    logits_adv = modelPredScore(model=model_qtbase_tst,x_orig=x_val_tst) 
    
    adv_probs = F.softmax(logits_adv, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y_val_tst, tmp1[:, -2], tmp1[:, -1])
    loss_adv = F.cross_entropy(logits_adv, y_val_tst) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(logits, dim=1)
    true_probs = torch.gather(nat_probs, 1, (y_val_tst.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(criterion_kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    beta=6.0 
    loss_valtmp = loss_adv + float(beta) * loss_robust


                  
    return loss_valtmp


def customLoss(model_qtbase,x_val,x_val_adv,y_val):
    predictionsadv = modelPredScore(model=model_qtbase,x_orig=x_val_adv)
    #predictionsadv = modelPredBase(model=model_qtbase,x=x_val_adv)
    #predictionsadv = model_qtbase(x_val_adv.cuda())                
    #predictionsadv = modelpred(model_qtbase, x_val_adv, bs=16)
    predictionsnor = modelPredScore(model=model_qtbase,x_orig=x_val)
    #predictionsnor = modelPredBase(model=model_qtbase,x=x_val)
    #predictionsnor = model_qtbase(x_val.cuda())
    #predictionsnor = modelpred(model_qtbase, x_val, bs=16)    
    lossnor = nn.functional.cross_entropy(predictionsnor,y_val)
    lossadv = nn.functional.cross_entropy(predictionsadv,y_val)
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    lossdif = criterion_kl(x_val,x_val_adv)#/len(x_val)
    loss_valtmp =  lossnor*0.2 + lossadv*0.6 + lossdif*0.2
                  
    return loss_valtmp

def genValAdv0(model_raw,x_val,y_val,eps=8.0/255,bs=32,verbose=False,log_path='./result.txt'):
    adversary = AutoAttack(model_raw, norm='Linf', eps=eps, version='custom',verbose=verbose,
                           log_path=log_path,
                           attacks_to_run=['apgd-ce', 'apgd-dlr'])
    adversary.apgd.n_restarts = 1
    #print('Generating Adversarial Validation Data.')
    x_val_adv = adversary.run_standard_evaluation(x_val, y_val,bs=32).cuda() 
    return x_val_adv

def genValAdv(model_raw,x_val0,y_val0,eps=8.0/255,bs=32,verbose=False,log_path='./result.txt',modelname='model'):
    xadvfilename = 'x_valDpfool64_'+modelname+'.npy'
    loadDat = True
    if os.path.exists(xadvfilename) and loadDat==True:
        x_val_adv = np.load(xadvfilename)
    else:
        x_val = x_val0.cpu().detach().numpy()
        y_val = y_val0.cpu().detach().numpy()
        classifier = artTorchModel(model_raw)   
        attacker = DeepFool(classifier,epsilon=eps,batch_size=bs)
        x_val_adv = attacker.generate(x=x_val, y=y_val)        
        np.save('x_valDpfool64_'+modelname+'.npy',x_val_adv)
    return torch.from_numpy(x_val_adv).cuda()

def concatDatList(dataloaderList):
    numbs = len(dataloaderList)
    dataloader = []
    xbs0 = dataloaderList[0]
    for i in range(1,numbs):
        xbs0 = torch.cat((xbs0, dataloaderList[i]), 0)
    
    dataloader.append(xbs0)
    return dataloader
    


from autoattack.autoattack import AutoAttack    
import random
from torchvision import transforms

def saveTensor2Npy(dataTensor,filename):
    dataloaderNpy = copy.deepcopy(dataTensor)
    dataloaderNpy = dataloaderNpy.cpu().detach().numpy()
    np.save(filename,dataloaderNpy)

def qtLinearRndMixedDF(model_raw,modelname,param_bitsList,fwd_bits=8,bn_bits=32,overflow_rate=0.0,n_sample=64,saveqt=False,numIterMP = 1):
    assert torch.cuda.is_available(), 'no cuda' 
    
    #state_dict = model_raw.state_dict()
    
    tensorFileDistilled = './models/cifar10/Linf/'+modelname+'_distilled256.npy'
    if os.path.exists(tensorFileDistilled):
        x_val = torch.from_numpy( np.load(tensorFileDistilled) )
        x_val = x_val[:n_sample]
        print('******'+str(len(x_val))+' Precomputed Distilled Data loaded ******')  
    else:
        # Generate distilled data
        dataloader = getDistilData(
            model_raw.cuda(),
            dataset='cifar10',
            batch_size=32,num_batch=8,
            for_inception=False)
        print('******Distilled Data generated/loaded ******')  
        dataloader = concatDatList(dataloader)
        x_val = dataloader[0]
        x_val = x_val[:n_sample]
        saveTensor2Npy(x_val,tensorFileDistilled)
    # dataloader[0].shape: torch.Size([32, 3, 32, 32])
    # dataloader[0][0].shape: torch.Size([3, 32, 32])
    #normalize12 = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
    #                                 std=(0.2023, 0.1994, 0.2010))
    #transform_test = transforms.Compose([transforms.ToTensor(), normalize]) 
    
    normalizefunc = transforms.Normalize(mean=(0.50, 0.4822, 0.4465),
                                       std=(0.2023, 0.1994, 0.2010))
    
    xmax = torch.max(x_val)
    xmin = torch.min(x_val)
    x_val = (x_val - xmin)/(xmax - xmin)
    #x_val = normalizefunc(x_val)    
    y_val = modelPredCls(model_raw, x_val, bs=32)
    
    #classifier = artTorchModel(model_raw)  
    #x_val = x_train[:n_sample]
    #y_val = y_train[:n_sample]    
    #attackerUAP = DeepFool(classifier,epsilon=8.0/255)
    #x_val_adv = attackerUAP.generate(x=x_val, y=y_val)    /home/xyzhou/anaconda3/lib/python3.7/site-packages/torch/nn/functional.py:2611: UserWarning: reduction: 'mean' divides the total loss by both the batch size and the support size.'batchmean' divides only by the batch size, and aligns with the KL div math definition.'mean' will be changed to behave the same as 'batchmean' in the next major release.


    

    
    modelstatedic = model_raw.state_dict()
    modelstatlist = list(modelstatedic)
    #numlayer = len(modelstatlist)
    random.shuffle(modelstatlist)
    
    

    

    #wbit_len = 0
    model_qtbase = copy.deepcopy(model_raw)    
    state_dict_quant = model_qtbase.state_dict()

    '''    
    # param_bit list should store target quant weights from high to low
    # init all quant weights to the highest possible quant weight first here
    param_bits = param_bitsList[0]
    for k, v in state_dict_quant.items():
        if 'running' in k:
            if  bn_bits >=32:
                #print("Ignoring {}".format(k))
                state_dict_quant[k] = v
                continue
            else:
                bits =  bn_bits
        else:
            bits =  param_bits
            #bits = random.choice(param_bitsList)

        sf = bits - 1. - quant.compute_integral_part(v, overflow_rate= overflow_rate)
        v_quant  = quant.linear_quantize(v, sf, bits=bits)

        state_dict_quant[k] = v_quant
        #print(k, bits)
    '''
    autoInit = False    
    if autoInit == True:
        x_val_adv = genValAdv(model_raw,x_val,y_val,modelname=modelname)
        abit = 32
        lossList = []        
        for wbit in param_bitsList:             
            model_qtbase = qtLinear(model_raw,modelname,wbit,abit)            
            loss_valtmp = customLossTRADES(model_qtbase,x_val,x_val_adv,y_val)
            lossList.append(loss_valtmp)
        initBit = param_bitsList[ lossList.index( min(lossList) ) ]        
        model_qtbase = qtLinear(model_raw,modelname,param_bits=initBit,fwd_bits=abit)
        print('Init Weight Bitwidth:',str(initBit))
    else:
        print('Default Init Quantization: w8a8')
        model_qtbase = qtLinear(model_raw,modelname,param_bits=8,fwd_bits=32)
    model_qtbase.load_state_dict(state_dict_quant)  
    
    x_val_adv = copy.deepcopy(x_val)
    wbit_sum = 0
    wbit_list = np.ones(len(modelstatlist))*(-1)
    numFlip = 0    
    freqFlipAdv = 1
    flagFlip = True
    freqPrintLoss = 5
    l_beta = 0.98
    #x_val_adv = genValAdv(model_qtbase,x_val_adv,y_val,eps=8.0/255)
    #loss_val0 = customLossTRADES(model_qtbase,x_val,x_val_adv,y_val)
    for i in range(numIterMP):
        #random.shuffle(modelstatlist)
        x_val_adv = genValAdv0(model_qtbase,x_val,y_val,eps=8.0/255)
        loss_val0 = customLossTRADES(model_qtbase,x_val,x_val_adv,y_val)
        print('#Iter ',str(i),' Custom Loss:',str(loss_val0),' #Layer Flip:',str(numFlip))
        
        
        for k in modelstatlist:            
            v = modelstatedic[k]
            flagFlip = False
            if 'running' in k:
                if  bn_bits >=32:
                    #print("Ignoring {}".format(k))
                    state_dict_quant[k] = v
                    continue
                else:
                    bits =  bn_bits
            else:
        
                #loss_val0 = np.inf
                wbit_best = param_bitsList[0]
                wbit_list[modelstatlist.index(k)] = wbit_best
                for bits in param_bitsList:
                    sf = bits - 1. - quant.compute_integral_part(v, overflow_rate= overflow_rate)
                    v_quant  = quant.linear_quantize(v, sf, bits=bits)
    
                    state_dict_quant[k] = v_quant
                    #print(k, bits)(
                    model_qtbase.load_state_dict(state_dict_quant)
                    #classifier = artTorchModel(model_qtbase) 
                    loss_valtmp = customLossTRADES(model_qtbase,x_val,x_val_adv,y_val)
                    if loss_valtmp<loss_val0*l_beta:
                        loss_val0 = loss_valtmp
                        wbit_best = bits
                        wbit_list[modelstatlist.index(k)] = bits
                        numFlip = numFlip+1
                        flagFlip = True

                    
                sf = wbit_best - 1. - quant.compute_integral_part(v, overflow_rate= overflow_rate)
                v_quant  = quant.linear_quantize(v, sf, bits=wbit_best)
                state_dict_quant[k] = v_quant   
                model_qtbase.load_state_dict(state_dict_quant)

                if numFlip%freqFlipAdv==0 and flagFlip==True:
                    x_val_adv = genValAdv0(model_qtbase,x_val,y_val,eps=8.0/255)
                    #print(x_val.shape)
                    #print(x_val_adv.shape)
                    #print(y_val.shape)
                    loss_val0 = customLossTRADES(model_qtbase,x_val,x_val_adv,y_val)
                    if numFlip%freqPrintLoss==0:
                        print('__#Iter ',str(i),' Custom Loss:',str(loss_val0),' #Layer Flip:',str(numFlip))

        #wbit_sum = wbit_sum+wbit_best
    print('#Iter ',str(i+1),' Final Loss:',str(loss_val0),' #Layer Flip:',str(numFlip))
        
    wbit_list = wbit_list[wbit_list>0]
    wbit_sum = np.sum(wbit_list)
    wbit_len = len(wbit_list)
    print('WBit Sum:',str(wbit_sum))
    print('WBit Mean:',str(float(wbit_sum)/wbit_len) )
    model_qtbase.load_state_dict(state_dict_quant)
    
    # quantize forward activation
    #loss_val0 = np.inf
    #abit_best = fwd_bits

    # quantize forward activation
    if  fwd_bits < 32:
        model_qtbase = quant.duplicate_model_with_quant(model_qtbase, bits= fwd_bits, overflow_rate= overflow_rate,
                                                     counter= n_sample, type= 'linear')           
    #classifier = artTorchModel(model_raw)
    #acc1 = getAcc(classifier,x_test,y_test)    
    #if acc1>0.80:
    #    print("QT Accuracy on benign test examples: {}%".format(acc1 * 100))
    #    #print('QT Acc>0.80~')
    if saveqt==True:
        filename = modelname+'_linearMP_DF.pth'
        torch.save(model_qtbase, filename)
    

    # print sf
    # sf decide the least precision, bits decide the range. So if sf=2 and bits=8, it means the least precision is 0.25, and range is [-127, 128] * 0.25
    #print(model_raw)
    return model_qtbase


def qtLinearRndMixed(model_raw,modelname,param_bitsList,
                     fwd_bits=8,bn_bits=32,overflow_rate=0.0,n_sample=32,saveqt=False,numIterMP = 1):
    assert torch.cuda.is_available(), 'no cuda' 
    
    #state_dict = model_raw.state_dict()


    
    #classifier = artTorchModel(model_raw)  
    #x_val = x_train[:n_sample]
    #y_val = y_train[:n_sample]    
    #attackerUAP = DeepFool(classifier,epsilon=8.0/255)
    #x_val_adv = attackerUAP.generate(x=x_val, y=y_val)    /home/xyzhou/anaconda3/lib/python3.7/site-packages/torch/nn/functional.py:2611: UserWarning: reduction: 'mean' divides the total loss by both the batch size and the support size.'batchmean' divides only by the batch size, and aligns with the KL div math definition.'mean' will be changed to behave the same as 'batchmean' in the next major release.
    x_val = x_val0.cuda()
    y_val = y_val0.cuda()
    #x_val_adv0 = genValAdv(model_raw,x_val0,y_val0,modelname=modelname,eps=2.0/255)
    # x_val_idx = torch.randint(200, (32,))
    # #x_val_idx = np.random.choice(200, 32, replace=False)
    # x_val = x_val0[x_val_idx]
    # #x_val = torch.from_numpy(x_val32).cuda()
    # #x_val_adv = x_val_adv0[x_val_idx]
    # #x_val_adv = torch.from_numpy(x_val_adv32).cuda()
    # y_val = y_val0[x_val_idx]
    # y_val = y_val.cuda()
    

    
    modelstatedic = model_raw.state_dict()
    modelstatlist = list(modelstatedic)
    #numlayer = len(modelstatlist)
    random.shuffle(modelstatlist)
    
    

    

    #wbit_len = 0
    model_qtbase = copy.deepcopy(model_raw)    
    state_dict_quant = model_qtbase.state_dict()


    autoInit = False    
    if autoInit == True:
        abit = 8
        x_val_adv = genValAdv0(model_raw,x_val,y_val,eps=8.0/255)
        lossList = []        
        for wbit in param_bitsList:            
            model_qtbase = qtLinear(model_raw,modelname,wbit,abit)            
            loss_valtmp = customLoss(model_qtbase,x_val0,x_val_adv0,y_val0)
            lossList.append(loss_valtmp)
        initBit = param_bitsList[ lossList.index( min(lossList) ) ]        
        model_qtbase = qtLinear(model_raw,modelname,param_bits=initBit,fwd_bits=abit)
        print('Init Weight Bitwidth:',str(initBit))
    else:
        print('Default Init Quantization: w8a8')
        model_qtbase = qtLinear(model_raw,modelname,param_bits=8,fwd_bits=8)
    model_qtbase.load_state_dict(state_dict_quant) 
    
    x_val_adv = x_val
    wbit_sum = 0
    wbit_list = np.ones(len(modelstatlist))*(-1)
    for i in range(numIterMP):
        print('Iteration ',str(i))
        random.shuffle(modelstatlist)
        x_val_adv = genValAdv0(model_raw,x_val_adv,y_val,eps=2.0/255)
        for k in modelstatlist:            
            v = modelstatedic[k]
            if 'running' in k:
                if  bn_bits >=32:
                    #print("Ignoring {}".format(k))
                    state_dict_quant[k] = v
                    continue
                else:
                    bits =  bn_bits
            else:
                # x_val_idx = torch.randint(200, (64,))
                # #x_val_idx = np.random.choice(200, 32, replace=False)
                # x_val = x_val0[x_val_idx].cuda()
                # #x_val = torch.from_numpy(x_val32).cuda()
                # x_val_adv = x_val_adv0[x_val_idx].cuda()
                # #x_val_adv = torch.from_numpy(x_val_adv32).cuda()
                # y_val = y_val0[x_val_idx]
                # y_val = y_val.cuda()
                
                loss_val0 = customLoss(model_qtbase,x_val,x_val_adv,y_val)
                #loss_val0 = np.inf
                wbit_best = param_bitsList[0]
                for bits in param_bitsList:
                    sf = bits - 1. - quant.compute_integral_part(v, overflow_rate= overflow_rate)
                    v_quant  = quant.linear_quantize(v, sf, bits=bits)
    
                    state_dict_quant[k] = v_quant
                    #print(k, bits)
                    model_qtbase.load_state_dict(state_dict_quant)
                    #classifier = artTorchModel(model_qtbase) 
                    loss_valtmp = customLoss(model_qtbase,x_val,x_val_adv,y_val)
                    if loss_valtmp<loss_val0:
                        loss_val0 = loss_valtmp
                        wbit_best = bits
                        wbit_list[modelstatlist.index(k)] = bits
                    
                sf = wbit_best - 1. - quant.compute_integral_part(v, overflow_rate= overflow_rate)
                v_quant  = quant.linear_quantize(v, sf, bits=wbit_best)
                state_dict_quant[k] = v_quant   
                model_qtbase.load_state_dict(state_dict_quant)


        #wbit_sum = wbit_sum+wbit_best
    wbit_list = wbit_list[wbit_list>0]
    wbit_sum = np.sum(wbit_list)
    wbit_len = len(wbit_list)
    print('WBit Sum:',str(wbit_sum))
    print('WBit Mean:',str(float(wbit_sum)/wbit_len) )
    model_qtbase.load_state_dict(state_dict_quant)

    # quantize forward activation
    #loss_val0 = np.inf
    #abit_best = fwd_bits

    # quantize forward activation
    if  fwd_bits < 32:
        model_qtbase = quant.duplicate_model_with_quant(model_qtbase, bits= fwd_bits, overflow_rate= overflow_rate,
                                                     counter= n_sample, type= 'linear')           
    #classifier = artTorchModel(model_raw)
    #acc1 = getAcc(classifier,x_test,y_test)    
    #if acc1>0.80:
    #    print("QT Accuracy on benign test examples: {}%".format(acc1 * 100))
    #    #print('QT Acc>0.80~')
    if saveqt==True:
        filename = modelname+'_linearMP_DF.pth'
        torch.save(model_qtbase, filename)
    

    # print sf
    # sf decide the least precision, bits decide the range. So if sf=2 and bits=8, it means the least precision is 0.25, and range is [-127, 128] * 0.25
    #print(model_raw)
    return model_qtbase



def qtLinearRndMixed0(model_raw,modelname,param_bitsList,fwd_bits=8,bn_bits=32,overflow_rate=0.0,n_sample=32,saveqt=False):
    assert torch.cuda.is_available(), 'no cuda' 
    
    #state_dict = model_raw.state_dict()
    
    
    #classifier = artTorchModel(model_raw)  
    #x_val = x_train[:n_sample]
    #y_val = y_train[:n_sample]    
    #attackerUAP = DeepFool(classifier,epsilon=8.0/255)
    #x_val_adv = attackerUAP.generate(x=x_val, y=y_val)
    
    #adversary = AutoAttack(model_raw, norm='Linf', eps=8/255, version='custom',
    #                       attacks_to_run=['apgd-ce', 'apgd-dlr'])
    #adversary.apgd.n_restarts = 1
    #x_val_adv = adversary.run_standard_evaluation(x_val, y_val,bs=32).cuda()    
    #predictions0 = classifier.predict(x_val_adv)
    #loss_val = nn.functional.cross_entropy(predictions0,y_val)
    #torch.nn.functional.kl_div
    
    modelstatedic = model_raw.state_dict()
    modelstatlist = list(modelstatedic)
    #numlayer = len(modelstatlist)
    random.shuffle(modelstatlist)
    
    

    

    #wbit_len = 0
    model_qtbase = copy.deepcopy(model_raw)
    state_dict_quant = model_qtbase.state_dict()    
    # param_bit list should store target quant weights from high to low
    # init all quant weights to the highest possible quant weight first here
    param_bits = param_bitsList[0]
    for k, v in state_dict_quant.items():
        if 'running' in k:
            if  bn_bits >=32:
                #print("Ignoring {}".format(k))
                state_dict_quant[k] = v
                continue
            else:
                bits =  bn_bits
        else:
            bits =  param_bits
            #bits = random.choice(param_bitsList)

        sf = bits - 1. - quant.compute_integral_part(v, overflow_rate= overflow_rate)
        v_quant  = quant.linear_quantize(v, sf, bits=bits)

        state_dict_quant[k] = v_quant
        #print(k, bits)
    model_qtbase.load_state_dict(state_dict_quant)    
    wbit_sum = 0
    wbit_list = np.ones(len(modelstatlist))*(-1)
    for i in range(5):
        for k in modelstatlist:
            random.shuffle(modelstatlist)
            v = modelstatedic[k]
            if 'running' in k:
                if  bn_bits >=32:
                    #print("Ignoring {}".format(k))
                    state_dict_quant[k] = v
                    continue
                else:
                    bits =  bn_bits
            else:
                #wbit_len = wbit_len+1
                x_val_idx = torch.randint(200, (32,))
                #x_val_idx = np.random.choice(200, 32, replace=False)
                x_val = x_val0[x_val_idx]
                #x_val = torch.from_numpy(x_val32).cuda()
                x_val_adv = x_val_adv0[x_val_idx]
                #x_val_adv = torch.from_numpy(x_val_adv32).cuda()
                y_val = y_val0[x_val_idx]
                y_val = y_val.cuda()
                #y_val = torch.from_numpy(y_val32).cuda()
                predictionsadv = model_qtbase(x_val_adv.cuda())                
                #predictionsadv = modelpred(model_qtbase, x_val_adv, bs=16)
                predictionsnor = model_qtbase(x_val.cuda())
                #predictionsnor = modelpred(model_qtbase, x_val, bs=16)
                loss_val0 = nn.functional.cross_entropy(predictionsadv,y_val) +\
                              nn.functional.cross_entropy(predictionsnor,y_val)
                #loss_val0 = np.inf
                wbit_best = param_bitsList[0]
                for bits in param_bitsList:
                    sf = bits - 1. - quant.compute_integral_part(v, overflow_rate= overflow_rate)
                    v_quant  = quant.linear_quantize(v, sf, bits=bits)
    
                    state_dict_quant[k] = v_quant
                    #print(k, bits)
                    model_qtbase.load_state_dict(state_dict_quant)
                    #classifier = artTorchModel(model_qtbase) 
                    predictionsadv = model_qtbase(x_val_adv.cuda())                
                    #predictionsadv = modelpred(model_qtbase, x_val_adv, bs=16)
                    predictionsnor = model_qtbase(x_val.cuda())
                    #predictionsnor = modelpred(model_qtbase, x_val, bs=16)
                    loss_valtmp = nn.functional.cross_entropy(predictionsadv,y_val) +\
                                  nn.functional.cross_entropy(predictionsnor,y_val)
                    if loss_valtmp<loss_val0:
                        loss_val0 = loss_valtmp
                        wbit_best = bits
                        wbit_list[modelstatlist.index(k)] = bits
                    
                sf = wbit_best - 1. - quant.compute_integral_part(v, overflow_rate= overflow_rate)
                v_quant  = quant.linear_quantize(v, sf, bits=wbit_best)
                state_dict_quant[k] = v_quant   
                model_qtbase.load_state_dict(state_dict_quant)


        #wbit_sum = wbit_sum+wbit_best
    wbit_list = wbit_list[wbit_list>0]
    wbit_sum = np.sum(wbit_list)
    wbit_len = len(wbit_list)
    print('WBit Sum:',str(wbit_sum))
    print('WBit Mean:',str(float(wbit_sum)/wbit_len) )
    model_qtbase.load_state_dict(state_dict_quant)

    # quantize forward activation
    #loss_val0 = np.inf
    #abit_best = fwd_bits

    # quantize forward activation
    if  fwd_bits < 32:
        model_qtbase = quant.duplicate_model_with_quant(model_qtbase, bits= fwd_bits, overflow_rate= overflow_rate,
                                                     counter= n_sample, type= 'linear')           
    #classifier = artTorchModel(model_raw)
    #acc1 = getAcc(classifier,x_test,y_test)    
    #if acc1>0.80:
    #    print("QT Accuracy on benign test examples: {}%".format(acc1 * 100))
    #    #print('QT Acc>0.80~')
    if saveqt==True:
        filename = modelname+'_linear_best.pth'
        torch.save(model_qtbase, filename)
    

    # print sf
    # sf decide the least precision, bits decide the range. So if sf=2 and bits=8, it means the least precision is 0.25, and range is [-127, 128] * 0.25
    #print(model_raw)
    return model_qtbase

# this original version aim to find optimal wb and ab together but does not seem that reliable
# the output optimal activation bits always 8 among [8,7,6]
def qtLinearRndMixed0(model_raw,modelname,param_bitsList,fwd_bitsList,bn_bits=32,overflow_rate=0.0,n_sample=32,saveqt=False):
    assert torch.cuda.is_available(), 'no cuda' 
    
    #state_dict = model_raw.state_dict()
    
    
    #classifier = artTorchModel(model_raw)  
    #x_val = x_train[:n_sample]
    #y_val = y_train[:n_sample]    
    #attackerUAP = DeepFool(classifier,epsilon=8.0/255)
    #x_val_adv = attackerUAP.generate(x=x_val, y=y_val)
    
    adversary = AutoAttack(model_raw, norm='Linf', eps=8/255, version='custom',
                           attacks_to_run=['apgd-ce', 'apgd-dlr'])
    adversary.apgd.n_restarts = 1
    x_val_adv = adversary.run_standard_evaluation(x_val, y_val,bs=32).cuda()    
    #predictions0 = classifier.predict(x_val_adv)
    #loss_val = nn.functional.cross_entropy(predictions0,y_val)
    #torch.nn.functional.kl_div
    
    modelstatedic = model_raw.state_dict()
    modelstatlist = list(modelstatedic)
    #numlayer = len(modelstatlist)
    random.shuffle(modelstatlist)
    
    
    wbit_sum = 0
    wbit_len = 0
    model_qtbase = copy.deepcopy(model_raw)
    state_dict_quant = model_qtbase.state_dict()
    for k in modelstatlist:
        v = modelstatedic[k]
        if 'running' in k:
            if  bn_bits >=32:
                #print("Ignoring {}".format(k))
                state_dict_quant[k] = v
                continue
            else:
                bits =  bn_bits
        else:
            wbit_len = wbit_len+1
            loss_val0 = np.inf
            wbit_best = 32
            for bits in param_bitsList:
                sf = bits - 1. - quant.compute_integral_part(v, overflow_rate= overflow_rate)
                v_quant  = quant.linear_quantize(v, sf, bits=bits)

                state_dict_quant[k] = v_quant
                #print(k, bits)
                model_qtbase.load_state_dict(state_dict_quant)
                #classifier = artTorchModel(model_qtbase) 
                predictionstmp = model_qtbase(x_val_adv)
                loss_valtmp = nn.functional.cross_entropy(predictionstmp,y_val)
                if loss_valtmp<loss_val0:
                    loss_val0 = loss_valtmp
                    wbit_best = bits
                
            sf = wbit_best - 1. - quant.compute_integral_part(v, overflow_rate= overflow_rate)
            v_quant  = quant.linear_quantize(v, sf, bits=wbit_best)
            state_dict_quant[k] = v_quant       

        wbit_sum = wbit_sum+wbit_best
        
    print('WBit Sum:',str(wbit_sum))
    print('WBit Mean:',str(float(wbit_sum)/wbit_len) )
    model_qtbase.load_state_dict(state_dict_quant)

    # quantize forward activation
    loss_val0 = np.inf
    abit_best = 32

    for  fwd_bits in fwd_bitsList:
        #model_qtbase2 = model_qtbase.load_state_dict(state_dict_quant)
        model_qtbase2 = copy.deepcopy(model_qtbase)
        model_qtbase2 = quant.duplicate_model_with_quant(model_qtbase2, bits= fwd_bits, overflow_rate= overflow_rate,
                                                     counter= n_sample, type= 'linear')
        #classifier = artTorchModel(model_qtbase2) 
        predictionstmp = model_qtbase2(x_val_adv)
        loss_valtmp = nn.functional.cross_entropy(predictionstmp,y_val)
        if loss_valtmp<loss_val0:
            loss_val0 = loss_valtmp
            abit_best = fwd_bits
        #print(model_raw)
    print('Best Activation Bitwdith:',str(abit_best))    
    model_qtbase = quant.duplicate_model_with_quant(model_qtbase, bits= abit_best, overflow_rate= overflow_rate,
                                                     counter= n_sample, type= 'linear')
           
    #classifier = artTorchModel(model_raw)
    #acc1 = getAcc(classifier,x_test,y_test)    
    #if acc1>0.80:
    #    print("QT Accuracy on benign test examples: {}%".format(acc1 * 100))
    #    #print('QT Acc>0.80~')
    if saveqt==True:
        filename = modelname+'_linear_best.pth'
        torch.save(model_qtbase, filename)
    

    # print sf
    # sf decide the least precision, bits decide the range. So if sf=2 and bits=8, it means the least precision is 0.25, and range is [-127, 128] * 0.25
    #print(model_raw)
    return model_qtbase