#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 16:40:46 2020

@author: xyzhou
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from utils import model_with_cfg

qtModelList = [ [1,1],[1,2],[2,2],[3,2],[3,3],[4,4],[8,8] ]
#qtModelList = [ [2,2],[4,4],[8,8] 
from CNV import CNV
#from CNVfp32 import CNVfp32
def loadBrevitasModel(Wb,Ab):
    in_bit_width = 8
    num_classes = 10 #cfg.getint('MODEL', 'NUM_CLASSES')
    in_channels = 3  #cfg.getint('MODEL', 'IN_CHANNELS')
    
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
        model_filename = 'bnn_models/cnv_w'+str(Wb)+'a'+str(Ab)+'.tar'
        print(model_filename)
        checkpoint = torch.load(model_filename)
        model.load_state_dict(checkpoint['state_dict'],strict=False)
       
    return model


def loadModels(path,qtModelList):
    models = []
    criterion = nn.CrossEntropyLoss()


    
    for modelCFG in qtModelList:
        Wb = modelCFG[0]
        Ab = modelCFG[1]
        model0 = loadBrevitasModel(Wb,Ab)
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
N_eps = 5+1
N_model = len(qtModelList)
accResults = np.zeros((N_eps,N_model))

modelidx = 0
for classifier in classifierSet:    
    predictions = classifier.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    accResults[0,modelidx] = accuracy
    modelidx = modelidx+1

#%% Step 5: Generate adversarial test examples
#from transfer_uap import TranferUniversalPerturbation
from art.attacks.evasion import UniversalPerturbation

N= 10000 #500
advMode = 'tr'
if advMode=='tr':
    x_dat = x_train[:N]
    y_dat = y_train[:N]    
elif advMode=='tst':
    x_dat = x_test[:N]
    y_dat = y_test[:N] 

epsidx = 0
modelidx = 0
N= len(x_test) #500
for epsDeg in [20]:#[4,8,16,26,32]:
    epsidx = epsidx+1
    advEps = epsDeg/255.0
    for classifier in classifierSet:
        modelCFG = qtModelList[modelidx]
        wb = modelCFG[0]
        ab = modelCFG[1]
        modeltag = 'w'+str(wb)+'a'+str(ab)
        print('Working on Model('+str(modelidx)+'):'+modeltag)
        
        
        #uap_attack = UniversalPerturbation(classifier=classifier, attacker='deepfool',eps=advEps, max_iter=10)
        uap_attack = UniversalPerturbation(
            classifier,
            attacker='fgsm',
            delta=0.000001,
            attacker_params={'eps':advEps/5},
            max_iter=10,
            eps=advEps)
        x_test_uap0 = uap_attack.generate(x=x_dat, y=y_dat)
        x_test_uap0 = x_test_uap0.astype('float32')
        x_noise_uap0 = uap_attack.noise
        
        datname = 'singleuqmodel/x_noise'+advMode+str(N)+'_'+modeltag+'_'+str(epsDeg)+'.npy'
        np.save(datname,x_noise_uap0)
        
        predictions = classifier.predict(x_test_uap0)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_dat, axis=1)) / len(y_test)
        print(datname)
        print("Accuracy on first N adv examples: {}%".format(accuracy * 100))
        accResults[epsidx,modelidx] = accuracy
        modelidx =modelidx +1
    modelidx = 0

datname = 'singleuqmodel/uapRawResults_'+advMode+'.npy'
np.save(datname,accResults)


#%%  Test on the whole test set
x_test_uapall = x_test.copy()
# Apply attack and clip
x_test_uapall = x_test_uapall + x_noise_uap0
x_test_uapall = np.clip(x_test_uapall, 0.0, 1.0)

print('Results on more test data:')

predictions = classifierSet[0].predict(x_test_uapall)
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
