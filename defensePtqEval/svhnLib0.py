#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 09:16:29 2021

@author: xyzhou
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import BasicBlock

import time
import numpy as np


class MeanValue:

    def __init__(self):
        self.value = 0
        self.counter = 0
    
    def add(self, value):
        self.value += value
        self.counter += 1
    
    def get(self):
        return self.value / self.counter
    
    def reset(self):
        self.value = 0
        self.counter = 0


class Accuracy:

    def __init__(self):
        self.n_correct = 0.
        self.n_sample = 0.
    
    def add(self, predicts, labels):
        self.n_sample += len(labels)
        self.n_correct += np.sum(predicts == labels)
    
    def get(self):
        return self.n_correct / self.n_sample
    
    def reset(self):
        self.n_correct = 0.
        self.n_sample = 0.


class TimeMeter:

    def __init__(self):
        self.start_time, self.duration, self.counter = 0., 0., 0.

    def add_counter(self):
        self.counter += 1

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.duration += time.perf_counter() - self.start_time

    def get(self):
        return self.duration / self.counter

    def reset(self):
        self.start_time, self.duration, self.counter = 0., 0., 0.

class Resnet20(nn.Module):

    def __init__(self, in_shape, n_class, *args, **kwargs):
        super(Resnet20, self).__init__(*args, **kwargs)
        self.in_shape = in_shape

        self.conv1 = nn.Conv2d(in_shape[0], 16, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.in_planes = 16

        block = BasicBlock
        self.layer1 = self._make_layers(block, 16, 2, 1)
        self.layer2 = self._make_layers(block, 32, 2, 2)
        self.layer3 = self._make_layers(block, 64, 2, 2)

        # compute the height of layer3's feature map
        fh = in_shape[1]
        for _ in range(2):
            fh = math.ceil(fh / 2)
        self.fh = int(fh)

        self.fc = nn.Linear(64, n_class)
    
    def _make_layers(self, block, planes, n_block, stride=1):
        layers = []

        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers.append(block(self.in_planes, planes, stride, downsample))
        for _ in range(1, n_block):
            layers.append(block(planes, planes))

        self.in_planes = planes
        return nn.Sequential(*layers)
    

    def forward(self, images):
        out = self.conv1(images)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, self.fh)

        shape = out.shape
        out = torch.reshape(out, (shape[0], -1))
        out = self.fc(out)
        return out
        

class SimpleCNN(nn.Module):

    def __init__(self, in_shape, n_class, *args, **kwargs):
        super(SimpleCNN, self).__init__(*args, **kwargs)
        self.in_shape = in_shape

        self.conv1 = nn.Conv2d(in_shape[0], 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

        fh = in_shape[1]
        for _ in range(3):
            fh = math.ceil(fh / 2)
        fh = int(fh)
        self.fc = nn.Linear(fh * fh * 128, n_class)
    
    def forward(self, images):
        out = self.conv1(images)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = F.relu(out)
        shape = out.shape
        out = torch.reshape(out, (shape[0], -1))
        out = self.fc(out)
        return out
    
    

def getModelSvhn():

    # build model
    model = Resnet20(in_shape=(3,32,32),n_class=10).cuda()
    #meta = {'n_class': 10}

    #net = model( (32,32,3) , meta['n_class']).cuda()
    #criterion = torch.nn.CrossEntropyLoss()

    state = torch.load('./models/svhn/checkpoint_9.pk')
    model.load_state_dict(state['net'])

    model.eval()
    return model