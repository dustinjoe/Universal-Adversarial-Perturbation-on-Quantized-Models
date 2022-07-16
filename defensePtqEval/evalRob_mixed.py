# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 09:28:09 2020

@author: xyzhou
"""
#from robustbench.data import load_cifar10
#x_test, y_test = load_cifar10(n_examples=50)


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

import numpy as np

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.utils import load_cifar10

from robustbench.utils import load_model
model = load_model(model_name='Carmon2019Unlabeled', norm='Linf')

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()

# Step 1a: Swap axes to PyTorch's NCHW format
x_train = x_train.transpose((0, 3, 1, 2)).astype(np.float32)
x_test = x_test.transpose((0, 3, 1, 2)).astype(np.float32)

# Step 3: Create the ART classifier
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


# Step 4: Evaluate the ART classifier on benign test examples
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

#%% Step 6: Generate adversarial test examples
from art.attacks.evasion import FastGradientMethod,ProjectedGradientDescent

advEps = 8/255
attack = FastGradientMethod(estimator=classifier, eps=advEps)
#attack = ProjectedGradientDescent(estimator=classifier, eps=advEps, eps_step=advEps/20,max_iter=40)
x_test_adv = attack.generate(x=x_test)

# Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
