# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP10. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        super(NeuralNet, self).__init__()

        self.loss_fn = loss_fn
        self.lrate = lrate
        hidden_size = 128

        self.conv1 = nn.Sequential(nn.Conv2d(3, 24, kernel_size=3), nn.LeakyReLU())
        self.pool1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=3))

        self.neural = nn.Sequential(
            nn.Linear(1944, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, out_size)
        )

        self.optimizer = optim.SGD(self.neural.parameters(), lr=lrate, momentum=0.9, weight_decay=1e-3)
    

    def forward(self, x):
        x = self.conv1(x.view(-1, 3, 31, 31))
        x = self.pool1(x)
        x = torch.flatten(x, 1)
        return self.neural(x)

    def step(self, x, y):
        self.optimizer.zero_grad()
        output = self.forward(x)
        loss = self.loss_fn(output, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()




def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    learn_rate = 0.005
    loss_function = nn.CrossEntropyLoss()
    in_size = train_set.shape[1]
    out_size = 4

    losses = []
    yhats = np.zeros(len(dev_set))
    net = NeuralNet(learn_rate, loss_function, in_size, out_size)

    train = (train_set - train_set.mean()) / train_set.std()
    params = {'batch_size': batch_size,
            'shuffle': True,
            'num_workers': 1}

    training_set = get_dataset_from_arrays(train, train_labels)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    for epoch in range(epochs):
        for local_batch, local_labels in enumerate(training_generator):
            losses.append(net.step(local_labels['features'], local_labels['labels']))

    dev = (dev_set - dev_set.mean()) / dev_set.std()
    output = net(dev).detach().cpu().numpy()

    for i in range(len(output)):
        yhats[i] = np.argmax(output[i])

    return losses, yhats.astype(int), net
