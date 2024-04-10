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
This is the main entry point for MP9. You should only modify code within this file.
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
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):
        in_size -> h -> out_size , where  1 <= h <= 256
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.lrate = lrate
        h = 100
        self.neural = nn.Sequential(nn.Linear(in_size, h), nn.ReLU(), nn.Linear(h, out_size))
        self.optimize = optim.SGD(self.neural.parameters(), lr=lrate, momentum = 0.9)
    

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        return self.neural(x)

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        self.optimize.zero_grad()
        output = self.forward(x)
        loss = self.loss_fn(output, y)
        loss.backward()
        self.optimize.step()
        return loss.item()




def fit(train_set, train_labels, dev_set, epochs, batch_size=100):
    learn_rate = 1e-2
    loss_function = nn.CrossEntropyLoss()
    in_size = train_set.shape[1]
    out_size = 4
    net = NeuralNet(learn_rate, loss_function, in_size, out_size)

    train_mean = train_set.mean()
    train_std = train_set.std()
    train = (train_set - train_mean) / train_std
    dev = (dev_set - train_mean) / train_std

    training_set = get_dataset_from_arrays(train, train_labels)
    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 6}
    training_generator = DataLoader(training_set, **params)

    losses = [] 
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in training_generator:
            x, y = batch['features'], batch['labels']
            loss = net.step(x, y)
            epoch_loss += loss
        losses.append(epoch_loss / len(training_generator))

    output = net(dev).detach().cpu().numpy()
    yhats = np.argmax(output, axis=1).astype(int)
    
    return losses, yhats, net
