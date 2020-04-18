#!/usr/bin/env python3

"""
Author: Zack Oldham
Date: 03/22/2020

Pytorch neural network for fault detection
"""


import numpy as np
import torch
from torch import nn
from torch import optim


class FaultDetectNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(31, 16)  # hidden layer 1
        self.fc2 = nn.Linear(16, 8)   # hidden layer 2
        self.fc3 = nn.Linear(8, 3)  # output layer

        self.relu = nn.ReLU()  # ReLU activation function
        self.softmax = nn.Softmax(dim=1)  # Softmax activation function

        self.optimizer = optim.Adam(self.parameters()) 
        self.loss_fn = nn.CrossEntropyLoss() 
        self.loss = None


    # define how an input shall progress through the neural net
    def forward(self, x, eval=False):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))

        if eval:
            return self.from_dist(x)
        else:
            return x


    # convert a set of probability distributions to class labels.
    # return the classs label with the greatest probability, along with that probability (our confidence of its correctness)
    # for each sample in x
    def from_dist(self, x):
        out = []
        conf = []
        for row in x:
            maxidx = row.argmax().item()
            out.append(maxidx)
            conf.append(row[maxidx].item())

        return out, conf


    # train the classifier on a batch of inputs
    def fit(self, data, labels, epoch=50):
        for i in range(epoch):
            self.train()
            for j in range(len(data)):
                self.optimizer.zero_grad()
                ypred = self(data, eval=False)
                self.loss = self.loss_fn(ypred, labels)
                self.loss.backward()
                self.optimizer.step()


