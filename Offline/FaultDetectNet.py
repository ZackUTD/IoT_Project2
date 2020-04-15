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
    # instantiate neural net with input size of sample_len (sample_size * 3 (gs, sr, load for each point))
    # and output size of 3 (probability distribution over class labels 0,1,2)
    def __init__(self):
        super().__init__()
        self.f1 = nn.Linear(32, 16)  # linear input layer with 3 inputs to each of three neurons
        self.f2 = nn.Linear(16, 8)  # 1st hidden linear layer with 3 inputs to each of three neurons
        self.ft1 = nn.Linear(8, 8)
        self.ft2 = nn.Linear(8, 8)
        self.ft3 = nn.Linear(8, 8)
        self.ft4 = nn.Linear(8, 8)
        self.f3 = nn.Linear(8, 4)  # 2nd "..."
        self.f4 = nn.Linear(4, 3)  # output layer with linear function taking 3 inputs and 3 output
        self.softmax = nn.Softmax(dim=1) # Softmax activation function used at output layer
        self.sigm = nn.Sigmoid()  # Sigmoid activation function used at input layer and each hidden layer
        
        #self.optimizer = optim.SGD(self.parameters(), lr=1e-5, weight_decay=0.01, momentum=0.9, nesterov=True)  # define the optimization algorithm to be used
        self.optimizer = optim.Adam(self.parameters(), lr=1e-5, weight_decay=0.01)
        self.loss_fn = nn.CrossEntropyLoss()  # loss function to be used
        self.loss = None  # placeholder for the current value of loss (needed for saving the model)
        self.apply(self.init_weights)  # initialize the weights in order to prevent underfitting



    # initialize the model's weights using the xavier uniform distribution for deep neural network
    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)


    # define how an input passes through the neural net
    def forward(self, x, eval=False):
        x = self.sigm(self.f1(x))  # apply linear transformation to inputs (WX + b), and feed result to relu activation function
        x = self.sigm(self.f2(x))  # "..."
        x = self.sigm(self.ft1(x))
        x = self.sigm(self.ft2(x))
        x = self.sigm(self.ft3(x))
        x = self.sigm(self.ft4(x))
        x = self.sigm(self.f3(x))  # "..."
        x = self.softmax(self.f4(x))  # apply linear transformation inputs, and feed result to softmax activation function to obtain probability distribution

        if eval:
            return self.from_dist(x)
        else:
            return x



    # convert list of probability distributions to list of labels
    def from_dist(self, dist):
        labels = []

        for row in dist:
            labels.append(row.argmax().item())

        return labels



    # convert list of single element labels to list of probability distributions
    def to_dist(self, labels):
        dist = []


        for y in labels:
            if y == 0:
                dist.append([1.0, 0.0, 0.0])
            elif y == 1:
                dist.append([0.0, 1.0, 0.0])
            else:
                dist.append([0.0, 0.0, 1.0])

        return torch.tensor(dist).float()



    # train classifier on a **batch-size** chunk of readings
    # data: a set of **batch-size** readings, each is 1D, **sample-size*3** long (3 attributes per point in sample)
    # labels: array-like structre (list, numpy, etc) containing 1D list of all training labels
    # epoch: number of times to train on a particular sample, default is 20
    def fit(self, data, labels, epoch=50):
        # epoch is how many times we should run the given set of data through the neural net for this learning cycle.
        for i in range(epoch):
            self.train()  # signify that we are in training mode
            for j in range(len(data)):  # We expect data to be a batch of training examples
                self.optimizer.zero_grad()  # zero the gradient for this fresh data point
                Y_pred = self(data, eval=False)
                self.loss = self.loss_fn(Y_pred, labels)  # how wrong were we? (the smaller this gets the better)
                #print('loss:', self.loss)
                self.loss.backward()  # perform backward propagation to adjust weights
                self.optimizer.step()  # take the next gradient descent 'step' (hopefully nearer to the bottom of the 'canyon')



