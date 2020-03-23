#!/usr/bin/env python3

"""
Author: Zack Oldham
Date: 03/22/2020

Pytorch neural network for fault detection
"""


import torch
from torch import nn
from torch import optim


class FaultDetectNet(nn.Module):
	def __init__(self):
 		super().__init__()
 		self.f1 = nn.Linear(3, 3)  # linear input layer with 3 inputs to each of three neurons
 		self.f2 = nn.Linear(3, 3)  # 1st hidden linear layer with 3 inputs to each of three neurons
 		self.f3 = nn.Linear(3, 3)  # 2nd "..."
 		self.f4 = nn.Linear(3, 1)  # output layer with three inputs to softmax function
 		self.sig_act = nn.Sigmoid()  # sigmoid activation function used at input layer and each hidden layer

 		self.optimizer = optim.SGD(self.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)  # define the optimization algorithm to be used
 		self.loss_fn = nn.L1Loss()  # loss function to be used
 		self.loss = None  # placeholder for the current value of loss (needed for saving the model)
	
 	# return the index of the maximum value in the tensor -- this represents its most likely class label
	def argmax_class(self, x):
		c = []
		for row in x:
			max_val = torch.max(row)
			idx_tensor = (row == max_val).nonzero()
			c.append(idx_tensor.float())


		c = torch.cat(c,dim=0)
		return c 

			

	def forward(self, x):
		x = self.sig_act(self.f1(x))  # apply linear transformation to inputs (WX + b), and feed result to sigmoid activation function
		x = self.sig_act(self.f2(x))  # "..."
		x = self.sig_act(self.f3(x))  # "..."
		x = self.f4(x)  # softmax generates prob distribution over the three labels (0, 1, 2) --> select label with max probability
		#print(x)
		#c = self.argmax_class(x) # determine the class from the probability distribution
		return x
		


	def fit(self, data, labels, epoch=20):
		# epoch is how many times we should run the given set of data through the neural net for this learning cycle.
		for i in range(epoch):
			self.train()  # signify that we are in training mode
			for j in range(len(data)):  # We expect data to be a batch of training examples
				self.optimizer.zero_grad()  # zero the gradient for this fresh data point
				Y_pred = self(data)
				self.loss = self.loss_fn(Y_pred, labels)  # how wrong were we? (the smaller this gets the better)
				self.loss.backward()  # perform backward propagation to adjust weights
				self.optimizer.step()  # take the next gradient descent 'step' (hopefully nearer to the bottom of the 'canyon')



