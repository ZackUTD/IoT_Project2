#!/usr/bin/env python3

"""
Author: Zack Oldham
Date: 03/19/2020

PyTorch Neural Network for learning model from offline data.
"""


import numpy as np
import pandas as pd 
import math, random
import torch
from influxdb import DataFrameClient as df_client
from FaultDetectNet import FaultDetectNet
from sklearn.metrics import accuracy_score
from pywt import downcoef


client = df_client(host='localhost', port=8086)
client.switch_database('team_3_test_offline')


MW = 0
SAMPLE_SIZE = 3000  # one input contains 3000 points -- this will be pre-processed with downcoef
BATCH_SIZE = 5  # 5 sets at a time
TRAIN_CUT = 70
TEST_CUT = 82


# Perform partial discrete wavelet transform on gs data to obtain approximate coefficients (condense sample size from 3000 to 30)
def preprocess(x_raw):
	gs = x_raw['gs'].to_numpy().astype(float).flatten()
	gs = downcoef('a', gs, 'db4', level=7)
	sr = float(x_raw['sr'][0])
	load = float(x_raw['load'][0])

	return np.append(gs, [sr, load])




# Retrieve a batch of data and preprocess it
def get_batch(cut):
	global MW
	X_batch = []
	Y_batch = []

	batch_end = MW + BATCH_SIZE

	while MW < batch_end and MW < cut:
		query = 'select gs, sr, load, label from bearing where mw = ' + str(MW)

		results = client.query(query)

		if results:
			
			x_sample = preprocess(results['bearing'])
			X_batch.append(x_sample)

			label = results['bearing']['label'][0]
			Y_batch.append(label)

			MW += 1
				
	X_batch = torch.tensor(X_batch).float()
	Y_batch = torch.tensor(Y_batch).long()
	return X_batch, Y_batch



# attach the contents of list Y to list X
def cons(X, Y):
	for y in Y:
		X.append(y)

	return X



# train offline classifier and save the model for later use
def main():
	global MW
	offline_nn = FaultDetectNet()

	while MW < TRAIN_CUT:	
		X_trn, Y_trn = get_batch(TRAIN_CUT)
		offline_nn.fit(X_trn, Y_trn)



	Y_pred = []
	Y_true = []

	while MW < TEST_CUT:
		X_tst, Y_tst = get_batch(TEST_CUT)
		Y_true = cons(Y_true, np.array(Y_tst).astype(int))
		offline_nn.eval()
		yp, co = offline_nn(X_tst, eval=True)
		Y_pred = cons(Y_pred, yp)

	

	Y_pred = np.array(Y_pred).flatten()

	print('Predicted Values:', Y_pred)
	print('True Values:', Y_true)



	# TODO: Run program with larger dataset to see if accuracy drops to more believeable level (but hopefully not terrible low)
	acc = accuracy_score(Y_true, Y_pred)
	print('Accuracy:', acc)


	torch.save({'model_state_dict': offline_nn.state_dict(), 'optimizer_state_dict': offline_nn.optimizer.state_dict(), 'loss': offline_nn.loss}, 'offline_nn.tar')

if __name__ == '__main__':
	main()





