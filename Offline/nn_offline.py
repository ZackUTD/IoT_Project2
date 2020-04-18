#!/usr/bin/env python3

"""
Author: Zack Oldham
Date: 03/19/2020

PyTorch Neural Network for learning model from offline data.
"""


import numpy as np
import pandas as pd 
import math, random
from datetime import datetime
import torch
from influxdb import DataFrameClient as df_client
from FaultDetectNet import FaultDetectNet
from sklearn.metrics import accuracy_score
from pywt import downcoef
import sys


client = df_client(host='localhost', port=8086)
client.switch_database('team_3_offline')


DEBUG = False
MW_MAX = 0
MW = 0
SAMPLE_SIZE = 3000  # one input contains 3000 points -- this will be pre-processed with downcoef
BATCH_SIZE = 10  # 10 samples at a time
TRAIN_CUT = 0
TEST_CUT = MW_MAX


# Perform partial discrete wavelet transform on gs data to obtain approximate coefficients (condense sample size from 3000 to 30)
def preprocess(x_raw):
    gs = x_raw['gs'].to_numpy().astype(float).flatten()
    gs = downcoef('a', gs, 'db4', level=7)
    load = float(x_raw['load'][0])/300
    return np.append(gs, load)




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
    global MW, DEBUG, MW_MAX, TRAIN_CUT 
       
    if len(sys.argv) < 2:
        print('Usage: python3 nn_offline.py -<train | debug> <max mw>')
    elif sys.argv[1] == '-debug':
        DEBUG = True
    elif sys.argv[1] == '-train':
        DEBUG = False
    else:
        print('ERR: Invalid argument; options are -debug or -train')    
    
    
    MW_MAX = int(sys.argv[2])

    offline_nn = FaultDetectNet()

    if DEBUG:
        TRAIN_CUT = int(0.8 * MW_MAX)
        print('Training...')
        while MW < TRAIN_CUT:
            print('MW:', MW)
            X_trn, Y_trn = get_batch(TRAIN_CUT)
            offline_nn.fit(X_trn, Y_trn)


        Y_pred = []
        Y_true = []

        print('Testing...')
        while MW < TEST_CUT:
            print('MW:', MW)
            X_tst, Y_tst = get_batch(TEST_CUT)
            Y_true = cons(Y_true, np.array(Y_tst).astype(int))
            offline_nn.eval()
            yp, conf = offline_nn(X_tst, eval=True)
            Y_pred = cons(Y_pred, yp)
         
         
        Y_pred = np.array(Y_pred).flatten().astype(int)

        print('Predicted Values:', Y_pred)
        print('True Values:', Y_true)


        acc = accuracy_score(Y_true, Y_pred)
        print('Accuracy: {0:4.4f}'.format(acc))




    else:
        TRAIN_CUT = MW_MAX

        while MW < TRAIN_CUT:
            print('MW:', MW)
            X_trn, Y_trn = get_batch(TRAIN_CUT)
            offline_nn.fit(X_trn, Y_trn)

        torch.save({'model_state_dict': offline_nn.state_dict(), 'optimizer_state_dict': offline_nn.optimizer.state_dict(), 'loss': offline_nn.loss}, 'offline_nn.tar')


if __name__ == '__main__':
	main()





