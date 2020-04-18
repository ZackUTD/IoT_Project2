import torch
from influxdb import DataFrameClient as df_client
from influxdb import SeriesHelper
from torch import nn
from torch import optim
from FaultDetectNet import FaultDetectNet
import numpy as np
import time
import pandas as pd
import os
import datetime
from sklearn.metrics import accuracy_score
from pywt import downcoef

DW = 1000
RTW = 1000
BW = 2
BATCH_SIZE = BW * RTW
t = 300

prev_count_DW = 0
prev_count_RTW = 0


def getModel():
    checkpoint = torch.load('offline_nn (2).tar')
    model = FaultDetectNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.loss = checkpoint['loss']
    return model


def connect_to_db():
    client1 = df_client(host=u'localhost', port=8086)  # client that reads/writes db with pandas dataframes
    client1.switch_database('team_3_test_online')
    return client1
def connect_to_db2():
    client1 = df_client(host=u'localhost', port=8086)  # client that reads/writes db with pandas dataframes
    client1.switch_database('team_3_test_detect')
    return client1


def getData(pc, W, client):
    select_query = 'select * from bearing where time >' + str(pc) + ' and time <=' + str(pc + W)
    data = client.query(select_query)
    data = data['bearing']
    return data


def preprocess(x_raw):
    gs = x_raw['gs'].to_numpy().astype(float).flatten()
    gs = downcoef('a', gs, 'db4', level=7)  
    load = float(x_raw['load'][0])/300  # normalize the load value

    return np.append(gs, load)

def cons(X, Y):
    for y in Y:
        X.append(y)
    return X


def getCount(client):
    count_query = 'select count(load) from bearing'
    results = client.query(count_query)
    count = results['bearing']['count'][0]
    return count


# connect to DB
client = connect_to_db()
client1 = connect_to_db2()


# get start time
start = datetime.datetime.now()
model = getModel()
X = pd.DataFrame()
Y_train = []
batch = torch.tensor([])
batch_count = 0
train_counter = 0
while True:
    # get current time
    now = datetime.datetime.now()
    # if t seconds has passed do everything
    if (now - start).seconds > 0:
        # get count from DB
        count = getCount(client)

        if count - prev_count_DW >= DW:
            data = getData(prev_count_DW, DW, client)

            prev_count_DW += DW
            # model = getModel()

            X = X.append(data[['gs', 'load', 'sr']])

            y = [data['label'][0]]

            y = np.array(y, dtype=int)

            # y_t = torch.tensor(np.array(y), dtype=torch.long)
        if X.shape[0] == 3000:
            model.eval()
            print(X.shape)
            X = preprocess(X)
            X = X.astype(dtype=float)
            X = torch.FloatTensor(np.array(X))
            X = X.view(1, 32)
            Y_pred, Conf = model(X, eval=True)
            X = pd.DataFrame()
            print('input tested...')

            print('confidence:', Conf)

            Y_pred = np.array(Y_pred).flatten()
            print('Predicted Values:', Y_pred)
            print('True Values:', y)

            acc = accuracy_score([y], Y_pred)
            print('Accuracy:', acc)
            results_dict = dict()
            results_dict['actual'] = list(y)
            results_dict['predicted'] = Y_pred
            results_dict['confidence'] = Conf[0]
            results_dict['retrain'] = train_counter
            results_dict['time'] = datetime.datetime.now()
            print(prev_count_DW)
            results_dict['time_stop'] = prev_count_DW + 1000
            results_dict['time_start'] = prev_count_DW - 2000
            results_df = pd.DataFrame(results_dict)
            results_df = results_df.set_index('time')
            client1.write_points(results_df, measurement='detect')


        y = None
        if count - prev_count_RTW >= RTW:
            print('getting data to train...')
            data = getData(prev_count_RTW, RTW, client)
            prev_count_RTW += RTW
            # model = getModel()
            y = data['label'][0]
            X = X.append(data[['gs', 'load', 'sr']])

        if X.shape[0] == 3000:
            print('batch retrieved...')
            X = preprocess(X)
            X = X.astype(dtype=float)
            X = torch.FloatTensor(np.array(X))
            X = X.view(1, 32)
            batch = torch.cat((batch, X), dim=0)
            Y_train.append(y)
            X = pd.DataFrame()
            batch_count += 1

        if batch_count == BW:  # when we have full batch -- then train
            print('training...')
            Y_train = torch.tensor(Y_train).long()
            print(Y_train)
            model.fit(batch, Y_train)
            batch_count = 0
            train_counter += 1


        start = datetime.datetime.now()
