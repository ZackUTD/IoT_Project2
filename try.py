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
RTW = 2000
t = 300

prev_count_DW = 0
prev_count_RTW = 0


def getModel():
    checkpoint = torch.load('offline_nn (1).tar')
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
    client1.switch_database('team_3_test_d')
    return client1


def getData(pc, W, client):
    select_query = 'select * from bearing where time >' + str(pc) + ' and time <=' + str(pc + W)
    data = client.query(select_query)
    data = data['bearing']
    return data


def preprocess(x_raw):
    gs = x_raw['gs'].to_numpy().astype(float).flatten()
    gs = downcoef('a', gs, 'db4', level=7)
    sr = float(x_raw['sr'][0])
    load = float(x_raw['load'][0])

    return np.append(gs, [sr, load])

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
class MySeriesHelper(SeriesHelper):
    """Instantiate SeriesHelper to write points to the backend."""

    class Meta:
        """Meta class stores time series helper configuration."""

        # The client should be an instance of InfluxDBClient.
        client = client1

        # The series name must be a string. Add dependent fields/tags
        # in curly brackets.
        series_name = 'bearing'

        # Defines all the fields in this time series.
        fields = ['y_pred', 'y_true']

        # Defines all the tags for the series.

        # Defines the number of data points to store prior to writing
        # on the wire.
        bulk_size = 1

        # autocommit must be set to True when using bulk_size
        autocommit = True


# get start time
start = datetime.datetime.now()
model = getModel()
X = pd.DataFrame()
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
            #model = getModel()

            X = X.append(data[['gs', 'load', 'sr']])

            y = data['label']

            y = np.array(y).astype(int)

            #y_t = torch.tensor(np.array(y), dtype=torch.long)
        if X.shape[0] == 3000:
            model.eval()
            print(X.shape)
            X = preprocess(X)
            X = X.astype(dtype=float)
            X = torch.FloatTensor(np.array(X))
            Y_pred = model(X, eval=True)

            #Y_pred = np.array(Y_pred).flatten()
            #print('Predicted Values:', Y_pred)
            #print('True Values:', y)


            #acc = accuracy_score(y, Y_pred)
            #print('Accuracy:', acc)
            #print('getting data')
            #for i in range(len(y)):
                #MySeriesHelper(server_name='bearing', y_pred=y[i], y_true=y[i])
            #MySeriesHelper.commit()


            # TODO: evaluate model
            # TODO: write results to DB

        if count - prev_count_RTW >= RTW:
            data = getData(prev_count_RTW, RTW, client)
            prev_count_RTW += RTW
            #model = getModel()
            y = data['label']
            X = X.append(data[['gs', 'load', 'sr']])

        print(X.shape)
        if X.shape[0] == 3000:
            print(X.shape)
            X = preprocess(X)
            X = X.astype(dtype=float)
            X = torch.FloatTensor(np.array(X))
            X = X.view(1, 32)
            y = data['label'][0]

            y = torch.tensor(np.array(y), dtype=torch.long)
            y = y.view(1)
            print(X)
            model.fit(X, y)
            X = pd.DataFrame()
            # TODO: retrain model
        start = datetime.datetime.now()
