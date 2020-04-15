import torch
from influxdb import DataFrameClient as df_client
from influxdb import SeriesHelper
from torch import nn
from torch import optim
from offline1 import FaultDetectNet
import numpy as np
import time
import pandas as pd
import os
import datetime

DW = 50
RTW = 1000
t = 300

prev_count_DW = 0
prev_count_RTW = 0


def getModel():
    checkpoint = torch.load('offline_nn.tar')
    model = FaultDetectNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.loss = checkpoint['loss']
    return model


def connect_to_db():
    client1 = df_client(host=u'localhost', port=8086)  # client that reads/writes db with pandas dataframes
    client1.switch_database('team_3_test_online')
    return client1


def getData(pc, W, client):
    select_query = 'select * from bearing where time >' + str(pc) + ' and time <=' + str(pc + W)
    data = client.query(select_query)
    data = data['bearing']
    return data


def getCount(client):
    count_query = 'select count(load) from bearing'
    results = client.query(count_query)
    count = results['bearing']['count'][0]
    return count


# connect to DB
client = connect_to_db()

# get start time
start = datetime.datetime.now()

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
            model = getModel()
            X = data[['gs', 'load', 'sr']]
            y = data['label']
            X = torch.tensor(np.array(X), dtype=torch.float)
            y = torch.tensor(np.array(y), dtype=torch.long)
            model.eval()
            # TODO: evaluate model
            # TODO: write results to DB

        if count - prev_count_RTW >= RTW:
            data = getData(prev_count_RTW, RTW, client)
            prev_count_RTW += RTW
            X = data[['gs', 'load', 'sr']]
            y = data['label']
            X = torch.tensor(np.array(X), dtype=torch.float)
            y = torch.tensor(np.array(y), dtype=torch.long)
            # TODO: retrain model

            # save model(not sure. You'll have to see the code where he saves the model)
            model.save('offline_nn.tar')
            # this should over write the saved model locally
        start = datetime.datetime.now()
