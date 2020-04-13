import torch
from influxdb import InfluxDBClient
from influxdb import SeriesHelper
from torch import nn
from torch import optim
from Offline import FaultDetectNet
import numpy as np
import pandas as pd

checkpoint = torch.load('Offline/offline_nn.tar')
model = FaultDetectNet.FaultDetectNet()
model.load_state_dict(checkpoint['model_state_dict'])
model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.loss = checkpoint['loss']
with open('prev_count.txt') as txt_file:
        prev_count = int(txt_file.readline())
client1 = InfluxDBClient(host=u'localhost', port=8086)  # client that reads/writes db with pandas dataframes
client1.switch_database('team_3_test_online')
count_query = 'select count(load) from bearing'
results = client1.query(count_query)
count = results.raw['series'][0]['values'][0][1]
RTW = 1000
if count - prev_count >= RTW:
        select_query = 'select * from bearing where time >' + str(prev_count) + ' and time <=' + str(prev_count+RTW)
        prev_count += RTW
        data = client1.query(select_query).raw['series'][0]

with open('prev_count.txt', 'w') as txt_file:
        txt_file.write(str(prev_count))
