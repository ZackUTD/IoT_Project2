# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 19:49:57 2020

@author: Raptor
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sklearn.preprocessing as sp
import plotly.graph_objects as go
import plotly.io as pio
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


plt.ion()
actual = []
predicted = []

loop = 20

while (loop > 0):
    
 
    a = random.randrange(3)
    
    if (a==0):
        p = random.choice([0,0,0,0,0,0,0,1,2])
    elif (a==1):
        p = random.choice([1,1,1,1,1,1,1,0,2])
    else:
        p = random.choice([2,2,2,2,2,2,2,0,1])
         
    
    actual.append(a)
    predicted.append(p)
    print ('Actual =    ', actual)
    print ('Predicted = ', predicted)
    
    results = confusion_matrix(actual, predicted)

    print ('Confusion Matrix :')
    print (results)
    print ('Accuracy Score :',accuracy_score(actual, predicted))
    print ('Report : ')
    print (classification_report(actual, predicted))
    
    loop -= 1
    time.sleep(1)