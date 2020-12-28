# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 22:45:48 2020

@author: shara
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

zoo = pd.read_csv("G:\DS Assignments\K-NN\Zoo.csv")
x = zoo.iloc[:,1:17]
y = zoo.iloc[:,17]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

neigh = KNC(n_neighbors=3)
neigh.fit(x_train, y_train)
train_acc = np.mean(neigh.predict(x_train) == y_train)
test_acc = np.mean(neigh.predict(x_test) == y_test)

acc = []
for i in range(1,50,1):
    neigh = KNC(n_neighbors= i)
    neigh.fit(x_train, y_train)
    train_acc = np.mean(neigh.predict(x_train) == y_train)
    test_acc = np.mean(neigh.predict(x_test) == y_test)
    acc.append([train_acc,test_acc])
    

