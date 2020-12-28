# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 20:12:29 2020

@author: shara
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
glass = pd.read_csv("F:\Warun\DS Assignments\DS Assignments\K-NN\glass.csv")

glass.Type.value_counts()
sns.heatmap(glass.corr())
glass.head()
sns.scatterplot(glass['RI'],glass['Na'],hue=glass['Type'])
sns.scatterplot(glass['Na'],glass['Mg'],hue=glass['Type'])
sns.scatterplot(glass['Mg'],glass['Al'],hue=glass['Type'])
sns.scatterplot(glass['Al'],glass['Si'],hue=glass['Type'])
sns.scatterplot(glass['Si'],glass['K'],hue=glass['Type'])
sns.scatterplot(glass['K'],glass['Ca'],hue=glass['Type'])
sns.scatterplot(glass['Ca'],glass['Ba'],hue=glass['Type'])
sns.scatterplot(glass['Ba'],glass['Fe'],hue=glass['Type'])




sns.pairplot(glass,hue='Type')
plt.show()

x = glass.iloc[: , 0:9]
y = glass.iloc[:, 9]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
x_train_scaled = preprocessing.scale(x_train)
x_test_scaled = preprocessing.scale(x_test)

neigh = KNC(n_neighbors=3)
neigh.fit(x_train_scaled, y_train)
train_acc = np.mean(neigh.predict(x_train_scaled) == y_train)
test_acc = np.mean(neigh.predict(x_test_scaled) == y_test)

# neigh_adv = KNC(algorithm='auto', leaf_size=30, metric='minkowski',
#            metric_params=None, n_jobs=1, n_neighbors=3, p=2,
#            weights='uniform')

# neigh_adv.fit(x_train_scaled, y_train)
# train_acc = np.mean(neigh_adv.predict(x_train_scaled) == y_train)
# test_acc = np.mean(neigh_adv.predict(x_test_scaled) == y_test)

acc = []
for i in range(3,50,1):
    neigh = KNC(n_neighbors= i)
    neigh.fit(x_train_scaled, y_train)
    train_acc = np.mean(neigh.predict(x_train_scaled) == y_train)
    test_acc = np.mean(neigh.predict(x_test_scaled) == y_test)
    acc.append([train_acc,test_acc])
    
plt.plot(np.arange(3,50,1), [i[0] for i in acc], "bo-")
plt.plot(np.arange(3,50,1), [i[1] for i in acc], "bo-")
