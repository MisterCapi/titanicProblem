# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 19:54:39 2020

@author: jmarchewka
"""
#imports
#tens 
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
#network
import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras import regularizers
from keras.regularizers import l2
#data
import numpy as np
import pandas as pd
#sklearn
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.callbacks import ModelCheckpoint
#imports

dataset=pd.read_csv('data.csv')
dataset.head();
dataset.drop(['Cabin','Name','Ticket','Embarked'], 1, inplace=True)

dataset.fillna({'Age':dataset["Age"].mean()}, inplace=True)
dataset.fillna({'Embarked':'S'}, inplace=True)
dataset['Sex']=np.where(dataset['Sex']=='male',1,0)

X=dataset.iloc[:,2:12].values
sc = StandardScaler()
X = sc.fit_transform(X)
Y=dataset.iloc[:,1].values

from keras.utils import to_categorical
Y = to_categorical(Y)

model = Sequential()
model.add(Dense(30, activation='relu' ))
model.add(Dense(15, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(Adam(learning_rate = 0.003), loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(X, Y, epochs=800, batch_size=10)
print(model.evaluate(X, Y))