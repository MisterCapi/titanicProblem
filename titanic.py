# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 19:54:39 2020

@author: jmarchewka
"""
#imports
#tens
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
#network
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
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

from tensorflow.keras.utils import to_categorical
Y = to_categorical(Y)

model = Sequential()
model.add(Dense(30, activation='relu' ))
model.add(Dense(15, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(Adam(learning_rate = 0.003), loss = 'categorical_crossentropy', metrics = ['accuracy'])
model_saver = ModelCheckpoint('saved_model', monitor='accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
model.fit(X, Y, epochs=800, batch_size=10, callbacks=[model_saver])
print(model.evaluate(X, Y))
