#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
np.random.seed(123)
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
 
# On charge un dataset MNIST en data de train et de test
(X_train, y_train), (X_test, y_test) = mnist.load_data()
 
# 5. On travaille sur les datas
#X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
 
# On change les labels 
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
 
# Ici on définit un modèle convolutionnel
model = Sequential()
 
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28, 1)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
 
# On compile ici le modèle
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
# On entraîne le modèle 
model.fit(X_train, Y_train, batch_size=32, verbose=1)
 
# On évalue le modèle sur les datasets
score = model.evaluate(X_test, Y_test, verbose=0)