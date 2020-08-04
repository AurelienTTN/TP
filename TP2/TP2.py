#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 12:12:29 2020

@author: aurelien
"""

import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import pandas as pd

dataframe = pd.read_csv("Iris.csv")
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4] 

# On change les classes en entiers 

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# 
dummy_y = np_utils.to_categorical(encoded_Y)

# Cette fonction permet de définir un modèle
def baseline_model():
	# on crée un modèle composé de deux layers
	model = Sequential()
	model.add(Dense(8, input_dim=4, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# On le compile puis on le retourne 
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# # On crée un KerasClassifier qui a pour architecture le baseline_model défini
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)

# # On utilise un k-fold pour évaluer notre modèle 

kfold = KFold(n_splits=10, shuffle=True)

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
#On affiche ici les moyennes des résultats 
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

