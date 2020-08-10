# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

dataset = loadtxt('diabetes2.csv',delimiter=',')
X = dataset[:,0:8]
y = dataset [:,8]

##On définit ici notre modèle composé de trois layers quipossèdent respctivement 12,8 et 1 noeuds
##Les deux premiers utilisent une fonction d'activation relu et le dernier une fonction sigmoïd

model = Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))

#Pour changer le nombre de sortie on met deux noeuds au dernier layer
model.add(Dense(2,activation='softmax'))

#On peut maintenant compiler notre modèle
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#On entraîne notre modèle sur notre dataset en fixant
#une epoque de 150, l'entraînement va parcourir 150 fois nos données 
#le batch-size corrspond aux nombres d'exemples pris en compte avant de mettre à jours les poids
model.fit(X, y, epochs=150, batch_size=10)

# On peut maintenant evaluer notre modèle 

_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

