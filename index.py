from __future__ import division
import numpy as np
from sklearn import svm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt


f = open("./DataSets/InteractiveNarrative .csv")
g = open("./DataSets/InteractiveNarrativeGame.csv")

dataG = np.loadtxt(g, delimiter=",")
data = np.loadtxt(f, delimiter=",")

XG = data[:, 0:21]
yG = data[:, 21]

X = data[:, 0:21]
y = data[:, 21]

train_feats, test_feats, train_labels, test_labels = tts(X, y, test_size=0.2)

train_featsG, test_featsG, train_labelsG, test_labelsG = tts(XG, yG, test_size=0.2)

clf = svm.SVC(kernel='poly')
print "Using", clf
clf.fit(train_feats, train_labels)
predictions = clf.predict(test_feats)
print "\nPredictions:", predictions
Myprediction = clf.predict([[1,20,1,1,3,1,2,1,3,3,2,2,1,2,1,1,3,2,2,1,1]])


clfg = svm.SVC(kernel='poly')
print "Using", clfg
clfg.fit(train_feats, train_labels)
predictionsg = clfg.predict(test_feats)
print "\nPredictionsG:", predictionsg
Mypredictiong = clf.predict([[1,20,1,1,3,1,2,1,3,3,2,2,1,2,1,1,3,2,2,1,1]])


score = 0
for i in range(len(predictions)):
    if predictions[i] == test_labels[i]:
        score += 1

scoreg = 0
for i in range(len(predictionsg)):
    if predictionsg[i] == test_labelsG[i]:
        scoreg += 1
print "Accuracy:", (score / len(predictions))
print "Accuracy G :", (score / len(predictionsg))

print "\n my predictionG ", Mypredictiong
print "\n my prediction ", Myprediction