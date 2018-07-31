from __future__ import division
import numpy as np
from sklearn import svm, datasets

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score


from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import matplotlib.pyplot as plt


f = open("./DataSets/INnoDup.csv")

data = np.loadtxt(f, delimiter=",")

X = data[:, 0:21]
y = data[:, 21]


#iris = datasets.load_iris()
#X = iris.data
#y = iris.target




names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]



testOut = []
scores = [0,0,0,0,0,0,0,0,0,0]
final= [0,0,0,0,0,0,0,0,0,0]
j=0
while j < 100 :
    i = 0
    while i < 1000 :
        for name, clf in zip(names, classifiers):
                train_feats, test_feats, train_labels, test_labels = tts(X, y, test_size=0.2)
                clf.fit(train_feats, train_labels)
                score = clf.score(test_feats, test_labels)
                scores [names.index(name)] = scores [names.index(name)] + score/1000
                final [names.index(name)] = final [names.index(name)] + score/1000
        i+=1
    testOut.append(scores)
    scores=[0,0,0,0,0,0,0,0,0,0]
    j+=1
final[:] = [x / 100 for x in final]
testOut.append("final")
testOut.append(final)
np.savetxt('testNoDup.out', testOut, fmt='%s')