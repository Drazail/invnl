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

train_feats, test_feats, train_labels, test_labels = tts(X, y, test_size=0.2)


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


for name, clf in zip(names, classifiers):
        clf.fit(train_feats, train_labels)
        score = clf.score(test_feats, test_labels)
        print name," : ", score