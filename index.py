from __future__ import division
import numpy as np
from sklearn import svm, datasets

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss


from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, voting_classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from pyearth import Earth

from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


f = open("./DataSets/CombinedNoDupe.csv")

data = np.loadtxt(f, delimiter=",")

X = data[:, 0:22]
y = data[:, 22]


#iris = datasets.load_iris()
#X = iris.data
#y = iris.target


names = ["AdaBoostedLogisticsRegression","LogisticRegression"]

boosted_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5))

classifiers = [
    AdaBoostClassifier(n_estimators=5000, learning_rate=0.05, base_estimator=LogisticRegression()),
    LogisticRegression()
]


testOut = []
scores = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
recalls = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0]
precisions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
logLosses = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
final = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

i = 0

while i < 10:
    for name, clf in zip(names, classifiers):
        train_feats, test_feats, train_labels, test_labels = tts(
            X, y, test_size=0.2)
        clf.fit(train_feats, train_labels)
        y_pred_class = clf.predict(test_feats)
        
        score = accuracy_score(test_labels, y_pred_class)
        recall = recall_score(test_labels, y_pred_class)
        precision = precision_score(test_labels, y_pred_class)

        scores[names.index(name)] = scores[names.index(name)] + score/10
        recalls[names.index(name)] = recalls[names.index(name)] + recall/10
        precisions[names.index(name)] = precisions[names.index(
            name)] + precision/10

        final[names.index(name)] = final[names.index(
            name)] + ((precision+score+recall)/30)
    print(str(i) + " ---"),
    i += 1
np.savetxt('pipe.out', [scores, recalls, precisions, final], fmt='%s')
