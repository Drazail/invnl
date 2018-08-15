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
    

# loading in data
f = open("./DataSets/CombinedNoDupe.csv")
p = open("./Preds/predictionsSets.csv")

data = np.loadtxt(f, delimiter=",")
pred_feats=(np.loadtxt(p, delimiter=","))

X = data[:, 0:22]
y = data[:, 22]

predictionsSets = pred_feats[:,0:22]

# constructing classifiers

names = ["AdaBoostedLogisticsRegression", "LogisticRegression","Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net",
          "QDA"]

boosted_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5))

classifiers = [
    AdaBoostClassifier(n_estimators=1000, learning_rate=0.5,
                       base_estimator=LogisticRegression()),
    LogisticRegression(),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    QuadraticDiscriminantAnalysis()
]


# instantiating variables
testOut = []
scores = [0 for i in xrange(10)]
recalls  = [0 for i in xrange(10)]
precisions  = [0 for i in xrange(10)]
logLosses  = [0 for i in xrange(10)]
final  = [0 for i in xrange(10)]
predictionRate =  [x[:] for x in [[0] * 10] * 10]
i = 0

# test and predict loop
while i < 100:
    for name, clf in zip(names, classifiers):
        train_feats, test_feats, train_labels, test_labels = tts(
            X, y, test_size=0.2)
        clf.fit(train_feats, train_labels)
        y_pred_class = clf.predict(test_feats)

        score = accuracy_score(test_labels, y_pred_class)
        recall = recall_score(test_labels, y_pred_class)
        precision = precision_score(test_labels, y_pred_class)
        prediction = clf.predict(pred_feats)


        scores[names.index(name)] = scores[names.index(name)] + score/100

        recalls[names.index(name)] = recalls[names.index(name)] + recall/100

        precisions[names.index(name)] = precisions[names.index(
            name)] + precision/100

        final[names.index(name)] = final[names.index(
            name)] + (precision+score+recall)/300

        for j in range(len(predictionsSets)):
            predictionRate[names.index(name)][j] = predictionRate[names.index(name)][j] +  prediction[j]/100

    print(str(i) + " ---"),
    i += 1

np.savetxt('pipe.out', [scores, recalls, precisions, final],fmt='%1.3f', header=str(names))
np.savetxt('pred.out', predictionRate, fmt='%1.3f')