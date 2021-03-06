import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn import tree


numOfSplits = 4

data = pd.read_csv("flare.csv")

data = data.values
features = data[:,0:32]
classification = data[:, 32]
classification = classification.astype(bool)

totalAccuracy = 0
totalConfusion = np.zeros((2,2))

# run kfold on trees
kf = KFold(n_splits=numOfSplits, shuffle=True)
for train_index, test_index in kf.split(features):
    # split the data to train set and validation set:
    features_train, features_test = features[train_index], features[test_index]
    classification_train, classification_test = classification[train_index], classification[test_index]

    # train the tree on train set
    estimator = tree.DecisionTreeClassifier(criterion="entropy")
    estimator.fit(features_train, classification_train)
    # test the tree on validation set
    totalAccuracy += accuracy_score(classification_test, estimator.predict(features_test))
    totalConfusion += confusion_matrix(classification_test, estimator.predict(features_test))

# calculate accuracy and confusion matrix
totalAccuracy = totalAccuracy / numOfSplits
totalConfusion = np.rint(totalConfusion).astype(int)

print(totalAccuracy)
print(totalConfusion)
