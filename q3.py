import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn import tree

numOfSplits = 4 # TODO change
# totalAccuracy = 0
# totalConfusion = np.zeros((2,2))
#
# ### average for shuflle ###
# avgFactor = 10


data = pd.read_csv("flare.csv")

data = data.values
features = data[:,0:32]
classification = data[:, 32]
classification = classification.astype(bool)



# for minSamples in range(2,31,7):
#     avgSum = 0
#     avgConfusion = np.zeros((2, 2))
# for i in range(avgFactor): # with shuffle
totalOverfitTrainAccuracy = 0
totalOverfitTestAccuracy = 0

totalUnderfitTrianAccuracy = 0
totalUnderfitTestAccuracy = 0

kf = KFold(n_splits=numOfSplits, shuffle=True)
for train_index, test_index in kf.split(features):
    # split the data to train set and validation set:
    features_train, features_test = features[train_index], features[test_index]
    classification_train, classification_test = classification[train_index], classification[test_index]

    # train the tree on train set
    overfitEstimator = tree.DecisionTreeClassifier(criterion="entropy")
    underfitEstimator = tree.DecisionTreeClassifier(criterion="entropy", max_depth = 1,  min_samples_split = 0.7, max_features = 1)

    overfitEstimator.fit(features_train, classification_train)
    underfitEstimator.fit(features_train, classification_train)

    # test the tree on test set
    totalOverfitTrainAccuracy += accuracy_score(classification_train, overfitEstimator.predict(features_train))
    totalOverfitTestAccuracy += accuracy_score(classification_test, overfitEstimator.predict(features_test))

    totalUnderfitTrianAccuracy += accuracy_score(classification_train, underfitEstimator.predict(features_train))
    totalUnderfitTestAccuracy += accuracy_score(classification_test, underfitEstimator.predict(features_test))


totalOverfitTrainAccuracy = totalOverfitTrainAccuracy / numOfSplits
totalOverfitTestAccuracy = totalOverfitTestAccuracy / numOfSplits

totalUnderfitTrianAccuracy = totalUnderfitTrianAccuracy / numOfSplits
totalUnderfitTestAccuracy = totalUnderfitTestAccuracy / numOfSplits
    # avgSum += totalAccuracy
    # avgConfusion += totalConfusion
# print("for Overfit estimator: train accuracy is: ",totalOverfitTrainAccuracy,
#       " test accuracy is: ", totalOverfitTestAccuracy)
# print("for Underfit estimator: train accuracy is: ",totalUnderfitTrianAccuracy,
#       " test accuracy is: ", totalUnderfitTestAccuracy)
print(totalOverfitTrainAccuracy)
print(totalUnderfitTrianAccuracy)


