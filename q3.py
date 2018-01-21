import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import id3

numOfSplits = 4 # TODO change
totalAccuracy = 0
totalConfusion = np.zeros((2,2))

### average for shuflle ###
avgFactor = 10


data = pd.read_csv("flare.csv")

data = data.values
features = data[:,0:32]
classification = data[:, 32]
classification = classification.astype(bool)



for minSamples in range(2,31,7):
    avgSum = 0
    avgConfusion = np.zeros((2, 2))
    for i in range(avgFactor): # with shuffle
        totalAccuracy = 0
        totalConfusion = np.zeros((2, 2))
        kf = KFold(n_splits=numOfSplits, shuffle=True)
        for train_index, test_index in kf.split(features):
            # split the data to train set and validation set:
            features_train, features_test = features[train_index], features[test_index]
            classification_train, classification_test = classification[train_index], classification[test_index]

            # train the tree on train set
            estimator = id3.Id3Estimator(min_samples_split=minSamples, prune=True)
            estimator.fit(features_train, classification_train)
            # test the tree on validation set
            totalAccuracy += accuracy_score(classification_test, estimator.predict(features_test))
            totalConfusion += confusion_matrix(classification_test, estimator.predict(features_test))

        totalAccuracy = totalAccuracy / numOfSplits
        totalConfusion = np.rint(totalConfusion / numOfSplits).astype(int)
        avgSum += totalAccuracy
        avgConfusion += totalConfusion
    print("for min samples =",minSamples,"the average accuracy is:",avgSum / avgFactor)
    print("for min samples =",minSamples,"the average confusion is:",avgConfusion / avgFactor)


