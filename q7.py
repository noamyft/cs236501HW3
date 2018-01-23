import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sfs import sfs


numOfSplits = 4

data = pd.read_csv("flare.csv")

data = data.values
examples = data[:,0:32]
classification = data[:, 32]
classification = classification.astype(bool)

# divide the examples to train and test sets
examples_count = examples.shape[0]
examples_index = int(examples_count * 0.75)

examples_train = examples[:examples_index,:]
examples_test = examples[examples_index:,:]

classification_train = classification[:examples_index]
classification_test = classification[examples_index:]

# train KNN simple classifier
knn_simple = KNeighborsClassifier(n_neighbors=5)
knn_simple.fit(examples_train, classification_train)

print("accuracy for simple knn: ", accuracy_score(classification_test, knn_simple.predict(examples_test)))


# score function for sfs
def scoreForKNN(knn, examples, classification):
    numOfSplits = 4
    totalAccuracy = 0

    kf = KFold(n_splits=numOfSplits, shuffle=True)
    for train_index, valid_index in kf.split(examples):
        # split the data to train set and validation set:
        examples_train, examples_valid = examples[train_index], examples[valid_index]
        classification_train, classification_valid = classification[train_index], classification[valid_index]

        # train the knn on train set
        knn.fit(examples_train, classification_train)
        # test the knn on validation set
        totalAccuracy += accuracy_score(classification_valid, knn.predict(examples_valid))

    totalAccuracy = totalAccuracy / numOfSplits
    return totalAccuracy

# train KNN_features classifier

knn_features = KNeighborsClassifier(n_neighbors=5)
selected_features = sfs(examples_train, classification_train, 8, knn_features, scoreForKNN)

knn_features.fit(examples_train[:,selected_features], classification_train)

print("accuracy for knn with feature selection: ",
      accuracy_score(classification_test, knn_features.predict(examples_test[:,selected_features])))




