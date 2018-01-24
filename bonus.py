

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sfs import sfs
from sklearn import tree
from sklearn import preprocessing


from sklearn.svm import SVC





numOfSplits = 4
numOfClasses = 8




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




data = pd.read_csv("diamonds.csv")

classification = data['clarity']
classification = pd.get_dummies(classification) # converting enum to binary vector

examples = data.drop(['clarity'],axis=1)
examples = pd.get_dummies(examples,columns=['cut','color']) # converting enum to binary vector
# print(examples.head(5))

classification = classification.values
classification = np.dot(classification,[i for i in range(numOfClasses)]).astype(int)

examples = examples.values
examples = examples[:,1:] # remove first col


min_max_scaler = preprocessing.MinMaxScaler()
norm_examples = min_max_scaler.fit_transform(examples)
# print(norm_examples[0:5])

# divide the examples to train and test sets
examples_count = examples.shape[0]
examples_index = int(examples_count * 0.75)

norm_examples_train = norm_examples[:examples_index,:]
norm_examples_test = norm_examples[examples_index:,:]

classification_train = classification[:examples_index]
classification_test = classification[examples_index:]



### train KNN simple classifier ###
knn_simple = KNeighborsClassifier(n_neighbors=5)
knn_simple.fit(norm_examples_train, classification_train)

# print(knn_simple.predict(examples_train[0:10]))
# print("real class is\n",classification_train[0:10])
print("simple knn",accuracy_score(classification_test, knn_simple.predict(norm_examples_test)))
### train KNN simple classifier ###


### train KNN_features classifier ###

knn_features = KNeighborsClassifier(n_neighbors=10)
selected_features = sfs(norm_examples_train, classification_train, 20, knn_features, scoreForKNN)

knn_features.fit(norm_examples_train[:,selected_features], classification_train)

print("feature selection knn",accuracy_score(classification_test, knn_features.predict(norm_examples_test[:,selected_features])))
### train KNN_features classifier ###


### tree of life ###
estimator = tree.DecisionTreeClassifier(criterion="entropy")
estimatorEmbedded = tree.DecisionTreeClassifier(criterion="entropy", max_depth=40)
estimator.fit(norm_examples_train, classification_train)
estimatorEmbedded.fit(norm_examples_train, classification_train)
# test the tree on validation set

print("simple tree",accuracy_score(classification_test, estimator.predict(norm_examples_test)))
print("tree with twist",accuracy_score(classification_test, estimatorEmbedded.predict(norm_examples_test)))
### tree of life ###


### SVM style ###

clfSVM = SVC(max_iter=500)
clfSVM.fit(norm_examples_train,classification_train)
print(clfSVM.get_params(True))
print("simple svm",accuracy_score(classification_test, clfSVM.predict(norm_examples_test)))

### SVM style ###





























