import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import id3

data = pd.read_csv("flare.csv")

data = data.values
features = data[:,0:32]
classification = data[:, 32]
classification = classification.astype(bool)

estimator = id3.Id3Estimator()
estimator.fit(features, classification)

print(accuracy_score(classification, estimator.predict(features)))

# print(data)