from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
)

from sklearn.model_selection import KFold

import pandas as pd
import itertools
import numpy as np

import utils

url = "https://raw.githubusercontent.com/AvaneeshM/WineDataset/main/WineQT.csv"

dataset = pd.read_csv(url)

print(dataset.columns)
print(dataset.head(10))

dataset = dataset.dropna()
string_to_list = utils.string_to_list

feature_cols = ['fixed acidity','volatile acidity','citric acid',
                'residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide',
                'density','pH','sulphates','alcohol']

Z = dataset[feature_cols]
y = dataset['quality']

X_train, X_test, y_train, y_test = train_test_split(Z, y, test_size=0.25, random_state=16)

# instantiate the model (using the default parameters)
model = GaussianNB()

model.fit(X_train, y_train);

y_pred = model.predict(X_test)

accuray = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy:", accuray)
print("F1 Score:", f1)


# 4-Fold Cross Validation

kfold = KFold(n_splits= 4, random_state=None, shuffle=False)

results = cross_val_score(model, X, Y, cv=kfold)

print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

model.fit(X_train, y_train)
predicted = model.predict(X_test)
report = classification_report(y_test, predicted)
print(report)      