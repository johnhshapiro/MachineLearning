import numpy as np
import pandas as pd
import matplotlib
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# models
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('iris.csv')
X = data.iloc[:,:-1]
y = data.iloc[:,4]
le = preprocessing.LabelEncoder()
ylabels = le.fit_transform(y)
trainX, testX, trainY, testY = train_test_split(X, ylabels, test_size=0.3, random_state=0)

model = svm.SVC(kernel='linear', C=1.0)
model.fit(trainX, trainY)
predicted_labels = model.predict(testX)
print(classification_report(testY, predicted_labels, target_names=le.classes_))

logistic_regression = LogisticRegression(max_iter=10000)
logistic_regression.fit(trainX, trainY)
y_pred = logistic_regression.predict(testX)
print(classification_report(testY, y_pred, target_names=le.classes_))

decision_tree = DecisionTreeClassifier()
decision_tree.fit(trainX, trainY)
y_pred = decision_tree.predict(testX)
print(classification_report(testY, y_pred, target_names=le.classes_))