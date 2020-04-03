
import numpy as np
import pandas as pd
import matplotlib
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
# models
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

dataset_path = 'iris.csv'
try:
    data = pd.read_csv(dataset_path)
except:
    print("Make sure the dataset_path is correct")
    exit(0)
X = data.iloc[:,:-1]
y = data.iloc[:,4]
le = preprocessing.LabelEncoder()
ylabels = le.fit_transform(y)

test_sizes = [i/10 for i in range(1,6)]

for size in test_sizes:
    trainX, testX, trainY, testY = train_test_split(X, ylabels, test_size=size, random_state=0)

    model = svm.SVC(kernel='linear', C=1.0)
    model.fit(trainX, trainY)
    y_pred = model.predict(testX)
    print("SVC linear kernel with test_size =", size)
    print(classification_report(testY, y_pred, target_names=le.classes_))

    logistic_regression = LogisticRegression(max_iter=10000)
    logistic_regression.fit(trainX, trainY)
    y_pred = logistic_regression.predict(testX)
    print("LogisticRegression with test_size =", size)
    print(classification_report(testY, y_pred, target_names=le.classes_))

    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(trainX, trainY)
    y_pred = decision_tree.predict(testX)
    print("DecisionTreeClassifier with test_size =", size)
    print(classification_report(testY, y_pred, target_names=le.classes_))

# Print out macro average f-scores for different test_sizes and svm_kernels
# best_f = 0.0
# best_kernel = ''
# svm_kernels = ['linear', 'poly', 'rbf']
# for  i in range(1,10):
#     trainX, testX, trainY, testY = train_test_split(X, ylabels, test_size=i/10, random_state=0)
#     print('test_size =', i/10)
#     for svm_kernel in svm_kernels:
#         kernel = svm.SVC(kernel=svm_kernel, C=1.0)
#         kernel.fit(trainX, trainY)
#         y_pred = kernel.predict(testX)
#         f_score = f1_score(testY, y_pred, average='macro')
#         print(svm_kernel, f_score)
#         if f_score > best_f:
#             best_f = f_score
#             best_kernel = svm_kernel