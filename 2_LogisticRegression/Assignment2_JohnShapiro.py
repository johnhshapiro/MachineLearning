import pandas as pd
import numpy as np
# import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import roc_auc_score,roc_curve, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay


# names of columns
data_columns = ['pregnant',
                'glucose',
                'bp',
                'skin',
                'insulin',
                'bmi',
                'pedigree',
                'age',
                'label']
# Read csv file using pre-determined column names
pima_data = pd.read_csv("pima-indians-diabetes.csv",
                        header=None,
                        names=data_columns)
# Pick 5 features to be used as data (X)
features = ['glucose',
            'bp',
            'insulin',
            'bmi',
            'pedigree']
X = pima_data[features]
y = pima_data.label

# Splitting training and testing data
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.4, random_state=0)

# Fit logistic regression model
logistic_regression = LogisticRegression(max_iter=10000)
logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)

print(X_test)

# # Print confusion matrix and classification report
# pima_conf_matrix = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:\n", pima_conf_matrix)
# print("Classification Report:\n", classification_report(y_test, y_pred))
# # Display confusion matrix as heatmap w/ seaborn
# plt.title("Confusion Matrix Heatmap")
# sn.heatmap(pima_conf_matrix, annot=True, fmt="d")
# plt.savefig('conf_matrix')

# # Display and save ROC with calculated AUC
# logreg_roc_auc = roc_auc_score(y_test, y_pred)
# fpr, tpr, thresholds = roc_curve(y_test, logistic_regression.predict_proba(X_test)[:,1])
# plt.figure()
# plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' %logreg_roc_auc)
# plt.plot([0, 1], [0, 1], 'r--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Reciever Operating Characteristic Curve')
# plt.legend(loc='lower right')
# plt.savefig('Log_ROC')
# plt.show()