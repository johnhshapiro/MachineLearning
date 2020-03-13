import os
import cv2
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier

K = [3,5,7]

dataset_path = "animals/"
data = []
labels = []

# List of classes (in our case; animals)
class_folders = os.listdir(dataset_path)

# traverse each class folder, get the image data,
# and label based on the folder the image is in
for class_name in class_folders:
    image_list = os.listdir(dataset_path + class_name)
    print(class_name)
    for image_name in image_list:
        image = cv2.imread(dataset_path + class_name + '/' + image_name)
        dimensions = image.shape
        image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
        data.append(image)
        labels.append(class_name.replace('s',''))
    print('Folder Done')

# Transform data into a list of vectors in a numpy array
data = np.array(data)
data = data.reshape((data.shape[0],3072))

# sklearn label encoder replaces labels with representative indexes
le = preprocessing.LabelEncoder()
labels = le.fit_transform(labels)
labels = np.array(labels)

# split training data from test data
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.3,random_state=42)
# split validation data from test data
data_valid, data_test, labels_valid, labels_test = train_test_split(data_test, labels_test, test_size=0.33, random_state=42)

for k in K:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(data_train, labels_train)
    labels_pred = model.predict(data_test)
    print(classification_report(labels_test, labels_pred,target_names=le.classes_))