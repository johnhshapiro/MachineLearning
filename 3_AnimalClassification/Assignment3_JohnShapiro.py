import os
import cv2
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.neighbors import KNeighborsClassifier

Distances = [1,2]
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

best_k = best_distance = 0
best_f = 0.0

# Find the k and distance value that results in the best macro avg f-score
for k in K:
    for distance in Distances:
        model = KNeighborsClassifier(n_neighbors=k, p=distance)
        model.fit(data_train, labels_train)
        labels_pred = model.predict(data_valid)
        f_score = f1_score(labels_valid, labels_pred, average='macro')
        if f_score > best_f:
            best_f = f_score
            best_k = k
            best_distance = distance
            print(best_f,best_k,best_distance)

# Print the report for the test data using the best f-score values for
# k and distance
model = KNeighborsClassifier(n_neighbors=best_k, p=best_distance)
model.fit(data_train, labels_train)
labels_pred = model.predict(data_test)
print(f"K = {best_k}, Distance = L{best_distance}")
print(classification_report(labels_test, labels_pred,target_names=le.classes_))