import os
import cv2
import numpy as np

dataset_path = "animals/animals/"
data = []
labels = []

class_folders = os.listdir(dataset_path)

for class_name in class_folders:
    image_list = os.listdir(dataset_path + class_name)
    for image_name in image_list:
        image = cv2.imread(dataset_path + image_name)
        image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
        data.append(image)
        labels.append(class_name.replace('s',''))

    print("Folder Done")

npdata = np.array(data)
print(npdata)
data_with_labels = (np.array(data),np.array(labels))