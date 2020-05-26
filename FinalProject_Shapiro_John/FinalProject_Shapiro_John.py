"""Name Generator

Author: John Shapiro

This program trains a sequential neural network to identify randomly
    generated string as either names or not-names in an attempt to
    create new unique names that can be given to people.

Names are from www.scarymommy.com. It's a list of about 1900 names. Weirdly
  they removed the page so I can't find it, but I still want to credit the
  source of the name list.
Got the list of most common english words to make sure my random words were
  really not word-like from:
  https://www.ef.edu/english-resources/english-vocabulary/top-1000-words/"""


# import the necessary packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.optimizers import SGD
from random import randint
import numpy as np
import pandas as pd
import keras.utils

# Used to convert numerical word representations to strings
alpha_list = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def name_numerical_representaion_list_to_name_string(name_numerical_representaion_list):
    """Converts a list of ints representing letters to a word

    Arguments:
        name_numerical_representaion_list {List<Int>} -- A numerical representation of a word

    Returns:
        String -- this is the word that corresponds to the numberical representation
    """
    name = ""
    for letter_numerical_representation in name_numerical_representaion_list:
        name = name + alpha_list[letter_numerical_representation]
    return name.replace(' ', '')

def generate_random_name_numerical_representation_list():
    """Generate a random word (letters are completely random) to be a candidate
        to be checked as a potential "name"


    Returns:
        List<Int> -- A numerical representation of a string with each letter
            stored as an item in a list
    """
    min_name_length = 2
    max_name_length = 15
    name_length = (randint(min_name_length, max_name_length))
    name = [randint(1, 26) for i in range(name_length)]
    for i in range(max_name_length- name_length):
        name.append(0)
    return name

# Load data set which I parsed and built using the sources listed at the top
dataset_path = 'name_data.csv'
try:
    data = pd.read_csv(dataset_path)
except:
    print("Make sure the dataset_path is correct")
    exit(0)
X = data.iloc[:,:-1]
y = data.iloc[:,15]

# split training and testing data
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.3, random_state=0)


# convert the labels from integers to vectors
trainY = keras.utils.to_categorical(trainY, num_classes=None)
testY = keras.utils.to_categorical(testY, num_classes=None)

# define the 784-256-128-2 architecture using Keras
model = Sequential()
model.add(Dense(256, input_shape=(15,), activation="relu"))  #'sigmoid'
model.add(Dense(128, activation="relu"))
model.add(Dense(2, activation="softmax"))


name_lists = []
sgd = SGD(lr= 0.01, decay=1e-6, momentum=0.9, nesterov=True)

# compile the model with chosen settings
model.compile(loss="categorical_crossentropy", optimizer=sgd,
	metrics=["accuracy"])

epochs_per_loop = 100000

# train the model for a specified number of epochs each time
for i in range(5):
    H = model.fit(trainX, trainY, validation_data=(testX, testY),
        epochs=epochs_per_loop, batch_size=128)

    # evaluate the network
    predictions = model.predict(testX, batch_size=128)
    print(classification_report(testY.argmax(axis=1),
    	predictions.argmax(axis=1)))

    # List the first ten random words to get categorized as a "name"
    names = []
    while len(names) < 10:
        name_representation = generate_random_name_numerical_representation_list()
        name_numerical_representaion_list = np.array([name_representation])
        name_numerical_representaion_list = (
            name_numerical_representaion_list).reshape(1, 15)
        predicted_label = model.predict(name_numerical_representaion_list)[0]
        if(predicted_label[0] < predicted_label[1]):
            name = name_numerical_representaion_list_to_name_string(
                name_representation)
            names.append(name)

    current_epochs = (i + 1) * epochs_per_loop

    # add lists of names after epoch groups with the epoch count appended
    # add the classification report after every loop of epochs
    names.append(f"Epochs: {current_epochs}")
    name_lists.append(str(names) + "\n")
    name_lists.append(str(classification_report(testY.argmax(axis=1),
    	predictions.argmax(axis=1))) + "\n")


# write name lists and classification reports to name_lists.txt
output_file = open("name_lists.txt", "w")
output_file.writelines(name_lists)
output_file.close()