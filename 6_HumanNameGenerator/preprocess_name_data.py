"""
Author: John Shapiro

Turns a csv of names (which was seperated by gender (boy, girl, unisex))
into a list of names (one by one).
This also tells me the max and min name lengths so I have an idea of how long
my randomly generated names should be.
"""
import numpy
from random import randint

alpha_list = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def word_to_numical_features(word, label):
    word = word_without_new_lines(word)
    features = ""
    for letter_index in range(15):
        if letter_index < len(word):
            features = features + (alphabet.get(word[letter_index]) + ",")
        else:
            features = features + "0,"
    features = features + label + "\n"
    return features

def word_without_new_lines(word):
    return word.replace("\n", "").lower()


def generate_non_word():
    length = randint(1, 15)
    not_a_word = ""
    for i in range(length):
        not_a_word = not_a_word + alpha_list[randint(1,26)]
    return not_a_word

        

alphabet = {
    'a': '1',
    'b': '2',
    'c': '3',
    'd': '4',
    'e': '5',
    'f': '6',
    'g': '7',
    'h': '8',
    'i': '9',
    'j': '10',
    'k': '11',
    'l': '12',
    'm': '13',
    'n': '14',
    'o': '15',
    'p': '16',
    'q': '17',
    'r': '18',
    's': '19',
    't': '20',
    'u': '21',
    'v': '22',
    'w': '23',
    'x': '24',
    'y': '25',
    'z': '26'
}

# These are names
name_file = open("names.csv", "r")
name_groups = name_file.readlines()
names_and_english_words = []

# These are not names
word_file = open("english_words.txt", "r")
not_names = word_file.readlines()

output_file = open("name_data.csv", "w")

words = []
not_names_list = []

name_max = 0
for name_goup in name_groups:
    names = name_goup.split(",")
    for name in names:
        if len(name) > 0 and name != "\n":
            words.append(word_without_new_lines(name))
            if len(name) > name_max:
                name_max = len(name)
            names_and_english_words.append(word_to_numical_features(name, "1"))

for not_name in not_names:
    not_names_list.append(word_without_new_lines(not_name))

not_a_word_list = []
while len(not_a_word_list) < 1500:
    not_a_word = generate_non_word()
    if not_a_word not in names_and_english_words and not_a_word not in not_names_list:
        not_a_word_list.append(not_a_word)
for not_word in not_a_word_list:
    names_and_english_words.append(word_to_numical_features(not_word, "0"))

output_file.writelines(names_and_english_words)