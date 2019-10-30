import nltk
from nltk.stem.lancaster import LancasterStemmer

import numpy
import tflearn
import tensorflow
import random
import json

stemmer = LancasterStemmer()
with open("intents.json") as file:
    data = json.load(file)

# debug
# print("JSON : {}".format(data))

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        token_words = nltk.word_tokenize(pattern)
        words.extend(token_words)
        docs_y.append(pattern)
        docs_y.append(intent["tag"])
    if intent["tag"] not in labels:
        labels.append(intent["tag"])
words = [stemmer.stem(w.lower()) for w in  words]
words = sorted(list(set(words)))
labels = sorted(labels)

training = []
output = []

# need to convert strings to bag of words for ml use 
# neural nets only use weighted numerical values == convert strings into numbers 
# bag represents one hot encoding 
# if word exists == 1 else == 0

out_empty = [0 for _ in range (len(classes))]

for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]
    for w in words:
        if w in wrds: 
            bag.append(1)
        else:
            bag.append(0)
# debug 
# print("JSON LABELS : {}".format(labels)) 

