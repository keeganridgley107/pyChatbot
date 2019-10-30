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

print(data)

words = []
labels = []
docs = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        token_words = nltk.word_tokenize(pattern)
        words.extend(token_words)
        docs.append(pattern)

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

print(labels)

