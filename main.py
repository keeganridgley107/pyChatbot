import nltk
from nltk.stem.lancaster import LancasterStemmer

import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle

stemmer = LancasterStemmer()
with open("intents.json") as file:
    data = json.load(file)

# check for training data; if trained load data 
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            token_words = nltk.word_tokenize(pattern)
            words.extend(token_words)
            docs_x.append(token_words)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    # need to convert strings to bag of words for ml use
    # neural nets only use weighted numerical values == convert strings into numbers
    # bag represents one hot encoding
    # if word exists == 1 else == 0

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# remove any previous data from tensorflow
tf.reset_default_graph()

# setup neural net 
# pass training input bc each training input will have same length 
# i.e. model should expect a array of 4 word length 
net = tflearn.input_data(shape=[None, len(training[0])])
# setup first 'hidden' layer connected to input layer with 8 neurons 
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
# output layer : get probabilities for each outcome
net = tflearn.fully_connected(net, len(output[8]), activation="softmax")
# run regression 
net = tflearn.regression(net)
# train model 
model = tflearn.DNN(net)

# load trained model; else train & save model
try:
    model.load("model.tflearn")
except:
    # pass model training data
    # change epoch to alter recursion amount num = times trained on data 
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    # save the trained model
    model.save("model.tflearn")