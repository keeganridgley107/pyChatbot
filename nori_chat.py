import nltk
from nltk.stem.lancaster import LancasterStemmer

import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle

def main():
    """
    main function in NoriChatbot
    """    
    stemmer = LancasterStemmer()
    with open("intents.json") as file:
        data = json.load(file)

    # load / save for word data
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
    # setup 'hidden' layers connected to input layer with 8 neurons 
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    # output layer : get probabilities for each outcome
    net = tflearn.fully_connected(net, len(output[8]), activation="softmax")
    # run regression 
    net = tflearn.regression(net)
    # create deep neural network model 
    model = tflearn.DNN(net,session=None)

    # load trained model; else train & save model
    #   NOTE: if intents file is updated =>
    #   delete pickle/model files, comment out load, in nori.py
    try:
        # nori.py
        model.load("model.tflearn")
    except:
        # train model 
        # change epoch to alter recursion amount num = times trained on data 
        model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
        # save the trained model
        model.save("model.tflearn")

    def bag_of_words(sentance, words):
        """
        Turn a sentance into a one hot encoded bag of words 

        Returns : numpy array 
        """
        # setup list with 0 for # of words
        bag = [0 for _ in range(len(words))]
        
        s_words = nltk.word_tokenize(sentance)
        s_words = [stemmer.stem(word.lower()) for word in s_words]

        # check if sentance words are in list; set bag = 1 if exists
        for se in s_words:
            for i, w in enumerate(words): 
                if w == se:
                    bag[i]= (1)

        return np.array(bag)

    def chat():
        """
        Main chat function
        """
        print("start talking with the bot! (type quit to leave)")
        while True:
            imp = input("You: ")
            if imp.lower() == "quit":
                break
            # predict expects a list, => list comprehension to create one out of b_o_W() return 
            results = model.predict([bag_of_words(imp, words)])[0]
            
            # return index of greatest value in list 
            results_index = np.argmax(results)
            # label of highest probability 
            tag = labels[results_index]

            # debug
            # print("Index was: {}".format(results[results_index]))
            # print(tag)

            # print response if confidence is 80% else print idk msg
            if results[results_index] > 0.7:
                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']

                # print random response from list with matching tag
                # TODO: improve  
                print(random.choice(responses))
            else:
                print("Sorry, I don't understand! Please try again!")
    chat()

if __name__ == '__main__':
    main()
    