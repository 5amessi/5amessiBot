from nltk import *
from nltk.tokenize import *
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras.preprocessing import sequence,text
from keras.layers import *
import keras
import os
import string
from keras.models import load_model
from keras.models import model_from_json
from flask import Flask,request

MAX_SEQUENCE_LENGTH = 20
app = Flask(__name__)
def load():
    global word2ind
    global ind2label
    word2ind = {}
    ind2label = []
    words = pd.read_csv('ind_to_word',header=None)
    for ind , line in enumerate(np.asarray(words)):
        word2ind[line[0]] = ind+1

    labels = pd.read_csv('ind2label',header=None)
    for ind , line in enumerate(np.asarray(labels)):
        ind2label.append(line[0])

def seq_data(data):
    list_of_X = []
    for i in range(len(data)):
        line = []
        for word in data[i]:
            if word2ind.get(word) is not None:
                line.append(word2ind[word])
            else:
                line.append(0)
        list_of_X.append(line)
    return list_of_X
import spacy
nlp = spacy.load('en_core_web_sm')
def ner(data):
    global outner
    outner = []
    for i, line in enumerate(data):
        doc = nlp(line)
        outner.append(doc)
        diflen = 0
        for ent in doc.ents:
            data[i] = data[i][0:ent.start_char+diflen] + ent.label_ + data[i][ent.end_char +diflen:]
            diflen += len(ent.label_)-len(ent.text)
            # print(ent.text, ent.start_char, ent.end_char, ent.label_)
    return data

def preprocess(data):
    data = ner(data)
    #stop = stopwords.words('english')
    stop = list(string.punctuation)
    tokenizer = TreebankWordTokenizer()
    list_of_X = []
    for i in range(len(data)):
        line = []
        for word in tokenizer.tokenize(data[i]):
            str = word.lower()
            if str not in stop:
                line.append(str)
        list_of_X.append(line)
    return list_of_X

load()

# @app.route('/',methods=['POST'])
def predict():
    global out
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("model.h5")
    inp = input()
    while(inp != "exit"):
        inp = [inp]
        inp = preprocess(inp)
        print(inp)
        inp = seq_data(inp)
        inp = keras.preprocessing.sequence.pad_sequences(inp, MAX_SEQUENCE_LENGTH, padding='pre', truncating='post',value=0)
        out = model.predict(inp)
        print(np.max(out))
        if(np.max(out) < 0.4 or np.max(inp) == 0):
            print("i can't understand you")
        else:
            out = np.argmax(out, axis=1)
            lable = ind2label[out[0]]
            print(lable)
            if (lable == "weather"):
                for ner_ in outner:
                    for ent in ner_.ents:
                        # print(ent.text, " ", ent.label_)
                        if ent.label_== "GPE":
                            print(ent.text)
        inp = input()
    return ind2label[out[0]]
# app.run(debug = True,port=8085)
predict()

