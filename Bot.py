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
import spacy
nlp = spacy.load('en_core_web_sm')

MAX_SEQUENCE_LENGTH = 20
app = Flask(__name__)
def load():
    global word2ind
    global ind2label
    global model
    word2ind = {}
    ind2label = []
    words = pd.read_csv('ind_to_word',header=None)
    for ind , line in enumerate(np.asarray(words)):
        word2ind[line[0]] = ind+1

    labels = pd.read_csv('ind2label',header=None)
    for ind , line in enumerate(np.asarray(labels)):
        ind2label.append(line[0])
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("model.h5")
    # model = load_model('miso.h5')
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
def response_(lable):
    if (lable == "weather"):
        for ner_ in outner:
            for ent in ner_.ents:
                # print(ent.text, " ", ent.label_)
                if ent.label_ == "GPE":
                    return "the weather in " + ent.text +" is ..."
    elif(lable == "make_call"):
        return "calling ..."

    elif (lable == "open_app"):
        return "... opening"

    elif (lable == "send_message"):
        return "i will send message to ..."

    elif (lable == "public_greeting"):
        return "hello <3"

    elif (lable == "location_app"):
        return "location"

    elif (lable == "how_are_you"):
        return "I am fine"

    elif (lable == "introduce_himself"):
        return "hello ..."

    elif (lable == "introduce_someone"):
        return "hello ..."

    elif (lable == "asking_for_you_name"):
        return "My name is 5amessi"

    elif (lable == "feeling_god"):
        return "good"

    elif (lable == "Conventional_closing"):
        return "see you soon"

    elif (lable == "feeling_love_with_me"):
        return "love you to <3"

    elif (lable == "feeling_nice_with_me"):
        return "nice"

    elif (lable == "Thanking"):
        return "your welcome"

    elif (lable == "set_alarm"):
        return "i will wake up you"

    elif (lable == "cansel_alarm"):
        return "i will cansel it"

    elif (lable == "set_reminder"):
        return "I will remind you"

    elif (lable == "cansel_reminder"):
        return "i will cansel it"

    elif (lable == "sleep"):
        return "Good night <3 :*"

    elif (lable == "bad_statement"):
        return "say sorry ..."
# @app.route('/',methods=['POST'])
def predict(str):
    # load()
    inp = [str]
    inp = preprocess(inp)
    # print(inp)
    inp = seq_data(inp)
    inp = keras.preprocessing.sequence.pad_sequences(inp, MAX_SEQUENCE_LENGTH, padding='pre', truncating='post',value=0)
    print(inp)
    out = model.predict(inp)
    lable = ind2label[np.argmax(out, axis=1)[0]]
    # print(np.max(out))
    if(np.max(out) < 0.4 or np.max(inp) == 0):
        # print("i can't understand you")
        response = "i can't understand you"
    else:
        # print(lable)
        response = response_(lable)
    return response
# app.run(debug = True,port=8085)
load()


