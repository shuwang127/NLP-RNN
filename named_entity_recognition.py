import sys
import re
import os
from collections import defaultdict
import itertools
import numpy as np
import time
import torch.utils.data as data_utils
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import gc
from sklearn.metrics import accuracy_score
import torch

# global path
rootPath = './'
dataPath = rootPath + '/conll2003/'
tempPath = rootPath + '/temp/'

# Logger: redirect the stream on screen and to file.
class Logger(object):
    def __init__(self, filename = "log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

def main():
    # initialize the log file.
    sys.stdout = Logger()
    print("-- AIT726 Homework 3 from Julia Jeng, Shu Wang, and Arman Anwar --")

    ReadData()

    return

def ReadData():
    # lower case capitalized words.
    def LowerCase(data):
        def LowerFunc(matched):
            return matched.group(1).lower()
        pattern = r'([A-Z]+[a-z]+)'
        data = re.sub(pattern, LowerFunc, data)
        return data

    vocab = defaultdict(list)
    dataTrain = []
    sentence = []
    # read text from train file.
    file = open(dataPath + '/train.txt').read()
    for word in file.splitlines():
        seg = word.split(' ')
        if len(seg) <= 1:
            if (sentence):
                dataTrain.append(sentence)
            sentence = []
        else:
            if seg[0] != '-DOCSTART-':
                sentence.append([LowerCase(seg[0]), seg[-1]])
                vocab[LowerCase(seg[0])].append(seg[-1])
    # get the max length of dataTrain.
    maxlenTrain = max([len(sentence) for sentence in dataTrain])

    dataValid = []
    sentence = []
    # read text from valid file.
    file = open(dataPath + '/valid.txt').read()
    for word in file.splitlines():
        seg = word.split(' ')
        if len(seg) <= 1:
            if (sentence):
                dataValid.append(sentence)
            sentence = []
        else:
            if seg[0] != '-DOCSTART-':
                sentence.append([LowerCase(seg[0]), seg[-1]])
                vocab[LowerCase(seg[0])].append(seg[-1])
    # get the max length of dataValid.
    maxlenValid = max([len(sentence) for sentence in dataValid])

    dataTest = []
    sentence = []
    # read text from test file.
    file = open(dataPath + '/test.txt').read()
    for word in file.splitlines():
        seg = word.split(' ')
        if len(seg) <= 1:
            if (sentence):
                dataTest.append(sentence)
            sentence = []
        else:
            if seg[0] != '-DOCSTART-':
                sentence.append([LowerCase(seg[0]), seg[-1]])
                vocab[LowerCase(seg[0])].append(seg[-1])
    # get the max length of dataTest.
    maxlenTest = max([len(sentence) for sentence in dataTest])

    # the maximum 
    maxlen = max(maxlenTrain, maxlenValid, maxlenTest)







    return

def GetVector():
    return

def RNNTrain():
    return

# The program entrance.
if __name__ == "__main__":
    main()