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

    vocab, sentences = ReadData()
    print(vocab)
    #print(sentences)

    return

def ReadData():
    def lower_repl(match):
        return match.group().lower()

    def lowercase_text(txt):
        txt = re.sub('([A-Z]+[a-z]+)', lower_repl, txt)  # lowercase words that start with captial
        return txt

    vocab = defaultdict(list)
    doc = []
    with open(dataPath + '/train.txt') as f:
        for word in f.read().splitlines():
            a = word.split(" ")
            if len(a) > 1:
                vocab[lowercase_text(a[0])].append(a[3])  # ad to vocab and to doc as dictionary with tag
                doc.append([lowercase_text(a[0]), a[3]])
            else:
                doc.append(a[0])
    doc.insert(0, '')

    # retaining the unique tags for each vocab word
    for k, v in vocab.items():
        vocab[k] = (list(set(v)))

    # getting the indices of the end of each sentence
    sentence_ends = []
    for i, word in enumerate(doc):
        if not word:
            sentence_ends.append(i)
    sentence_ends.append(len(doc) - 1)
    # creating a list of all the sentences
    sentences = []
    for i in range(len(sentence_ends) - 1):
        sentences.append(doc[sentence_ends[i] + 1:sentence_ends[i + 1]])

    # getting the length of the longest sentence
    max_sent_len = len(max(sentences, key=len))

    ## padding all of the sentences to make them length 113
    for sentence in sentences:
        sentence.extend(['0', '<pad>'] for i in range(max_sent_len - len(sentence)))
    # This is the code to read the embeddings
    vocab['0'] = '<pad>'
    return vocab, sentences

def GetVector():
    return

def RNNTrain():
    return

# The program entrance.
if __name__ == "__main__":
    main()