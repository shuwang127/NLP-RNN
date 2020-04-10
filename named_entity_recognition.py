'''
  Author: Julia Jeng, Shu Wang, Arman Anwar
  Brief: AIT 726 Homework 3
  Usage:
      Put file 'named_entity_recognition.py' and folder 'conll2003' in the same folder.
      -named_entity_recognition.py
      -conll2003
        |---conlleval.py
        |---test.py
        |---train.py
        |---valid.py
      -temp
        |---GoogleNews-vectors-negative300.bin
  Command to run:
      python named_entity_recognition.py
  Description:
      Build and train a recurrent neural network (RNN) with hidden vector size 256.
      Loss function: Adam loss.
      Embedding vector: 128-dimensional.
      Learning rate: 0.0001.
      Batch size: 256
'''

import sys
import re
import os
from collections import defaultdict
from gensim import models
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as torchdata
import gc
from sklearn.metrics import accuracy_score
from conll2003.conlleval import evaluate_conll_file
import torch

# global path
rootPath = './'
dataPath = rootPath + '/conll2003/'
tempPath = rootPath + '/temp/'
outsPath = rootPath + '/outputs/'
modsPath = rootPath + '/models/'
# global variable
maxEpoch = 1000
perEpoch = 5

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
    sys.stdout = Logger('named_entity_recognition.txt')
    print("-- AIT726 Homework 3 from Julia Jeng, Shu Wang, and Arman Anwar --")
    # read data from files.
    dataTrain, dataValid, dataTest, vocab = ReadData()
    # get mapping.
    wordDict, classDict = GetDict(vocab)
    dTrain, lTrain = GetMapping(dataTrain, wordDict, classDict)
    dValid, lValid = GetMapping(dataValid, wordDict, classDict)
    dTest, lTest = GetMapping(dataTest, wordDict, classDict)
    # get the embedding.
    preWeights = GetEmbedding(wordDict)
    # demo
    model = TrainRNN(dTrain, lTrain, dValid, lValid, preWeights, preTrain=True, Type='RNN', bidirect=False)
    TestRNN(model, dTest, lTest, wordDict, classDict, preTrain=True, Type='RNN', bidirect=False)
    model = TrainRNN(dTrain, lTrain, dValid, lValid, preWeights, preTrain=True, Type='RNN', bidirect=True)
    TestRNN(model, dTest, lTest, wordDict, classDict, preTrain=True, Type='RNN', bidirect=True)
    model = TrainRNN(dTrain, lTrain, dValid, lValid, preWeights, preTrain=True, Type='LSTM', bidirect=False)
    TestRNN(model, dTest, lTest, wordDict, classDict, preTrain=True, Type='LSTM', bidirect=False)
    model = TrainRNN(dTrain, lTrain, dValid, lValid, preWeights, preTrain=True, Type='LSTM', bidirect=True)
    TestRNN(model, dTest, lTest, wordDict, classDict, preTrain=True, Type='LSTM', bidirect=True)
    model = TrainRNN(dTrain, lTrain, dValid, lValid, preWeights, preTrain=True, Type='GRU', bidirect=False)
    TestRNN(model, dTest, lTest, wordDict, classDict, preTrain=True, Type='GRU', bidirect=False)
    model = TrainRNN(dTrain, lTrain, dValid, lValid, preWeights, preTrain=True, Type='GRU', bidirect=True)
    TestRNN(model, dTest, lTest, wordDict, classDict, preTrain=True, Type='GRU', bidirect=True)
    # best is bi-LSTM.
    model = TrainRNN(dTrain, lTrain, dValid, lValid, preWeights, preTrain=False, Type='LSTM', bidirect=True, learnRate=0.0001)
    TestRNN(model, dTest, lTest, wordDict, classDict, preTrain=False, Type='LSTM', bidirect=True, learnRate=0.0001)
    return

def ReadData():
    '''
    Read train, valid, test data from files. And get vocabulary.
    :param: none
    :return: dataTrain - training data
             dataValid - validation data.
             dataTest - testing data.
    '''
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
    print('[Info] Load %d training sentences (max:%d words) from %s/train.txt.' % (len(dataTrain), maxlenTrain, dataPath))

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
    print('[Info] Load %d validation sentences (max:%d words) from %s/valid.txt.' % (len(dataValid), maxlenValid, dataPath))

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
    print('[Info] Load %d testing sentences (max:%d words) from %s/test.txt.' % (len(dataTest), maxlenTest, dataPath))

    # the max length of dataTrain, dataValid and dataTest.
    maxLen = max(maxlenTrain, maxlenValid, maxlenTest)
    # append 0s at the end of sentence to the max length.
    for sentence in dataTrain:
        for ind in range(maxLen - len(sentence)):
            sentence.append(['0', '<pad>'])
    for sentence in dataValid:
        for ind in range(maxLen - len(sentence)):
            sentence.append(['0', '<pad>'])
    for sentence in dataTest:
        for ind in range(maxLen - len(sentence)):
            sentence.append(['0', '<pad>'])

    # clear up the vocabulary.
    for word, label in vocab.items():
        vocab[word] = list(set(label))
    vocab['0'] = '<pad>'
    print('[Info] Get %d vocabulary words successfully.' % (len(vocab)))

    return dataTrain, dataValid, dataTest, vocab

def GetDict(vocab):
    '''
    Get the word and vocabulary dictionary.
    :param vocab: vocabulary
    :return: wordDict - word dictionary
             classDict - class dictionary
    '''
    # get word and class dictionary.
    wordDict = {word: index for index, word in enumerate(vocab)}
    classType = ['<pad>', 'O', 'B-ORG', 'B-PER', 'B-LOC', 'B-MISC', 'I-ORG', 'I-PER', 'I-LOC', 'I-MISC']
    classDict = {cls: index for index, cls in enumerate(classType)}
    # print(classDict)
    # return
    return wordDict, classDict

def GetMapping(data, wordDict, classDict):
    '''
    Map the data into index-form vectors.
    :param data: Input data.
    :param wordDict: word dictionary
    :param classDict: class dictionary
    :return: data2index - index-form data
             label2index - index-form label
    '''
    # map data and label to index-form.
    data2index = []
    label2index = []
    for sentence in data:
        # for each sentence.
        sent2index = []
        lb2index = []
        for word in sentence:
            sent2index.append(wordDict[word[0]])
            lb2index.append(classDict[word[1]])
        data2index.append(sent2index)
        label2index.append(lb2index)
    # return
    return np.array(data2index), np.array(label2index)

def GetEmbedding(wordDict):
    '''
    Get the embedding vectors from files.
    :param wordDict: word dictionary
    :return: pre-trained weights.
    '''
    # load preWeights.
    weightFile = 'preWeights.npy'
    if not os.path.exists(tempPath + '/' + weightFile):
        # find embedding file.
        embedFile = 'GoogleNews.txt'
        if not os.path.exists(tempPath + '/' + embedFile):
            # path validation.
            modelFile = 'GoogleNews-vectors-negative300.bin'
            if not os.path.exists(tempPath + '/' + modelFile):
                print('[Error] Cannot find %s/%s.' % (tempPath, modelFile))
                return
            # find word2vec file.
            model = models.KeyedVectors.load_word2vec_format(tempPath + '/' + modelFile, binary=True)
            model.save_word2vec_format(tempPath + '/' + embedFile)
            print('[Info] Get the word2vec format file %s/%s.' % (tempPath, embedFile))

        # read embedding file.
        embedVec = {}
        file = open(tempPath + '/' + embedFile, encoding = 'utf8')
        for line in file:
            seg = line.split()
            word = seg[0]
            embed = np.asarray(seg[1:], dtype = 'float32')
            embedVec[word] = embed
        np.save(tempPath+'/embedVec.npy', embedVec)

        # get mapping to preWeights.
        numWords = len(wordDict)
        numDims = 300
        preWeights = np.zeros((numWords, numDims))
        for ind, word in enumerate(wordDict):
            if word in embedVec:
                preWeights[ind] = embedVec[word]
            else:
                preWeights[ind] = np.random.normal(size=(numDims,))

        # save the preWeights.
        np.save(tempPath + '/' + weightFile, preWeights)
        print('[Info] Get pre-trained word2vec weights.')
    else:
        preWeights = np.load(tempPath + '/' + weightFile)
        print('[Info] Load pre-trained word2vec weights from %s/%s.' % (tempPath, weightFile))
    return preWeights

class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, preWeights, preTrain=True, bidirect=False, hiddenSize=256):
        super(RecurrentNeuralNetwork, self).__init__()
        # sparse parameters.
        numWords, numDims = preWeights.size()
        numBiDirect = 2 if bidirect else 1
        # embedding layer.
        self.embedding = nn.Embedding(num_embeddings=numWords, embedding_dim=numDims)
        self.embedding.load_state_dict({'weight': preWeights})
        if preTrain:
            self.embedding.weight.requires_grad = False
        # RNN layer.
        self.rnn = nn.RNN(input_size=numDims, hidden_size=hiddenSize, batch_first=True, bidirectional=bidirect)
        # fully-connected layer.
        self.fc = nn.Linear(in_features=hiddenSize*numBiDirect, out_features=10)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, x):
        embeds = self.embedding(x)
        rnn_out, hidden = self.rnn(embeds)
        out = rnn_out.contiguous().view(-1, rnn_out.shape[2])
        a1 = self.fc(out)
        a2 = self.sm(a1)
        return a2

class LongShortTermMemoryNetworks(nn.Module):
    def __init__(self, preWeights, preTrain=True, bidirect=False, hiddenSize=256):
        super(LongShortTermMemoryNetworks, self).__init__()
        # sparse parameters.
        numWords, numDims = preWeights.size()
        numBiDirect = 2 if bidirect else 1
        # embedding layer.
        self.embedding = nn.Embedding(num_embeddings=numWords, embedding_dim=numDims)
        self.embedding.load_state_dict({'weight': preWeights})
        if preTrain:
            self.embedding.weight.requires_grad = False
        # LSTM layer.
        self.lstm = nn.LSTM(input_size=numDims, hidden_size=hiddenSize, batch_first=True, bidirectional=bidirect)
        # fully-connected layer.
        self.fc = nn.Linear(in_features=hiddenSize*numBiDirect, out_features=10)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, x):
        embeds = self.embedding(x)
        rnn_out, hidden = self.lstm(embeds)
        out = rnn_out.contiguous().view(-1, rnn_out.shape[2])
        a1 = self.fc(out)
        a2 = self.sm(a1)
        return a2

class GatedRecurrentUnitNetwork(nn.Module):
    def __init__(self, preWeights, preTrain=True, bidirect=False, hiddenSize=256):
        super(GatedRecurrentUnitNetwork, self).__init__()
        # sparse parameters.
        numWords, numDims = preWeights.size()
        numBiDirect = 2 if bidirect else 1
        # embedding layer.
        self.embedding = nn.Embedding(num_embeddings=numWords, embedding_dim=numDims)
        self.embedding.load_state_dict({'weight': preWeights})
        if preTrain:
            self.embedding.weight.requires_grad = False
        # GRU layer
        self.gru = nn.GRU(input_size=numDims, hidden_size=hiddenSize, batch_first=True, bidirectional=bidirect)
        # fully-connected layer.
        self.fc = nn.Linear(in_features=hiddenSize*numBiDirect, out_features=10)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, x):
        embeds = self.embedding(x)
        rnn_out, hidden = self.gru(embeds)
        out = rnn_out.contiguous().view(-1, rnn_out.shape[2])
        a1 = self.fc(out)
        a2 = self.sm(a1)
        return a2

def TrainRNN(dTrain, lTrain, dValid, lValid, preWeights, preTrain=True, Type='RNN', bidirect=False, hiddenSize=256, batchsize=256, learnRate=0.001):
    '''
    The demo program of RNN training, validation and testing.
    :param dTrain: training data
    :param lTrain: training label
    :param dValid: validation data
    :param lValid: validation label
    :param dTest: testing data
    :param lTest: testing label
    :param wordDict: word dictionary
    :param classDict: class dictionary
    :param preWeights: pre-trained weights
    :param preTrain: Enable pre-trained?
    :param Type: RNN Type?
    :param bidirect: Enable Bidirection?
    :param hiddenSize: number of hidden nodes
    :param batchsize: batch size
    :param learnRate: learning rate
    :return: model in pytorch
    '''
    # tensor data processing.
    xTrain = torch.from_numpy(dTrain).long().cuda()
    yTrain = torch.from_numpy(lTrain).long().cuda()
    xValid = torch.from_numpy(dValid).long().cuda()
    yValid = torch.from_numpy(lValid).long().cuda()
    # batch size processing.
    train = torchdata.TensorDataset(xTrain, yTrain)
    trainloader = torchdata.DataLoader(train, batch_size=batchsize, shuffle=False)
    valid = torchdata.TensorDataset(xValid, yValid)
    validloader = torchdata.DataLoader(valid, batch_size=batchsize, shuffle=False)

    # get training weights
    lbTrain = [item for sublist in lTrain.tolist() for item in sublist]
    weights = []
    for lb in range(1, 10):
        weights.append(1 - (lbTrain.count(lb) / (len(lbTrain) - lbTrain.count(0))))
    weights.insert(0, 0)
    lbWeights = torch.FloatTensor(weights).cuda()

    # build the model of recurrent neural network.
    preWeights = torch.from_numpy(preWeights)
    if 'RNN' == Type:
        model = RecurrentNeuralNetwork(preWeights, preTrain=preTrain, bidirect=bidirect, hiddenSize=hiddenSize)
    elif 'LSTM' == Type:
        model = LongShortTermMemoryNetworks(preWeights, preTrain=preTrain, bidirect=bidirect, hiddenSize=hiddenSize)
    elif 'GRU' == Type:
        model = GatedRecurrentUnitNetwork(preWeights, preTrain=preTrain, bidirect=bidirect, hiddenSize=hiddenSize)
    else:
        print('[Error] RNN type %s is wrong.' % (Type))
        return
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print('[Demo] --- RNNType: %s | HiddenNodes: %d | Bi-Direction: %s | Pre-Trained: %s ---' % (Type, hiddenSize, bidirect, preTrain))
    print('[Para] BatchSize=%d, LearningRate=%.4f, MaxEpoch=%d, PerEpoch=%d.' % (batchsize, learnRate, maxEpoch, perEpoch))
    # optimizing with stochastic gradient descent.
    optimizer = optim.Adam(model.parameters(), lr=learnRate)
    # seting loss function as mean squared error.
    criterion = nn.CrossEntropyLoss(weight=lbWeights)
    # memory
    #torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.enabled = True

    # run on each epoch.
    accList = [0]
    for epoch in range(maxEpoch):
        # training phase.
        model.train()
        lossTrain = 0
        predictions = []
        labels = []
        for iter, (data, label) in enumerate(trainloader):
            data = data.to(device)
            label = label.contiguous().view(-1)
            label = label.to(device)
            optimizer.zero_grad()  # set the gradients to zero.
            yhat = model.forward(data)  # get output
            loss = criterion(yhat, label)
            loss.backward()
            optimizer.step()
            # statistic
            lossTrain += loss.item() * len(label)
            preds = yhat.max(1)[1]
            predictions.extend(preds.int().tolist())
            labels.extend(label.int().tolist())
            torch.cuda.empty_cache()
        #gc.collect()
        #torch.cuda.empty_cache()
        lossTrain /= len(lbTrain)
        # train accuracy.
        padIndex = [ind for ind, lb in enumerate(labels) if lb == 0]
        for ind in sorted(padIndex, reverse=True):
            del predictions[ind]
            del labels[ind]
        accTrain = accuracy_score(labels, predictions) * 100

        # validation phase.
        model.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            for iter, (data, label) in enumerate(validloader):
                data = data.to(device)
                label = label.contiguous().view(-1)
                label = label.to(device)
                yhat = model.forward(data)  # get output
                # statistic
                preds = yhat.max(1)[1]
                predictions.extend(preds.int().tolist())
                labels.extend(label.int().tolist())
                torch.cuda.empty_cache()
        #gc.collect()
        #torch.cuda.empty_cache()
        # valid accuracy.
        padIndex = [ind for ind, lb in enumerate(labels) if lb == 0]
        for ind in sorted(padIndex, reverse=True):
            del predictions[ind]
            del labels[ind]
        accValid = accuracy_score(labels, predictions) * 100
        accList.append(accValid)

        # output information.
        if 0 == (epoch + 1) % perEpoch:
            print('[Epoch %03d] loss: %.3f, train acc: %.3f%%, valid acc: %.3f%%.' % (epoch + 1, lossTrain, accTrain, accValid))
        # save the best model.
        if accList[-1] > max(accList[0:-1]):
            torch.save(model.state_dict(), tempPath + '/model.pth')
        # stop judgement.
        if (epoch + 1) >= 5 and accList[-1] < min(accList[-5:-1]):
            break

    # load best model.
    model.load_state_dict(torch.load(tempPath + '/model.pth'))
    return model

def TestRNN(model, dTest, lTest, wordDict, classDict, preTrain=True, Type='RNN', bidirect=False, hiddenSize=256, batchsize=256, learnRate=0.001):
    # test period
    xTest = torch.from_numpy(dTest).long().cuda()
    yTest = torch.from_numpy(lTest).long().cuda()
    test = torchdata.TensorDataset(xTest, yTest)
    testloader = torchdata.DataLoader(test, batch_size=batchsize, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # testing phase.
    model.eval()
    words = []
    predictions = []
    labels = []
    with torch.no_grad():
        for iter, (data, label) in enumerate(testloader):
            data = data.to(device)
            label = label.contiguous().view(-1)
            label = label.to(device)
            yhat = model.forward(data)  # get output
            # statistic
            words.extend(data.contiguous().view(-1).int().tolist())
            preds = yhat.max(1)[1]
            predictions.extend(preds.int().tolist())
            labels.extend(label.int().tolist())
            torch.cuda.empty_cache()
    #gc.collect()
    #torch.cuda.empty_cache()
    # testing accuracy.
    padIndex = [ind for ind, lb in enumerate(labels) if lb == 0]
    for ind in sorted(padIndex, reverse=True):
        del words[ind]
        del predictions[ind]
        del labels[ind]
    accuracy = accuracy_score(labels, predictions) * 100
    print('[Eval] Testing accuracy: %.3f%%.' % (accuracy))

    # get inverse index dictionary.
    wordIndDict = {ind: item for ind, item in enumerate(wordDict)}
    classIndDict = {ind: item for ind, item in enumerate(classDict)}
    # output preparation.
    outWords = [wordIndDict[item] for item in words]
    outLabels = [classIndDict[item] for item in labels]
    predictions = [(1 if 0 == item else item) for item in predictions]
    outPredictions = [classIndDict[item] for item in predictions]

    # file operation.
    if not os.path.exists(outsPath):
        os.mkdir(outsPath)
    filename = 'output_'+ str(Type) +'_pre' + str(preTrain) + '_bi' + str(bidirect) + '_' + \
               str(hiddenSize) + '_' + str(batchsize) + '_' + str(learnRate) + '.txt'
    fout = open(outsPath + '/' + filename, 'w')
    for i in range(len(outLabels)):
        fout.write(outWords[i] + ' ' + outLabels[i] + ' ' + outPredictions[i] + '\n')
    fout.close()
    evaluate_conll_file(open(outsPath + '/' + filename, 'r'))

    # save model.
    if not os.path.exists(modsPath):
        os.mkdir(modsPath)
    filename = 'model_'+ str(Type) +'_pre' + str(preTrain) + '_bi' + str(bidirect) + '_' + \
               str(hiddenSize) + '_' + str(batchsize) + '_' + str(learnRate) + '.pth'
    torch.save(model.state_dict(), modsPath + '/' + filename)
    print('[Info] Save the %s model in %s/%s' % (Type, modsPath, filename))
    print('[Info] --------------------------------------------------------------------------------')
    return model

# The program entrance.
if __name__ == "__main__":
    main()