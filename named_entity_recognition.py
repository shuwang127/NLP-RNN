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
import torch

# global path
rootPath = './'
dataPath = rootPath + '/conll2003/'
tempPath = rootPath + '/temp/'
# global variable
numEpoch = 1
perEpoch = 1
learnRate = 0.001

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
    DemoRNN(dTrain, lTrain, dValid, lValid, dTest, lTest, preWeights, preTrain=True, Type='RNN', bidirect=False)
    DemoRNN(dTrain, lTrain, dValid, lValid, dTest, lTest, preWeights, preTrain=True, Type='RNN', bidirect=True)
    DemoRNN(dTrain, lTrain, dValid, lValid, dTest, lTest, preWeights, preTrain=True, Type='LSTM', bidirect=False)
    DemoRNN(dTrain, lTrain, dValid, lValid, dTest, lTest, preWeights, preTrain=True, Type='LSTM', bidirect=True)
    DemoRNN(dTrain, lTrain, dValid, lValid, dTest, lTest, preWeights, preTrain=True, Type='GRU', bidirect=False)
    DemoRNN(dTrain, lTrain, dValid, lValid, dTest, lTest, preWeights, preTrain=True, Type='GRU', bidirect=True)
    DemoRNN(dTrain, lTrain, dValid, lValid, dTest, lTest, preWeights, preTrain=False, Type='LSTM', bidirect=True)
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
    maxlen = max(maxlenTrain, maxlenValid, maxlenTest)
    # append 0s at the end of sentence to the max length.
    for sentence in dataTrain:
        sentence.extend(['0', '<pad>'] for i in range(maxlen - len(sentence)))
    for sentence in dataValid:
        sentence.extend(['0', '<pad>'] for i in range(maxlen - len(sentence)))
    for sentence in dataTest:
        sentence.extend(['0', '<pad>'] for i in range(maxlen - len(sentence)))

    # clear up the vocabulary.
    for word, label in vocab.items():
        vocab[word] = list(set(label))
    vocab['0'] = '<pad>'
    print('[Info] Get %d vocabulary words successfully.' % (len(vocab)))

    return dataTrain, dataValid, dataTest, vocab

def GetDict(vocab):
    # get word and class dictionary.
    wordDict = {word: index for index, word in enumerate(vocab)}
    classDict = {'<pad>': 0, 'O': 1, 'B-ORG': 2, 'B-PER': 3, 'B-LOC': 4, 'B-MISC': 5, 'I-ORG': 6, 'I-PER': 7, 'I-LOC': 8, 'I-MISC': 9}
    # return
    return wordDict, classDict

def GetMapping(data, wordDict, classDict):
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
    def __init__(self, preWeights, preTrain=True, Type='RNN', bidirect=False, hiddenSize=256):
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
        if 'RNN' == Type:
            self.rnn = nn.RNN(input_size=numDims, hidden_size=hiddenSize, batch_first=True, bidirectional=bidirect)
        elif 'LSTM' == Type:
            self.rnn = nn.LSTM(input_size=numDims, hidden_size=hiddenSize, batch_first=True, bidirectional=bidirect)
        elif 'GRU' == Type:
            self.rnn = nn.GRU(input_size=numDims, hidden_size=hiddenSize, batch_first=True, bidirectional=bidirect)
        else:
            print('[Error] RNN Type \'%s\' invalid!' % (Type))
        # fully-connected layer.
        self.fc = nn.Linear(in_features=hiddenSize*numBiDirect, out_features=10)
        self.sm = nn.Softmax()

    def forward(self, x):
        embeds = self.embedding(x)
        out, _ = self.rnn(embeds)
        out = out.contiguous().view(-1, out.shape[2])
        yhats = self.fc(out)
        return yhats

def DemoRNN(dTrain, lTrain, dValid, lValid, dTest, lTest, preWeights, preTrain=True, Type='RNN', bidirect=False, hiddenSize=256, batchsize=256):
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
    model = RecurrentNeuralNetwork(preWeights, preTrain=preTrain, Type=Type, bidirect=bidirect, hiddenSize=hiddenSize)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print('[Demo] --- RNNType: %s | HiddenNodes: %d | Bi-Direction: %s | Pre-Trained: %s ---' % (Type, hiddenSize, bidirect, preTrain))
    # optimizing with stochastic gradient descent.
    optimizer = optim.Adam(model.parameters(), lr=learnRate)
    # seting loss function as mean squared error.
    criterion = nn.CrossEntropyLoss(weight=lbWeights)
    # memory
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # run on each epoch.
    accList = [0]
    for epoch in range(numEpoch):
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
        gc.collect()
        torch.cuda.empty_cache()
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
        gc.collect()
        torch.cuda.empty_cache()
        # valid accuracy.
        padIndex = [ind for ind, lb in enumerate(labels) if lb == 0]
        for ind in sorted(padIndex, reverse=True):
            del predictions[ind]
            del labels[ind]
        accValid = accuracy_score(labels, predictions) * 100
        accList.append(accValid)

        # output information.
        if 0 == (epoch + 1) % perEpoch:
            print('[Epoch %03d] loss: %.3f, train acc: %.3f%%, valid acc: %.3f%%' % (epoch + 1, lossTrain, accTrain, accValid))
        # save the best model.
        if accList[-1] > max(accList[0:-1]):
            torch.save(model.state_dict(), tempPath + '/model.pth')
        # stop judgement.
        if (epoch + 1) >= 5 and accList[-1] < min(accList[-5:-1]):
            break

    # load best model.
    model.load_state_dict(torch.load(tempPath + '/model.pth'))

    # test period
    xTest = torch.from_numpy(dTest).long().cuda()
    yTest = torch.from_numpy(lTest).long().cuda()
    test = torchdata.TensorDataset(xTest, yTest)
    testloader = torchdata.DataLoader(test, batch_size=batchsize, shuffle=False)
    # testing phase.
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for iter, (data, label) in enumerate(testloader):
            data = data.to(device)
            label = label.contiguous().view(-1)
            label = label.to(device)
            yhat = model.forward(data)  # get output
            # statistic
            preds = yhat.max(1)[1]
            predictions.extend(preds.int().tolist())
            labels.extend(label.int().tolist())
            torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    # testing accuracy.
    padIndex = [ind for ind, lb in enumerate(labels) if lb == 0]
    for ind in sorted(padIndex, reverse=True):
        del predictions[ind]
        del labels[ind]
    accuracy = accuracy_score(labels, predictions) * 100
    print('[Info] Testing accuracy: %.3f%%' % (accuracy))

    '''
    print("--- Formatting Results for conlleval.py Official Evaluation --- %s seconds ---" % (round((time.time() - start_time),2)))
    formattedcontexts = []
    formattedlabels = []
    formattedpredictions = []
    for element in labelsfull: #convert to real words and labels
        formattedlabels.extend(labelindex[element])
    for element in predictionsfull:
        if element == 0:
            element = 1 #remove stray <pad> predictions
        formattedpredictions.extend(labelindex[element])
    for element in contextfull:
        formattedcontexts.extend(wordindex[element])
    #write to file
    fname = 'results/{}--bidir={}--hidden_size={}--pretrain={}--results.txt'.format(RNNTYPE,bidirectional,hidden_dim,pretrained_embeddings_status)
    if os.path.exists(fname):
        os.remove(fname)
    f = open(fname,'w')
    for (i,element) in enumerate(labelsfull):
        f.write(formattedcontexts[i] + ' ' + formattedlabels[i] + ' ' + formattedpredictions[i] + '\n')
    f.close()
    print('--- {}--bidir={}--hidden_size={}--pretrain={}--results ---'.format(RNNTYPE,bidirectional,hidden_dim,pretrained_embeddings_status))
    evaluate_conll_file(open(fname,'r')) #evaluate using conll script
    '''
    return model

# The program entrance.
if __name__ == "__main__":
    main()