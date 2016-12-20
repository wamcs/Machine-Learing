# -*- coding:utf-8 -*-
import numpy as np
import re

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
    vocabList = set([])
    for data in dataSet:
        vocabList = vocabList | set(data)
    return list(vocabList)

# vec 中的元素仅含0,1
def createVector(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print 'the word: %s is not in vocabulary list' %word
    return returnVec

# trainMat是训练集，每个item的长度都是字典的长度
def trainNB(trainMat,trainCategory):
    numTrainBayes = len(trainMat)
    numWords = len(trainMat[0])
    PAbusive = sum(trainCategory)/float(numTrainBayes) # P(c) c belong to {0,1},P(1)
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)      #change to ones()
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainBayes):
        if trainCategory[i] == 1:
            p1Num += trainMat[i]
            p1Denom += sum(trainMat[i])
        else:
            p0Num +=trainMat[i]
            p0Denom += sum(trainMat[i])

    p0Vec = np.log(p0Num/p0Denom)
    p1Vec = np.log(p1Num/p1Denom)
    return p0Vec,p1Vec,PAbusive

# p(c=1|x) = p(x|y=1)*p(y=1)/p(x)
# p(x|y=1) = multiply p(xi|y=1)
# p(y=1) equal to pAb
# p(x)是相同的，所以只需要比较p(x|y=1)*p(y=1)和p(x|y=0)*p(y=0)
def classifyNB(vec2Classify,p0Vec,p1Vec,pAb):

    p1 = sum(vec2Classify*p1Vec)+np.log(pAb)
    p0 = sum(vec2Classify*p0Vec)+np.log(1.0 - pAb)

    if p0>p1:
        return 0
    else:
        return 1

def testingNB():
    postingList,classVec = loadDataSet()
    vocabList = createVocabList(postingList)
    trainMat = []
    for i in postingList:
        trainMat.append(createVector(vocabList,i))
    p0Vec,p1Vec,pAb = trainNB(np.array(trainMat),np.array(classVec))
    testEntry = ['love','my','dalmation']
    thisDoc = np.array(createVector(vocabList,testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0Vec,p1Vec,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(createVector(vocabList,testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0Vec,p1Vec,pAb)

# vec 中的元素每一位都表示该词出现的次数 (bag of word model)
def bagOfModel2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec

def testParse(String):
    regEx = re.compile(r'\W*')
    splitList = regEx.split(String)
    # length of word is bigger than 2,because when it is smaller than 2,the word is preposition or subject
    return [word.lower()  for word in splitList if len(word)>2]

def spamTest():
    docList = []
    classList = []
    for i in range(1,26):
        wordList = testParse(open('email/spam/%d.txt' %i).read())
        docList.append(wordList)
        classList.append(1)
        wordList = testParse(open('email/ham/%d.txt' %i).read())
        docList.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainSet = range(50)
    testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0,len(trainSet)))
        testSet.append(trainSet[randIndex])
        del(trainSet[randIndex])
    trainMat = []
    trainClassSet = []
    for i in trainSet:
        trainMat.append(bagOfModel2Vec(vocabList,docList[i]))
        trainClassSet.append(classList[i])
    p0Vec,p1Vec,pAb = trainNB(np.array(trainMat),trainClassSet)
    errorNum = 0
    for item in testSet:
        wordVec = bagOfModel2Vec(vocabList,docList[item])
        if classifyNB(np.array(wordVec),p0Vec,p1Vec,pAb) != classList[item]:
            errorNum+=1
            print "classification error",docList[item]
    print 'the error rate is ',float(errorNum)/len(testSet)
