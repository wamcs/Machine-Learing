# -*- coding utf-8 -*-
import numpy as np
import csv
import operator

# transform type of number in data set from int to numpy.int
def toInt(array):
    array = np.mat(array)
    m,n = np.shape(array)
    newArray = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            newArray[i,j] = np.int(array[i,j])
    return newArray

def normlizing(array):
    m,n = np.shape(array)
    for i in range(m):
        for j in range(n):
            if array[i,j]!=0:
                array[i,j] = 1
    return array

def loadTestData():
    l=[]
    fr = open('test.csv')
    lines = csv.reader(fr)
    for line in lines:
        l.append(line)
    l.remove(l[0])
    data = np.array(l)
    return normlizing(toInt(data))

def loadTrainData():
    l =[]
    fr =open('train.csv')
    lines = csv.reader(fr)
    for line in lines:
        l.append(line)
    l.remove(l[0])
    l = np.array(l)
    trainsData = l[:,1:]
    trainsLabel = l[:,0]
    return normlizing(toInt(trainsData)),toInt(trainsLabel)

def classify(inx,dataSet,label,K):
    m = dataSet.shape[0]
    testSet = np.tile(inx,(m,1))
    diffMat = dataSet - testSet
    diffMat = diffMat**2
    sqDistance = diffMat.sum(axis = 1)
    distance = sqDistance**0.5
    sortedDistIndicies = distance.argsort()
    classLabel = {}
    for i in range(K):
        votedLabel = label[sortedDistIndicies[i]]
        classLabel[votedLabel] = classLabel.get(votedLabel,0)+1
    result = sorted(classLabel.iteritems(),key = operator.itemgetter(1),reverse=True)
    return result[0][0]

def saveToFile(result):
    fr = open('result.csv','wb')
    writer = csv.writer(fr)
    writer.writerow(['ImageId','Label'])
    for i in result:
        writer.writerow(i)

def handWritingClassTest():
    hoRatio = 0.1
    trainsData,trainsLabel = loadTrainData()
    m = trainsData.shape[0]
    errorCount = 0
    numTest = np.int(m*hoRatio)
    for i in range(numTest):
        resultLabel = classify(trainsData[i],trainsData[numTest:m],trainsLabel[0,numTest:m],5)
        print "the classifier came back with: %d, the real answer is: %d" %(resultLabel,trainsLabel[0,i])
        if resultLabel != trainsLabel[0,i]:
            errorCount +=1
    print 'The error rate is %f' %(errorCount/np.float(numTest))

def handWritingClass():
    testSet = loadTestData()
    trainsData,trainsLabel = loadTrainData()
    m = testSet.shape[0]
    result = []
    index = 1
    for i in range(m):
        resultLabel = classify(testSet[i],trainsData,trainsLabel[0,:],5)
        result.append([index,np.int(resultLabel)])
        print index
        index+=1
    saveToFile(result)
    print 'finish'
