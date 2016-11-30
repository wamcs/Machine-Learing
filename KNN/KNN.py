# -*- coding:utf-8 -*-
import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
## 注释

# @Parameter inx is the data set which will be cxlassified
# @Parameter dataSet is the data which will be used to train this algorithm
# @Parameter label is label vector
# @Parameter K is the number of points that are close to the input point
# the distance is Euclidean distance
def classify(inx,dataSet,label,K):
    m = dataSet.shape[0]
    diffMat = np.tile(inx,(m,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistance = np.sum(sqDiffMat,axis=1)
    distance = sqDistance**0.5
    sortedDistIndicies = distance.argsort()
    #rgsort function return the index of item in the array order by size
    labelClass = {}
    for i in range(K):
        votedLabel = label[sortedDistIndicies[i]]
        labelClass[votedLabel] = labelClass.get(votedLabel,0)+1
    sortedClassCount = sorted(labelClass.iteritems(),key = operator.itemgetter(1),reverse=True)
    # operator.itemgetter(a,b,c,...) return a function,which return the data of input object in dimension a,b,c,...,the default value of paramete 'reverse' is False,ASC
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    lines = fr.readlines()
    numberOfLines = len(lines)
    Mat = np.zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in lines:
        line = line.strip()
        listFormLine = line.split('\t')
        Mat[index,:] = listFormLine[0:3]
        classLabelVector.append(np.int(listFormLine[-1]))
        index+=1
    return Mat,classLabelVector

def drawData(filename):
    dataMat,dataLable = file2matrix(filename)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0],dataMat[:,1],15.0*np.array(dataLable),15.0*np.array(dataLable))
    plt.show()
def autoNorm(dataSet):
    minValues = dataSet.min(axis = 0)
    maxValues = dataSet.max(axis = 0)
    ranges = maxValues - minValues
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minValues,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minValues

def datingClassTest():
    hoRatio = 0.1
    dataMat,dataLabel = file2matrix('/Users/kaili/CodeStore/Machine-Learning/KNN/text.txt')
    normDataSet,ranges,minValues = autoNorm(dataMat)
    m = normDataSet.shape[0]
    numTest = np.int(m*hoRatio)
    errorCount = 0
    for i in range(numTest):
        classifierResult = classify(normDataSet[i],normDataSet[numTest:m],dataLabel[numTest:m],3)
        print "the classifier came back with : %d,the real answer is %d" %(classifierResult,dataLabel[i])
        if (classifierResult != dataLabel[i]):
            errorCount +=1.0
    print "the total error rate is %f" %(errorCount/np.float(numTest))

def img2vector(filename):
    imgVector = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            imgVector[0,32*i+j] = np.int(line[j])
    return imgVector

def handWritingClassTest():
    hwLabels = []
    trainingFiledList = listdir('/Users/kaili/Desktop/machineLearningInAction/machinelearninginaction/Ch02/digits/trainingDigits')
    m = len(trainingFiledList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFiledList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = np.int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('/Users/kaili/Desktop/machineLearningInAction/machinelearninginaction/Ch02/digits/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('/Users/kaili/Desktop/machineLearningInAction/machinelearninginaction/Ch02/digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = np.int(fileStr.split('_')[0])
        vetorTest = img2vector('/Users/kaili/Desktop/machineLearningInAction/machinelearninginaction/Ch02/digits/testDigits/%s' % fileNameStr)
        classifierResult = classify(vetorTest,trainingMat,hwLabels,3)
        print 'the classifier came back with : %d,the real answer is: %d' %(classifierResult,classNumStr)
        if classifierResult != classNumStr:
            errorCount+=1
            print 'error is %s' %fileStr
    print 'the total error rate is %f' %(errorCount/np.float(mTest))
