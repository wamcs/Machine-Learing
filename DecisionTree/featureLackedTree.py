# -*- coding:utf-8 -*-
'''
created on Nov 19 2016

@author :wamcs
'''
from math import log
import csv

# 不考虑连续值情况

# 加载数据
def load(filename):
    fr = open(filename)
    lines = csv.reader(fr)
    data = []
    for line in lines:
        data.append(line[1:])
    labelVec = data[0]
    data.remove(data[0])
    return data, labelVec


def calculateEntropy(dataSet):
    num = len(dataSet)
    LabelCounts = {}
    for i in dataSet:
        currentLabel = i[-1]
        if currentLabel not in LabelCounts.keys():
            LabelCounts[currentLabel] = 0
        LabelCounts[currentLabel] += 1
    Entropy = 0.0
    for key in LabelCounts:
        prob = float(LabelCounts[key]) / num
        Entropy -= prob * log(prob, 2)
    return Entropy


def splitData(dataSet,axis):
    num = len(dataSet)
    splitData = {}
    lackList = []
    selectDataSet = []
    validNum = 0
    for data in dataSet:
        temp = data[:axis]
        if data[axis] == '':
            temp.extend(data[axis+1:])
            lackList.append(temp)
            continue
        validNum+=1
        temp.extend(data[axis+1:])
        if data[axis] not in splitData.keys():
            splitData[data[axis]] = []
        splitData[data[axis]].append(temp)
        selectDataSet.append(temp)

    # splitData为分类后的数据集合，selectDataSet为去除残缺值之后的数据集，lacklist为指定特征残缺的数据集，validNum为指定特征不残缺的数据数量
    return splitData,selectDataSet,lackList,validNum

def chooseBestFeature(dataSet):
    print '_______'
    num = len(dataSet)
    attrNum = len(dataSet[0]) - 1
    bestGain = 0.0
    tempGain = 0.0
    bestFeature = -1
    bestClassSet = {}
    tempClassSet = {}
    tempLackList = []
    # 思想是将指定特征残缺的数据去除之后，用新的数据集来计算信息熵增益
    for i in range(attrNum):
        tempClassSet,selectDataSet,lackList,vaildNum = splitData(dataSet,i)
        baseEntropy = calculateEntropy(selectDataSet)
        classEntropy = 0
        for key in tempClassSet:
            num = len(tempClassSet[key])
            classEntropy-=(float(num)/vaildNum)*calculateEntropy(tempClassSet[key])
        tempGain = baseEntropy - classEntropy
        if (bestGain < tempGain):
            bestGain = tempGain
            bestFeature = i
            bestClassSet = tempClassSet
            tempLackList = lackList
        print 'feature %s is %f' %(dataSet[0][i],tempGain)
    for key in bestClassSet:
        for data in tempLackList:
            bestClassSet[key].append(data)

    print 'bestFeature is %d' %bestFeature

    return bestClassSet,bestFeature

def create(filename):
    dataSet, labelVec = load(filename)
    tree = creatTree(dataSet, labelVec)
    return tree


def creatTree(dataSet, labelVec):
    vec = [example[-1] for example in dataSet]
    if len(set(vec)) == 1:
        return vec[0]

    classSet,feature = chooseBestFeature(dataSet)
    if feature == -1:
        return 'unkown'
    sign = labelVec[feature]
    labelVec.remove(sign)
    tree = {sign: {}}
    tree[sign] = classSet
    for key in classSet:
        tree[sign][key] = creatTree(classSet[key], labelVec)
    labelVec.insert(feature, sign)
    return tree


def classify(trainfile, testfile):
    tree = create(trainfile)
    testdata, testVec = load(testfile)
    errorCount = 0.0
    for item in testdata:
        result = classifier(tree, item, testVec)
        if result != item[-1]:
            errorCount += 1
        print 'item %d classify result is %s, the real result is %s' % (testdata.index(item), result, item[-1])
    print 'the error rate is %f' % (float(errorCount) / len(testdata))


def classifier(tree, testdata, testVec):
    firstStr = tree.keys()[0]
    index = testVec.index(firstStr)
    secondDict = tree[firstStr]
    feature = secondDict[testdata[index]]
    if isinstance(feature, dict):
        return classifier(feature, testdata, testVec)
    else:
        return feature
def test():
    tree = create('trainLackData.csv')
    return tree
