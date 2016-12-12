# -*- coding:utf-8 -*-
from math import log
import csv


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


# 以在dataSet中的item的末一位值为分类标准（一般该值即为分类值），计算信息熵
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

# def calculateScatterIV(dataSet,axis):
#     num = len(dataSet)
#     LabelCounts = {}
#     for i in dataSet:
#         currentLabel = i[axis]
#         if currentLabel not in LabelCounts.keys():
#             LabelCounts[currentLabel] = 0
#         LabelCounts[currentLabel] += 1
#     IV = 0.0
#     for key in LabelCounts:
#         prob = float(LabelCounts[key]) / num
#         IV -= prob * log(prob, 2)
#     return IV
#
# def calculateContinousIV(dataSet,axis,point):
#     num = len(dataSet)
#     small = 0
#     big = 0
#     for i in dataSet:
#         number = float(i[axis])
#         if number < point:
#             small+=1
#         else:
#             big+=1
#     IV = -(float(small)/num*log(float(small)/num,2)+float(big)/num*log(float(big)/num,2))
#     return IV
#


# axis 为特征的维度，dataset为数据集，返回一个dict，以特征为key，分类结果为value
def splitScatterData(dataSet, axis):
    splitData = {}
    for i in dataSet:
        currentLabel = i[axis]
        if currentLabel not in splitData.keys():
            splitData[currentLabel] = []
        temp = i[:axis]
        # extend方法和append方法均为单参数方法，不同的是append将参数以item加入到list中，extend将参数拆开加入到list中
        temp.extend(i[axis + 1:])
        splitData[currentLabel].append(temp)
    return splitData


# 基本思想是取连续值的中点，根据这些中点去划分，计算得到的信息熵，越小越好，但这样的计算量很大
# 考虑将这些特征值排序，只有在决策属性发生改变的地方才需要切开
def splitContinuousData(dataSet, axis):
    # print '=========='
    # print axis
    # print len(dataSet[0])
    splitPoints = []
    sortPoints = sorted(dataSet, key=lambda x: x[axis])

    num = len(sortPoints)
    for i in range(num - 1):
        if sortPoints[i][-1] != sortPoints[i + 1][-1]:
            splitPoints.append((float(sortPoints[i][axis]) + float(sortPoints[i + 1][axis])) / 2)

    bestSplitPoint = 0
    bestEntroy = -1  # 要求最小
    bestFront = []
    bestBack = []

    tempData = [float(i[axis]) for i in dataSet]
    for point in splitPoints:
        front = []
        back = []
        for i in range(num):
            if tempData[i] < point:
                front.append(dataSet[i])
            elif tempData[i] > point:
                back.append(dataSet[i])
        frontEnt = calculateEntropy(front)
        backEnt = calculateEntropy(back)
        tempEnt = (float(len(front)) / num) * frontEnt + (float(len(back)) / num) * backEnt
        if bestEntroy < 0:
            bestEntroy = tempEnt
            bestFront = front
            bestBack = back
            bestSplitPoint = point
        elif bestEntroy > tempEnt:
            bestEntroy = tempEnt
            bestFront = front
            bestBack = back
            bestSplitPoint = point
    # 此处不能用remove，会修改dataSet，改变数据集
    front = []
    back = []
    for i in bestFront:
        temp = i[0:axis]
        temp.extend(i[axis + 1:])
        front.append(temp)

    for i in bestBack:
        temp = i[0:axis]
        temp.extend(i[axis + 1:])
        back.append(temp)

    splitData = {'point': bestSplitPoint, 'small than %f' % float(bestSplitPoint): front,
                 'big than %f' % float(bestSplitPoint): back}
    return splitData, bestSplitPoint, bestEntroy

# 使用C4.5决策树中的信息熵增益率来进行比较 (貌似有问题，连续值如何处理)
def calculateBestAttribute(dataSet, labelSign, Label):
    print '++'
    print len(dataSet[0])
    attrNum = len(dataSet[0]) - 1
    num = float(len(dataSet))
    baseEntropy = calculateEntropy(dataSet)
    bestGain = 0.0
    tempGain = 0.0
    bestFeature = -1
    bestClassSet = {}
    tempClassSet = {}
    for i in range(attrNum):
        if labelSign[i] == Label[1]:
            tempClassSet = splitScatterData(dataSet, i)
            # IV = calculateScatterIV(dataSet,i)
            ent = 0.0
            for key in tempClassSet:
                prob = len(tempClassSet[key]) / num
                ent += prob * calculateEntropy(tempClassSet[key])
            # tempGainRatio = (baseEntropy - ent)/IV
            tempGain = baseEntropy - ent

        elif labelSign[i] == Label[0]:
            tempClassSet, bestSplitPoint, bestEntroy = splitContinuousData(dataSet, i)
            # IV = calculateContinousIV(dataSet,i,bestSplitPoint)
            # tempGainRatio = (baseEntropy - bestEntroy)/IV
            tempGain = baseEntropy - bestEntroy

        if (bestGain < tempGain):
            bestGain = tempGain
            bestFeature = i
            bestClassSet = tempClassSet
        print 'feature %s is %f' %(dataSet[0][i],tempGain)
    print 'bestFeature is %d' %bestFeature

    return bestFeature, bestClassSet


def create(filename, labelSign):
    dataSet, labelVec = load(filename)
    LabelSet = set(labelSign)
    # 需要对数据进行预处理，对于连续的属性和离散的属性应该有标签区分,'s'(scatter),'c'(continuous)
    # 建议使用标记‘s’(scatter),'c'(continuous)如果使用别的标记，注意要使连续的在离散的前面
    Label = []
    for i in LabelSet:
        Label.append(i)
    Label.sort()
    tree = creatTree(dataSet, labelVec, labelSign, Label)
    return tree


def creatTree(dataSet, labelVec, labelSign, Label):
    # print '*****'
    # print labelSign
    vec = [example[-1] for example in dataSet]
    if len(set(vec)) == 1:
        return vec[0]

    feature, classSet = calculateBestAttribute(dataSet, labelSign, Label)
    choosedLabelSign = labelSign[feature]
    sign = labelVec[feature]
    labelVec.remove(sign)
    tree = {sign: {}}
    tree[sign] = classSet
    for key in classSet:
        if key == 'point':
            continue
        labelSign.remove(choosedLabelSign)
        tree[sign][key] = creatTree(classSet[key], labelVec, labelSign, Label)
        labelSign.insert(feature, choosedLabelSign)
    labelVec.insert(feature, sign)
    return tree


def classify(trainfile, labelSign, testfile, testSign):
    tree = create(trainfile, labelSign)
    testdata, testVec = load(testfile)
    errorCount = 0.0
    for item in testdata:
        result = classifier(tree, item, testVec, testSign)
        if result != item[-1]:
            errorCount += 1
        print 'item %d classify result is %s, the real result is %s' % (testdata.index(item), result, item[-1])
    print 'the error rate is %f' % (float(errorCount) / len(testdata))


def classifier(tree, testdata, testVec, testSign):
    firstStr = tree.keys()[0]
    index = testVec.index(firstStr)
    if testSign[index] == 's':
        secondDict = tree[firstStr]
        feature = secondDict[testdata[index]]
        if isinstance(feature, dict):
            return classifier(feature, testdata, testVec, testSign)
        else:
            return feature

    elif testSign[index] == 'c':
        secondDict = tree[firstStr]
        point = secondDict['point']
        if float(testdata[index]) < point:
            feature = secondDict['small than %f' % float(point)]
            if isinstance(feature, dict):
                return classifier(feature, testdata, testVec, testSign)
            else:
                return feature
        elif float(testdata[index]) > point:
            feature = secondDict['big than %f' % float(point)]
            if isinstance(feature, dict):
                return classifier(feature, testdata, testVec)
            else:
                return feature


def test():
    sign = ['s', 's', 's', 's', 's', 's', 'c', 'c']
    tree = create('trainData.csv', sign)
    return tree


def testClass():
    sign = ['s', 's', 's', 's', 's', 's', 'c', 'c']
    classify('trainData.csv', sign, 'trainData.csv', sign)
