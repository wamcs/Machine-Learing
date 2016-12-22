import LogRegression as logist
import numpy as np
import plot as plt

def loadDataSet():
    classLabel = []
    dataMat = []
    fr = open('testSet.txt')
    lines = fr.readlines()
    for line in lines:
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        classLabel.append(int(lineArr[2]))
    return classLabel,dataMat

def testLogist():
    alpha =0.001
    tol = 0.001
    classLabel,dataMat = loadDataSet()
    testCount = 20
    testList = []
    trainList = range(len(classLabel))
    for i in range(testCount):
        randomIndex = int(np.random.uniform(0,len(trainList)))
        testList.append(randomIndex)
        del(trainList[randomIndex])

    trainMat = []
    trainLabel = []
    testMat = []
    testLabel = []
    for i in testList:
        testMat.append(dataMat[i])
        testLabel.append(classLabel[i])
    for i in trainList:
        trainMat.append(dataMat[i])
        trainLabel.append(classLabel[i])
    weight = logist.GradientAscent(alpha,trainMat,trainLabel,tol)
    weight1 = logist.stocGradAscent0(alpha,trainMat,trainLabel)
    # weight2 = logist.stocGradAscent1(trainMat,trainList)
    logist.classifer(weight,testMat,testLabel)
    return classLabel,dataMat,weight1


def testPlot():
    classLabel,dataMat,weight = testLogist()
    plt.plotBestFit(np.array(weight)[0],dataMat,classLabel)
