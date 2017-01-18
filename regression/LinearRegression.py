
# -*- coding utf-8 -*-
import numpy as np



def loadDataSet(filename):
    classLabel = []
    dataMat = []
    fr = open(filename)
    lines = fr.readlines()
    for line in lines:
        lineArr = line.strip().split()
        temp = [1.0]
        for i in lineArr[:-1]:
            temp.append(float(i))
        dataMat.append(temp)
        classLabel.append(lineArr[-1])
    return classLabel,dataMat

#利用梯度下降求解w
def GradientAscent(alpha = 1,x,y,tol = 0.0001):
    classLabel = np.mat(y)
    dataMat = np.mat(x).transpose()
    m,n = np.shape(dataMat)
    weight = np.ones((1,m))
    tempWeight = np.zeros((1,m))
    while abs(np.max(weight - tempWeight))>tol:
        tempWeight = weight
        h = weight*dataMat
        bais = h - classLabel
        weight = weight - alpha * bais * dataMat.transpose()
    return weight

#利用矩阵的迹和定义的矩阵导数求解
def standRegress(xArr,yArr):
    X = np.mat(xArr)
    Y = np.mat(yArr)
    XTX = X.T*X
    if np.linalg.det(XTX) == 0.0:
        print 'the matrix cant do inverse'
        return
    return XTX.I*(X.T*Y)

# ridge regression (W = (X.T*X+lamda*I).I*X.I*y)
def ridgeRegress(xMat,yArr,lam = 0.2):
    X = np.mat(xMat)
    Y = np.mat(yArr)
    denom = X.T*X+np.eye(np.shape(X)[1])*lam
    if denom == 0:
        print 'the matrix cant do inverse'
        return
    return denom.I*(X.I*Y)


# 前向逐步回归算法，和lasso类似，但是更简单，是一种贪心算法
# eps:每次迭代调整的步长，numIt，迭代次数
def stageWise(xArr,yArr,eps =0.01,numIt = 100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat,0)
    # 数据标准化，分布满足期望为0，方差为1
    yMat = yMat-yMean
    xMat = regularize(xMat)
    m,n = np.shape(xMat)
    returnMat = np.zeros((numIt,n))
    # 所有的权重设为1，每一步决策都对权重进行更改
    ws = np.zeros((n,1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError = np.inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] +=eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        return[i,:] = ws.T
    return returnMat

#局部加权线性回归，通过添加核，来使得附近点有更高的权重
# testPoint 为xArr 某一行
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(Xmat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testPoint - Xmat[j:]
        weights[j,j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0:
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def regularize(xMat):
    inMat = xMat.copy()
    inMeans = mean(inMat,0)
    inVar = var(inMat,0)
    inMat = (inMat - inMeans)/inVar
    return inMat

#最小二乘
def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

def classifer(trainfile,testfile):
    trainLabel,traindataMat = loadDataSet(trainfile)
    testLabel,testdataMat = loadDataSet(testfile)

    weight1 = GradientAscent(traindataMat,trainLabel)
    weight2 = standRegress(traindataMat,trainLabel)

    label = np.mat(testLabel).transpose()
    data = np.mat(testdataMat)

    test1 = weight1 * data.T
    test2 = weight2 * data.T

    n = np.shape(label)[1]
    err  = [0] * n
    err1 = err.copy()
    err2 = err.copy()
    err1[label != test1] = 1
    err2[label != test2] = 1

    print 'err1:',sum(err1)/float(n)
    print 'err2:',sum(err2)/float(n)
