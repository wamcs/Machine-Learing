# -*- coding:utf-8 -*-

# AdaBoosting是元算法中最经典的一种，基函数为单层决策树或阀值函数，聚合模型为加法模型，损失函数为指数函数，实际上基函数可以为任意的分类器
# 具体的算法参考《机器学习实战》，算法分析参考《机器学习》－周志华 & http://kubicode.me/2016/04/18/Machine%20Learning/AdaBoost-Study-Summary/#AdaBoost训练误差分析


from numpy import *

#阀值函数
# dim: dim of feature
# threshVal:阀值
# threIneq: 'lt' 小于阀值对应为负数据  'gt' 大于阀值对应为负数据
def stumpClassify(dataMat,dim,threshVal,threshIneq):
    retArr = ones((shape(dataMat)[0],1))
    if threshIneq == 'lt':
        retArr[dataMat[:,dim]<=threshVal] = -1.0
    else:
        retArr[dataMat[:,dim]>threshVal] = -1.0
    return retArr

# 单层决策数生成
# ＊D:权重向量
def buildStump(dataArr,classLabels,D):
    m,n = shape(dataArr)
    minError = inf
    bestClassEst = mat(ones((m,1)))
    bestStump = {}
    numStep = 10.0
    for i in range(n):
        #对每个特值利用最大最小值来确定步长
        rangeMin = dataArr[:,i].min()
        rangeMax = dataArr[:,i].max()
        stepSize = (rangeMax-rangeMin)/numStep
        # 从－1起是为了考虑到所有数据都划分到一边的情况
        for j in range(-1,int(numStep)+1):
            for threshIneq in ['lt','gt']:
                threshVal = (rangeMin+float(j)*stepSize)
                #利用阀值函数来生成单层决策树
                predictArr = stumpClassify(dataArr,i,threshVal,threshVal)
                errArr = mat(ones((m,1)))
                errArr[predictArr == classLabels]=0
                weightedError = D.T * errArr
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictArr.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = threshIneq
    return bestStump,minError,bestClassEst

    #AdaBoost主函数
def adaBoostTrainDs(dataArr,classLabels,numIt=40):
    dataMat = mat(dataArr)
    labelMat = mat(classLabels).T
    weakClassArr = []
    m = shape(dataMat)[0]
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataMat,labelMat,D)
        #print 'D:',D.T
        #防止出现分母为0
        alpha = float(0.5*log(1.0-error)/max(error,1e-16))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)

        #Dn+1 为 Dn * exp(-alpha*f(x)*Hn(x)) 单位化
        expon = multiply(-1*alpha*labelMat,classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        #这个实际就是在增加元来调整结果
        aggClassEst += alpha*classEst
        aggErrors = multiply(sign(aggClassEst)!=labelMat,ones((m,1)))
        errorRate = aggErrors.sum()/m
        #print "total error:",errorRate,"\n"
        if errorRate == 0:break
    return weakClassArr

def adaClassify(dataArr,classifierArr):
    dataMat = mat(dataArr)
    m = shape(dataMat)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEnt = stumpClassify(dataMat,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEnt
    return sign(aggClassEst)

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def test():
    trainData,trainLabel = loadDataSet('horseColicTraining2.txt')
    testData,testLabel = loadDataSet('horseColicTest2.txt')
    weakClassArr = adaBoostTrainDs(trainData,trainLabel)
    classResult = adaClassify(testData,weakClassArr)
    test = mat(testLabel).T
    m = shape(mat(trainData))[0]
    errorArr = mat(zeros((m,1)))
    errorArr[test == classResult] = 0
    errorArr[test!=classResult] = 1
    errorRate = errorArr.sum()/m
    print 'errorRate is',errorRate,'\n'
