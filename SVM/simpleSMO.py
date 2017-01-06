# -*- coding:utf-8 -*-
from numpy import *
import plot as plot

def loadData(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

# 选择与alpha（i）同时变换的alpha（j），保证变换之后总和不改变
def selectJrand(i,m):
    j=i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

# 确保alpha更新值在L－H间
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

# 详细数学推导可以看Platt 1998 论文 《Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines》
def smoSimple(dataMatIn,LabelMatIn,C,toler,maxIter):
    dataMat = mat(dataMatIn)
    labelMat = mat(LabelMatIn).transpose()
    b = 0
    m,n = shape(dataMat)
    alphas = mat(zeros((m,1)))
    iter = 0 # i means cycle times
    while(iter<maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T * (dataMat * dataMat[i,:].T))+ b
            Ei = fXi - float(labelMat[i])
            if((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labelMat).T * (dataMat * dataMat[j,:].T))+ b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0,alphas[j]-alphas[i])
                    H = min(C,C+alphas[j]-alphas[i])
                else:
                    L = max(0,alphas[j]+alphas[i]-C)
                    H = min(C,alphas[i]+alphas[j])
                if L==H:
                    print 'L==H'
                    continue #alpha没有可变的余地
                eta = dataMat[i,:]*dataMat[i,:].T + dataMat[j,:]*dataMat[j,:].T -2.0*dataMat[i,:]*dataMat[j,:].T
                if eta <= 0:
                    print 'eta<=0';continue # 由kernel的非负定性，eta应该大于0，另外的情况出现几率小，简单实现中不做处理
                alphas[j] += labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j]-alphaJold)<0.00001):
                    print 'j not moving enough'
                    continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])

                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMat[i,:]*dataMat[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMat[i,:]*dataMat[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMat[i,:]*dataMat[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMat[j,:]*dataMat[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged+=1
                print "iter:%d,i: %d,pairs changed %d" %(iter,i,alphaPairsChanged)
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        #print "iteration number: %d" % iter
    w = multiply(alphas,labelMat).T * dataMat
    return b,alphas,w

def test():
    reload(plot)
    C = 0.6
    toler = 0.001
    maxIter = 40
    dataMat,labelMat = loadData('testSet.txt')
    b,alphas,w = smoSimple(dataMat,labelMat,C,toler,maxIter)
    print b,alphas
    plot.plotPicture(w,b,dataMat,labelMat)
