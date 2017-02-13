# -*- coding:utf-8 -*-
from numpy import *

# 代码来自《机器学习实战》
class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

    def kernelTrans(x,y,kTup):
        m,n = shape(x)
        K = mat(zeros((m,1)))
        if kTup[0] == 'lin':
            K = x*y.T
        elif kTup[0] == 'rbf':
            for j in range(m):
                deltaRow = X[j,:] - A
                K[j] = deltaRow*deltaRow.T
            K = exp(K/(-1*kTup[1]**2))
            #高斯核
        else: raise NameError('not have this kernel yet')
        return K

    def selectJrand(i,m):
        j=i
        while(j==i):
            j = int(random.uniform(0,m))
        return j

    # 确保alpha更新值在L－H间
    def clipAlpha(aj,H,L):
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj

    def calcEk(oS,k):
        fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[k,:]+oS.b)
        Exk = fXk - float(oS.labelMat[k])
        return Exk

    def selectJ(i,oS,Ei):
        maxK = -1
        maxDeltaE = 0
        Ek = 0
        oS.eCache[i] = [1,Ei]
        validEcacheList = nonzero(oS.eCache[:,0].A)[0]
        if (len(validEcacheList)>1):
            for j in validEcacheList:
                if i==j:continue
                Ej = oS.eCache[j,1]
                delta = abs(Ej - Ei)
                if delta>maxDeltaE:
                    maxDeltaE = delta
                    maxK = j
                    Ek = Ej
            return maxK,Ek
        else:
            j = selectJrand(i, oS.m)
            Ek = calcEk(oS, j)
        return j,Ek

    def updataEk(oS,k):
        Ek = calcEk(os,K)
        oS.eCache[k] = [1,Ek]

    def Inner(i,oS):
        if oS.eCache[i,1] == 0:
            Ei = calcEk(oS,i)
        else:
            Ei = oS.eCache[i,1]
        if ((oS.labelMat[i]*Ei<-oS.toler) and (oS.alphas[i] <C)) or ((oS.labelMat[i]*Ei>oS.toler) and (oS.alphas[i] >0)):
            j,Ej = selectJ(i,oS,Ei)
            alphasIold = oS.alphas[i].copy()
            alphasJold = oS.alphas[j].copy()
            if (oS.labelMat[i]!=oS.LabelMat[j]):
                L = max(0, oS.alphas[j] - oS.alphas[i])
                H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
            else:
                L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
                H = min(oS.C, oS.alphas[j] + oS.alphas[i])
            if L==H: print "L==H"; return 0
            eta = oS.K[i,i] + oS.K[j,j] - 2.0*oS.K[i,j]
            if eta <= 0: return 0
            oS.alphas[j] += oS.labelMat[j]*(Ei - Ej)/eta
            oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
            updataEk(oS,j)
            if (abs(oS.alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; return 0
            oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
            updateEk(oS, i)
            b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
            b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
            if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
            elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
            else: oS.b = (b1 + b2)/2.0
            return 1
        else: return 0

    def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):
        oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
        iter = 0
        entireSet = True; alphaPairsChanged = 0
        while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
            alphaPairsChanged = 0
            if entireSet:   #go over all
                for i in range(oS.m):
                    alphaPairsChanged += innerL(i,oS)
                    print "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
                iter += 1
            else:#go over non-bound (railed) alphas
                nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged += innerL(i,oS)
                    print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
                iter += 1
            if entireSet: entireSet = False #toggle entire set loop
            elif (alphaPairsChanged == 0): entireSet = True
            print "iteration number: %d" % iter
        return oS.b,oS.alphas
