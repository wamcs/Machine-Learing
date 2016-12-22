import numpy as np


def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def GradientAscent(alpha,x,y,tol):
    classLabel = np.mat(y)
    dataMat = np.mat(x).transpose()
    m,n = np.shape(dataMat)
    weight = np.ones((1,m))
    tempWeight = np.zeros((1,m))
    while abs(np.max(weight - tempWeight))>tol:
        tempWeight = weight
        h = sigmoid(weight*dataMat)
        bais = classLabel - h
        weight = weight + alpha * bais * dataMat.transpose()
    return weight

def stocGradAscent0(alpha,x,y):
    dataMat = np.mat(x)
    m,n = np.shape(dataMat)
    weight = np.ones((1,n))
    for i in range(m):
        h = sigmoid(weight*dataMat[i].transpose())
        bais = y[i] - h
        weight = weight +alpha *bais*dataMat[i]
    return weight

# has bug
# def stocGradAscent1(x,y,num = 150):
#     dataMat = np.mat(x)
#     m,n = np.shape(dataMat)
#     weight = np.ones((1,n))
#     for j in range(num):
#         dataIndex = range(m)
#         for i in range(m):
#             alpha = 1/(1.0+j+i)+0.01
#             randomIndex = int(np.random.uniform(0,len(dataIndex)))
#             h = sigmoid(weight*dataMat[randomIndex].transpose())
#             bais = y[randomIndex] - h
#             weight = weight +alpha *bais*dataMat[randomIndex]
#             del(dataIndex[randomIndex])
#     return weight

def classifer(weight,testMatIn,testLabel):
    labelMat = np.mat(testLabel)
    testMat = np.mat(testMatIn).transpose()
    m,n = np.shape(testMat)
    errorCount = 0
    classifer = np.array(weight * testMat)
    for i in range(n):
        result = 0
        if sigmoid(classifer[0][i])>0.5:
            result = 1
        if result != testLabel[i]:
            errorCount += 1
            print 'item',i,'is error'
    print 'error rate is %f' %(float(errorCount)/n)
