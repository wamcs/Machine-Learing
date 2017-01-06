import matplotlib.pyplot as plot
import numpy as np

def plotPicture(alpha,b,dataMat,labelMat):
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = [];ycord1 = []
    xcord2 = [];ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,0])
            ycord1.append(dataArr[i,1])
        else:
            xcord2.append(dataArr[i,0])
            ycord2.append(dataArr[i,1])
    fig = plot.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c = 'red',marker = 's')
    ax.scatter(xcord2,ycord2,s=30,c = 'green')
    x = np.arange(-2.0,10.0,0.1)
    y = np.array(-(alpha[0,0]*x+b)/alpha[0,1])
    print type(x),type(y)
    print x,y[0]
    ax.plot(x,y[0])
    plot.xlabel('X1')
    plot.ylabel('X2')
    plot.show()
