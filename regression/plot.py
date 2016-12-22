import matplotlib.pyplot as plot
import numpy as np

def plotBestFit(weights,dataMat,labelMat):
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = [];ycord1 = []
    xcord2 = [];ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plot.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c = 'red',marker = 's')
    ax.scatter(xcord2,ycord2,s=30,c = 'green')
    x = np.arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plot.xlabel('X1')
    plot.ylabel('X2')
    plot.show()
