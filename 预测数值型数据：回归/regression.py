
from numpy import *



def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.sprip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLint[-1]))
    return dataMat, labelMat

"""
简单的线性回归函数
该函数用来计算最佳拟合直线的系数
"""
def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr)
    xtx = xMat.T * xMat
    #判断矩阵的行列式是否为0，为0时， 矩阵将不能直接的求逆
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    #计算拟合直线的系数
    ws = xTx.I * (xMat.T*yMat)
    return ws
        
"""
局部加权线性回归函数，给定X空间中的任意一点，计算出对应的预测值yHat
参数：
testPoint：表示多维空间的一个点，以向量的形式来表示
xArr:表示训练数据集
yArr：对应训练集的真实值
k：局部加权线性回归函数采用“核”来对附近的点赋予更高的权重，这里核的类型我们采用高斯核，
   k表示的就是高斯核的参数，控制着权重衰减的速度，离测试点越远，得到的权重就越小，而且
   呈现的是指数级下降
"""
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    #对于预测一个点，都必须遍历所有的训练数据集，因为每一个数据点都有对测试点的不同的权重
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        #得到训练数据集数据点对该测试点的权重
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx)  =0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I*(xMat.T*(weights*yMat))
    #返回预测的数据值
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


"""
岭回归,该函数用于计算岭回归系数
"""
def ridgeRegres(xMat, yMat, lam = 0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse ")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws

"""
测试不同的lambda对回归系数的影响
"""
def ridgeTest(xArr, yArr):
    #对数据进行标准化处理
    xMat = mat(xArr)
    yMean = mean(yMat, 0)
    #真实值减去均值
    yMat  =yMat - yMean
    xMeans = mean(xMat, 0)
    #求方差
    xVar = var(xMat, 0)
    #所有特征值都减去各自的均值再除以方差
    xMat = (xMat - xMeans)/xVar
    #测试30组数据
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i-10))
        wMat[i, :] = ws.T
    return wMat


"""
该函数用于计算误差，平方误差
"""
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()

"""
向前逐步回归，该算法属于一种贪心算法，即每一步都尽可能的减少误差，一开始所有的权重都
设置为1，然后每一步所做的决策是对某个权重增加或减少一个很小的值。贪心算法在所有的特征
上运行两次循环，分别计算增加或减少该特征对误差的影响。这里使用的是平方误差，可以直接的
使用上面的rssError函数。该误差的初始值设置为无穷大，经过与所有的误差比较后取最小的误差。
整个过程循环迭代进行
参数：
eps：表示每一次改变W的步长
numIt：表示迭代的次数
"""
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m,n = shape(xMat)
    #用于存储每一次迭代后的权重向量
    returnMat = zeros((numIt, n))
    ws = zers((n, 1))
    wsTest = ws.copy()
    wsMat = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf
        #遍历所有的特征得到最好的权重向量
        for j in range(n):
            #对于每一个特征，调整其对应的权重，得到该特征对应的最好的权重，从正、负两个方向调整
            for sign in [-1, 1]:
                wsTest = ws.copy()
                #改变特征的权重
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMat = wsTest
        ws = wsMat.copy()
        returnMat[i, :] = ws.T
    return returnMat


"""
交叉验证测试岭回归
参数：
xArr:测试数据的向量list
yArr：测试数据的标签list
numVal：交叉验证的次数
"""
def crossValidation(xArr, yArr, numVal= 10):
    m = len(yArr)
    indexList = range(m)
    errorMat  =zeros((numVal, 30))
    #创建训练数据集和测试数据集
    for i in range(numVal):
        trainX = []
        trainY = []
        testX = []
        testY = []
        random.shuffle(indexList)
        for j in range(m):
            if j < m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
    #用来保存岭回归中的所有回归系数
    #函数ridgeTest使用30个不同的lambda值创建了30组不同的回归系数
    wMat = ridgeTest(trainX, trainY)
    #利用测试数据，用30组回归系数来循环的测试回归的效果
    for k in range(30):
        #利用训练时的参数将测试数据标准化
        matTestX = mat(testX)
        matTrainX = mat(trainX)
        meanTrain = mean(matTrainX,0)
        varTrain = var(matTrainX,0)
        matTestX = (matTestX - meanTrain)/varTrain
        #根据回归系数获得测时数据的预测值
        yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)
        #得到误差值
        errorMat[i,k] = rssError(yEst.T.A, array(testY))
    meanErrors = mean(errorMat, 0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors == minMean)]
    xMat = mat(xArr)
    yMat = mat(yArr).T
    meanX = mean(xMat,0)
    varX = var(xMat,0)
    unReg = bestWeights/varX
    print("the best model from Ridge Regression is :")
    print(unReg)
    print("with constant term:")
    print(-1*sum(multiply(meanX, unReg)) + mean(yMat))



