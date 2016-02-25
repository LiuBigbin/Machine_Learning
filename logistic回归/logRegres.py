from numpy import *

"""
该函数根据训练数据文件的名称，将文件数据加载到numpy数组中
"""
def loadDataSet(trainningFileName):
    dataMat = []
    labelMat = []
    fr = open(trainningFileName)
    for line in fr.readlines():
        lineArr = line.strip().split()
        #数据以数值的形式出现，dataMat中的每一个元素的第一个值为1.0，这个是第一个回归系数的默认值，
        #这里默认数据只是二维数据
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


"""
sigmoid函数，根据参数计算sigmoid值
参数inX为向量，根据向量里面的每一个不同的元素计算sigmoid值，返回对应的sigmoid值向量
"""
def sigmoid(inX):
    print(len(inX))
    return 1.0/(1+exp(-inX))

"""
该函数根据梯度上升优化算法获得回归系数值，返回回归系数向量
"""
def gradAscent(dataMat, classLabels):
    dataMatrix = mat(dataMat)
    labelMatrix = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    #设置训练循环次数
    maxCycles = 500
    weights = ones((n, 1))
    #由于根据数学公式的推导，一下的计算方法由梯度上升方法提到而来，所以可以直接利用下面的方法来代替梯度上升方法
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMatrix - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


"""
随机梯度上升算法
"""
def stocGradAscent0(dataMatrix, classLabels):
    #由于下面的系数计算需要用到数组的运算，这里dataMatrix的类型为list， 要先转换成为numpy的ndArray类型
    dataMatrix = array(dataMatrix)
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


"""
改进的随机梯度上升算法.
"""
def stocGradAscent1(dataMatrix, classLabels, numIter):
    #由于下面的系数计算需要用到数组的运算，这里dataMatrix的类型为list， 要先转换成为numpy的ndArray类型
    dataMatrix = array(dataMatrix)
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            #alpha 每次迭代都要进行调整
            alpha  = 4/(1.0+j+i)+0.01
            #随机选取要更新的项
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


"""
画出数据集和Logistic回归最佳拟合直线的函数
"""
def plotBestFit(fileName):
    import matplotlib.pyplot as plt
    #加载数据
    dataMat, labelMat = loadDataSet(fileName)
    #训练数据获得相应的回归系数
    weights = array(stocGradAscent1(dataMat, labelMat, 500))
    print(weights)
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    #因为X为一维的数组，所以Y也应该是一维的数组，所以要先将gradAscent函数返回的矩阵转换成为数组，在进行数组计算
    #得到的结果就是一维的数组，如果没有进行数组转换，那么将会进行的是矩阵运算，得到的也不是一维的数组
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
 

"""
logistic回归分类函数
"""
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

"""
从疝气病症预测病马死亡率
"""
def colicTest(trainningFileName, testFileName):
    frTrain= open(trainningFileName)
    frTest = open(testFileName)
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = gradAscent(array(trainingSet), trainingLabels)
    print(trainWeights)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is : %f" % errorRate)
    return errorRate

def multiTest(trainingFileName, testFileName):
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest(trainingFileName, testFileName)
    print("after %d iterations the average error rate is: %f" %(numTests, errorSum/float(numTests)))

if __name__ == '__main__':
    #plotBestFit(r"testSet.txt")
    colicTest(r"Training.txt",
              r"Test.txt")
