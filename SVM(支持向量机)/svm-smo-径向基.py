from numpy import *


"""
加载数据集
"""
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
        
    return dataMat, labelMat
"""
用于简化版的SMO算法，随机选择alpha对
"""
def selectJrand(i, m):
    j = i;
    while (j == i):
        j = int(random.uniform(0, m))

    return j

"""
用于确定新的alpha的值，alphs必须在H和L之间,以满足约束条件
"""
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    elif aj < L:
        aj = L

    return aj

"""
SMO算法是将大优化问题分解成为许多小优化问题来求解的。这些小优化问题往往很容易求解，并且将他们进行顺序求解的结果与将他
们作为整体来求解的结果是完全一样的。在结果完全相同时，SMO算法的求解时间短很多。

SMO的目标是求出一系列的alpha和b，一旦求出来，就可以很容易地计算出权重向量，得到分割超平面。
SMO的工作原理：每一次循环中选择两个alpha进行优化处理，一旦找到一对合适的alpha，那么就增大其中的一个，同时减少另一个，
这里所谓的合适就是指两个alpha必须要符合一定的条件。条件之一就是，这两个alpha必须要在间隔边界之外，第二个条件就是这两
个alpha还没有进行过区间化处理或者不在边界上。
"""

"""
SMO的简化版本，Platt SMO算法中的外循环确定要优化的最佳alpha对。简化版的却跳过这一部分，首先在数据集上遍历每一个alpha，
然后在剩下的alpha集合中随机选择另一个alpha，从而构成alpha对。

参数：
dataMatIn是一个lis，
classLabels也是一个list
toler表示容错的范围
"""
def smoSimple(dataMatIn, classLabels, c, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    #变量itera用于存储在没有任何alpha改变的情况下遍历数据集的次数，当该变量达到输入值maxIter时，函数结束运行并退出
    itera = 0
    while(itera < maxIter):
        #用于存储改变的alpha对的数目
        alphaParisChanged = 0
        for i in range(m):
            #函数进行学习过程，根据当前的alphs得到所预测的类别，在于实际的类别进行比较，从而对alpha进行调优
            fxi = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fxi - float(labelMat[i])
            if((labelMat[i]*Ei < -toler) and (alphas[i] < c)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fxj = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fxj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if(labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(c, c+alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - c)
                    H = min(c, alphas[j] + alphas[i])
                if L == H:
                    print("L == H")
                    continue
                eta = 2.0*dataMatrix[i,:]*dataMatrix[j, :].T - dataMatrix[i, :]*dataMatrix[i, :].T - dataMatrix[j, :]*dataMatrix[j, :].T
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j] -= labelMat[j]*(Ei-Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if(abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough ")
                    continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]*dataMatrix[i, :].T - \
                     labelMat[j]*(alphas[j] - alphaJold)*dataMatrix[i, :]*dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]*dataMatrix[j, :].T - \
                     labelMat[j]*(alphas[j] - alphaJold)*dataMatrix[j, :]*dataMatrix[j, :].T

                if(0<alphas[i]) and (c>alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (c > alphas[j]):
                    b = b2
                else:
                    b = (b1+b2)/2
                alphaParisChanged += 1
                print("itera: %d, i: %d, paris changed %d " %(itera, i, alphaParisChanged))
        if(alphaParisChanged == 0):
            itera += 1
        else:
            itera == 0
        print("iteration number is %d" % itera)
    return b, alphas

"""
核函数转换
"""
def kernelTrans(X, A, kTup):
    m,n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == "lin":
        K = X*A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2))

    else:
        raise NameError("Houston We Have a Problem -- That Kernel is not recognized")
    return K

"""
核函数需要用到的变量类
"""
class optStruct:
    def __init__(self, dataMatIn, classLabels, c, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.c = c
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))
        #以下为增加的部分
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


"""
完整版的Platt SMO算法 从这里开始
"""
class optStruct1:
    def __init__(self, dataMatIn, classLabels, c, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.c = c
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))

def calcEk(oS,k):
    #在使用核函数的时候，这里将替换下面的fXk表达式
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    #fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X*oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if(len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k ==i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei-Ek)
            if(deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

""""
完整Platt SMO 算法中的优化例程
"""
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.c)) or ((oS.labelMat[i] *Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if(oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.c, oS.c + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.c)
            H = min(oS.c, oS.alphas[j] + oS.alphas[i])

        if L ==H:
            print("L == H")
            return 0
        #在使用核函数的时候，这里将替换下面的eta表达式
        eta = 2.0*oS.K[i, j] - oS.K[i, i] -oS.K[j, j]
        #eta = 2.0 * oS.X[i, :]*oS.X[j, :].T - oS.X[i, :]*oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            print("eta >= 0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if(abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS, i)
        #在使用核函数的时候，这里的两条表达式将替换下面的两条表达式
        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i, i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i, j] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j, j]
        #b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i, :]*oS.X[i, :].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i, :]*oS.X[j, :].T
        #b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.X[i, :]*oS.X[j, :].T - oS.labelMat[j]*(oS.alphas[j] - alphaJold)*oS.X[j, :]*oS.X[j, :].T
        if(0 < oS.alphas[i]) and (oS.c > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.c > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0


"""
完整版Platt SMO的外循环代码
"""
def smoP(dataMatIn, classLabels, c, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), c, toler, kTup)
    itera = 0
    entireSet = True
    alphaPairsChanged = 0
    while(itera < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            #遍历所有的数据
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
            print("fillSet, iter: %d, i:%d, pairs changed %d" %(itera, i, alphaPairsChanged))
            itera += 1
        else:
            #遍历非边界的alpha值
            nonBoundIs = nonzero((oS.alphas.A > 0)*(oS.alphas.A < c))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter:%d i:%d, pairs changed: %d" %(itera, i, alphaPairsChanged))
            itera += 1
        if entireSet:
            entireSet = False
        elif(alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % itera)
    return oS.b, oS.alphas

"""
完整版的Platt SMO算法到这里结束
"""

"""该函数用来根据alphas计算分类超平面所对应的W
"""
def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i], X[i, :].T)
    return w


"""
利用核函数进行分类的径向基测试函数
"""
def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet(r"testSetRBF.txt")
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A>0)[0]
    sVs = dataMat[svInd]
    labelSv = labelMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        #先利用核函数进行核转换
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T*multiply(labelSv, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the trainning error rate is : %f" % (float(errorCount)/m))
    dataArr, labelArr = loadDataSet(r"testSetRBF2.txt")
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m,n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T*multiply(labelSv, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    #这里  float(errorCount)/m  必须加括号，不然会默认从左到右运算 抛出str 不能 除以 int的异常
    print("the test error rate is : %f" % (float(errorCount)/m))

if __name__ == "__main__":
    #dataMat, labelMat= loadDataSet(r"testSet.txt")
    #b, alphas = smoP(dataMat, labelMat, 0.6, 0.001, 40)
    #w = calcWs(alphas, dataMat, labelMat)
    testRbf()
