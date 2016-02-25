from numpy import *
import pdb
"""
该函数用来通过阀值比较对数据进行分类,注意这里的数据类别是1和-1，不可以是1和0，因为不同的弱分类器有不同的权重，如果类别为0，那么
分类器的权重乘上去后还是0，此时该弱分类器的作用将被消除
参数：
dimen:用于分类的特征下标
threshVal:特征的阀值，大于该阀值的数据为一类，小于的为另一类
threshIneq:比法制大还是比阀值小的信号
"""
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

"""
函数调用上面的stumpClassify函数，根据数据的所有特征，以及每一个特征所取的步长作为阀值，遍历stumpClassify函数所有的可能输入值
并找到数据集上最佳的单层决策树,从而得到一个弱分类器
参数：
D:训练数据每条记录对应的权重
"""
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    #设置每一个特征值的遍历步长
    numSteps = 10.0
    #用来存储得到的弱分类器
    bestStump = {}
    #存储弱分类器对数据进行分类的结果
    bestClassEst = mat(zeros((m, 1)))
    minError = float("inf")
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        #得到对应的步长
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                #降分类正确的记录的对应errArr值设置为0
                errArr[predictedVals == labelMat] = 0
                #分类错误的训练数据记录的加权求和
                #pdb.set_trace()
                weightederror = D.T * errArr
                if weightederror < minError:
                    minError = weightederror
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
        #pdb.set_trace()
        #print("the weight: %f" % weightederror)
        #print("the dime: %d" % bestStump['dim'])
        #print("the minError: %f" % minError)
    return bestStump, minError, bestClassEst

"""
基于单层决策树的AdaBoost训练过程
参数：
numIt：迭代的次数，相当于如分类器的个数，但是如果所有弱分类器的加权分类结果错误率为0的时候，就直接退出迭代
"""
def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    #存放弱分类器的数组
    weakClassArr = []
    m = shape(dataArr)[0]
    #训练数据中没一条记录所对应的权重向量，初始化为1/m
    D = mat(ones((m, 1))/m)
    #存放所有训练数据记录的类别估计累计值
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        #得到一个弱分类器
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        #根据错误率计算对应的alpha值
        alpha = float(0.5*log((1.0 - error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        #将弱分类器加入到弱分类器数组中
        weakClassArr.append(bestStump)
        #根据当前的错误率和训练数据记录对应的权重D，更新权重向量D，为下一个弱分类器的训练做准备
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D, exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum()/m
        print("the total error rate is : %f" % errorRate)
        if errorRate == 0.0:
            break

    return weakClassArr

"""
AdaBoost分类函数
参数：
classifierArr：弱分类器数组
""" 
def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    #存放所有弱分类器的分类加权结果
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        #得到每一个弱分类器的分类结果
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        #将分类结果乘以弱分类器对应的权重alpha
        aggClassEst += classifierArr[i]['alpha']*classEst

    return sign(aggClassEst)
    
def loadData(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        label = float(curLine[-1])
        #数据的类别标签只能为1和-1 ，如果为1和0的话，要将0的标签改为-1
        if(label == 0.0):
            labelMat.append(-1)
        else:
            labelMat.append(label)
    return dataMat, labelMat


if __name__ == '__main__':
    dataMat, labelMat = loadData(r"Training.txt")
    classifierArray = adaBoostTrainDS(dataMat, labelMat, 80)
    testData, testLabel = loadData(r"Test.txt")
    predict = adaClassify(testData, classifierArray)
    errArr = mat(ones((67, 1)))
    errArr = errArr[predict != mat(testLabel).T]
    #print("the error number of the testData is : %d" % errArr.sum())
    print("the error rate of the testData is : %f" % (errArr.sum()/67))


    
