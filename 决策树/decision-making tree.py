#决策树
from math import log
#该模块用于序列化对象，方便决策树的存储
import pickle

"""
计算给定数据集的香农熵
"""
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts.keys():
        prob = float(labelCounts[key]/numEntries)
        shannonEnt -= prob*log(prob, 2)
    return shannonEnt


"""
按照给定的特征划分数据集
"""
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

"""
选择最好的数据集划分方式
"""
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    #计算数据集的基本信息熵
    baseEntropy = calcShannonEnt(dataSet)
    #最大的信息增益
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        #根据特诊i的值，对数据集进行划分，并计算划分后的每一个子数据集的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        #根据当前的特征分割后所得到的的信息增益
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

"""
改函数返回出现次数最多的分类名称
"""
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

"""
创建决策树的函数代码
"""
def createTree(dataSet, labels):
    #获得dataSet中的所有记录的类别,如果dataSet中的数据集都是相同类别的，那么直接返回对应的类别，停止函数的递归
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #如果数据集里面的特征已经只剩下一个了，那么直接利用统计方法，返回所剩下的记录中类别出现最多的一个类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    #获得当前数据集中最好的分类的特征
    bestFeat= chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    #在这里，以字典的方式存储决策树的数据
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    #根据最好的特征下标，获得该特征所具有的所有值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        #将数据集根据当前最好的特征的具体值进行分解，得到相应的不同子数据集，然后利用递归的方法，将子数据集进行建树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree

"""
使用决策树的分类函数
"""
def classify(inputTree, featLabels, testVect):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index[firstStr]
    for key in secondDict.keys():
        if testVect[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classify(secondDict[key], featLabels, testVect)
            else:
                return secondDict[key]
    print("error: the testVect feat:%s, has the value: %s, but there is no the value in trainingDataSet" % (str(firstStr), str(testVect[featIndex])))      
    

"""
使用pickle模块存储决策树
"""
def storeTree(inputTree, fileName):
    #以二进制的形式存储
    fw = open(fileName, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

"""
使用pickle模块读取决策树
"""
def grabTree(fileName):
    #以二进制的形式读取
    fr = open(fileName, 'rb')
    return pickle.load(fr)

if __name__ == '__main__':
    fr = open(r"lenses.txt")
    dataSet = [line.strip().split('\t') for line in fr.readlines()]
    classLables = ['age','prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(dataSet, classLables)
    print(lensesTree)
    import plotTree as pt
    pt.createPlot(lensesTree)
    #storeTree(lensesTree, r"lensesTree.txt")
    #print(grabTree(r"lensesTree.txt"))
