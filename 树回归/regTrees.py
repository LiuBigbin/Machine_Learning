
from numpy import *

"""
该函数用来下载数据，与以往的下载数据不同的是，该函数下载后的数据是特征向量和标签值是合成为一个向量
"""
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        #将每行映射成浮点数
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat

"""
该函数用来根据特征以及特征值来切分集合
参数;
dataSet:待切分的数据集
feature：切分的特征
value：根据该值将dataSet切分为两个部分
"""
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]
    return mat0, mat1
"""
该函数是一个树构建函数，采用递归的方法，先找到目前的数据集中最好的切分的特征以及该特征的值
然后将根据所得到的特征和值将数据集切分，分别获得左右子数据集，在递归的调用函数，返回对应的子树，
作为目前的节点的左右子树
参数:
dataSet:待创建的数据集
leafType：给出建立叶节点的函数
errType：代表误差计算函数
ops:包含树构建所需要的其他参数的元组
"""
def crateTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

"""
该函数用来生成叶节点， 当chooseBestSplit函数确定不再对数据进行切分时，将调用该函数来得
到叶子节点的模型
"""
def regLeaf(dataSet):
    return mean(dataSet[:, -1])

"""
误差估计函数，该函数在给定的数据上计算目标变量的平方误差
因为这里需要返回的是总方差，所以要用均方差乘以数据集中样本的个数
"""
def regErr(dataSet):
    #var为均方差函数
    return var(dataset[:, -1])*shape(dataSet)[0]


"""
该函数用于找到最佳二元切分方式
"""
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr,ops=(1,4)):
    #容许的误差阀值
    tolS = ops[0]
    #切分的最少样本数
    tolN = ops[1]
    #当数据集只有一个元素的时候，直接生成叶节点，放返回
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    #切分前的误差
    S = errType(dataSet)
    #最好的切分后，所得到的误差
    bestS = inf
    bestIndex = 0
    bestValue = 0
    #遍历所有的特征
    for featIndex in range(n-1):
        #遍历某个特征的所有可能取值，以便获得该特征的最好的切分值
        for splitVal in set(dataSet[:, featIndex]):
            mat0,mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if(shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #如果前后切分的误差改变不大，那么没有必要在切分，直接的返回叶子节点
    if (S - bestS) < tols:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    #如果切分后的数据集的大小小于规定大小，那么也没有必要进行切分，直接的返回叶子节点
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


def isTree(obj):
    return (type(obj).__name__ == 'dict')

"""
该函数用来计算一棵树的平均值，左右子树的平均值的和除以2
该函数计算后，tree将不再是原来的树，根节点的左节点是原来的左子树的均值，右节点也一样，不再是指向一棵子树
"""
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right'])/2.0

"""
后剪枝函数。后剪枝方法需要将数据集分成测试数据和训练数据，对建立后的树，从上而下找到叶子节点，
用测试数据集来判断将这些叶节点合并是否能降低测试误差，如果是的话，就合并。
后剪枝可能不如预剪枝有效，一般的，为了寻求最佳的模型，可以同时的使用两种剪枝技术
参数：
tree：待剪枝的树
testData：剪枝所需要的测试数据
"""
def prune(tree, testData):
    #如果没有测试数据直接返回树的均值
    if shape(testData)[0] == 0:
        return getMean(tree)
    #只要有一方为子树，那么不能进行合并，必须继续剪枝
    if (isTree(tree['right'])) or (isTree(tree['left'])):
        #将测试数据根据当前的树的划分规则，分为左右子数据集
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    #左子树剪枝
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    #右子树剪枝
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    #必须左右节点都是叶节点，才可以跟并在一起
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        #没有合并时的误差
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + sum(power(rSet[:, -1] - tree['right'], 2))
        #获得树的均值，此时tree已经不再是原来的树了
        treeMean = (tree['left'] + tree['right'])/2.0
        #合并后的误差
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        #如果合并后，误差变小了，那么合并
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        #不能合并时，直接返回树
        else:
            return tree
    else:
        return tree

"""
对数据建立线性模型
"""
def linearSolve(dataSet):
    m,n = shape(dataSet)
    X = mat(ones((m,n)))
    Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]
    Y = dataSet[:,-1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is aingular, cannot do inverse, try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y
"""
数据集不需要切割时，将叶节点建模，返回线性模型的系数
"""
def modelLeaf(dataSet):
    ws,X, Y  =linearSolve(dataSet)
    return ws
"""
计算模型的误差
"""
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))

"""
该函数用来对回归树叶子节点进行预测
"""
def regTreeEval(model, inDat):
    return float(model)
"""
改函数用来对模型树叶子节点进行预测
"""
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n+1)))
    X[:, 1:n+1] = inDat
    return float(X*model)
"""
对于输入的测试数据inData，该函数返回对应的预测值
参数：
modelEval：当到达模型的叶子节点时，可以根据该参数来对数据进行相应的预测——回归树预测还是模型树预测
tree:已训练好的模型
"""
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)
"""
该函数对testData进行预测，每一次获得testData的一行数据，然后调用treeForeCast进行预测，最终返回所有测试数据的预测值
"""
def createForeCast(tree, testData, modelEval = regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat
            
