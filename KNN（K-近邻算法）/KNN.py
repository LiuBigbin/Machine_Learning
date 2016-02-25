from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
import os
"""
改函数用于创建训练集group为训练样本的数据，labels为每一个group训练样本所对应的分类
"""
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


"""
该函数将文本数据转换为numpy对应的数据集和标签向量，这里默认文本的每一行数据为:特征1\t特征2\t特征3\t标签值
返回值：
    returnMat：数据特征值矩阵
    classLabelVector：returnMat矩阵中每一行特征值所对应的标签
"""
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    #这里默认数据集里面只有三个特征
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

"""
该函数用于将32*32的二进制图像矩阵转换为1*1024的向量
"""
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in  range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0][i*32+j] = int(lineStr[j])
    return returnVect

"""
数据的归一化,在该函数中，所有的运算都是针对于矩阵中的每一个值来进行的，并不是整一个矩阵的运算
"""
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normdataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


"""
针对于约会网站数据的分类测试函数
"""
def dataClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix("TestSet.txt")
    normMat, ranges, minvals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, : ], normMat[numTestVecs:m, : ], datingLabels[numTestVecs:m], 9)
        print("the classifier came back with : %s, the real answer is : %s" % (classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount+=1.0

    print("the total error rate is: %f" % (errorCount/numTestVecs))

"""
该函数用于手写数字识别系统
"""
def handwritingClassTest(trainingdirname, testdirname):
    hwLabels = []
    print(trainingdirname)
    trainingFileList = os.listdir(trainingdirname)
    #获得训练文件夹里面的文件的数量
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, : ] = img2vector(trainingdirname+'\\'+fileNameStr)
    testFileList = os.listdir(testdirname)
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = fileStr.split('_')[0]
        vectorUnderTest = img2vector(testdirname+'\\'+fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 9)
        #print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, int(classNumStr)))
        if int(classNumStr) != classifierResult:
            errorCount += 1
    print("the total number of error is : %d" % errorCount)
    print("the total error rate is: %f" % (errorCount/mTest))
        

"""
该函数的功能是使用K-近邻算法将每一组数据划分到某个类中。
参数inX表示用于分类的输入向量， dadaSet表示输入的训练样本集，labels表示标签向量，
K表示用于选择最近邻居的数目
"""
def classify0(inx, dataSet, labels, k):
    #获得numpy数组的维度，shape返回数组的行数n， 列数m元组（n, m）
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inx, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):        
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]


        
if __name__ == '__main__':
    
    returnMat, labels = file2matrix("datingTestSet2.txt")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.scatter(returnMat[:, 0], returnMat[:, 2])
    ax.scatter(returnMat[:, 1], returnMat[:, 2], 15.0*array(labels), 15.0*array(labels))
    plt.show()
    """
    handwritingClassTest(r"trainingDigits", r"testDigits")

    """
