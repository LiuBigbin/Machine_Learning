from numpy import *



"""
创建所有词的一个不重复的列表
"""
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)#集合的并运算，将每一篇文档的词语先以set去重，再与vocabSet合并
    return list(vocabSet)

"""
词集模型
根据所创建的词列表，将所输入的数据以此列表的顺序转换为0,1向量
"""
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word %s is not in my Vocabulary." % word)

    return returnVec

"""
词袋模型
根据所创建的词列表，将所输入的数据以此列表的顺序转换为词所出现次数向量
"""
def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word %s is not in my Vocabulary." % word)
   
    return returnVec

"""
朴素贝叶斯分类器训练函数,该函数目前默认文档只有两类，所以：
trainMatrix参数以0， 1向量形式出现，表示每一篇文档对应于Vocabulary的向量
trainCategory：只有1和0两种类型，用来表示每一篇文档所对应的类别
"""
def trainNB(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    #类别为1的文档占所有训练文档的概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    """
    默认的，将所有的词出现的次数先初始化为0，如果词在类中没有出现过，那么其随后的概率为0，如果待分类的文本有该类中没有
    出现的词，那么=该词所得到的的概率为0，将会使后面所有的计算都为0，所以应该将其初始化为1，这也是伯努利模型的特征，只在乎
    单词出现与不出现，不在乎其在一个文档中所出现的次数，也就是说所有词的权重是一样的
    p0Num = zeros(numWords) #类0中每一个词出现的次数向量
    p1Num = zeros(numWords) #类1中每一个词出现的次数向量
    #默认为0，在伯努利模型中，将改为2.0，代表词的出现与不出现两种情况
    p0Denom = 0.0 #类0中所有的文档出现的词数目（相同文档中出现相同的词按一次计算，不同文档出现相同的词则按照出现的次数算）
    p1Denom = 0.0 #类1中所有的文档出现的词数目（相同文档中出现相同的词按一次计算，不同文档出现相同的词则按照出现的次数算）
    """
    p0Num = ones(numWords) #类0中每一个词出现的次数向量
    p1Num = ones(numWords) #类1中每一个词出现的次数向量
    p0Denom = 2.0 #类0中所有的文档出现的词数目（相同文档中出现相同的词按一次计算，不同文档出现相同的词则按照出现的次数算）
    p1Denom = 2.0 #类1中所有的文档出现的词数目（相同文档中出现相同的词按一次计算，不同文档出现相同的词则按照出现的次数算）
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num +=trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    """
    类1中每一个词出现的概率,此处将概率对数化，是小的概率也会变成较大的数，避免一大堆小概率相乘
    造成的下溢出问题
    """
    p1Vect = log(p1Num/p1Denom) #类1中每一个词出现的概率
    p0Vect = log(p0Num/p0Denom) #类0中每一个词出现的概率
    return p0Vect, p1Vect, pAbusive

"""
朴素贝叶斯分类函数
参数：
vec2Classify：待分类的文本0,1化的词向量
p0Vec：所有单词在类别0中出现的概率向量
p1Vec：所有单词在类别1中出现的概率向量
pClass1：类别1的在训练数据集中的概率
"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify*p1Vec) + log(pClass1)
    p0 = sum(vec2Classify*p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

    
"""
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB(array(trainMat), array(listClasses))
"""



"""
解析邮件
返回长度大于2的单词
"""
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower for tok in listOfTokens if len(tok) > 2]


"""
分类垃圾邮件
"""
def spamTest(spamDirName, hamDirName):
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        #解析垃圾邮件
        wordList = textParse(open(spamDirName + '\\%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        #解析非垃圾邮件
        wordList = textParse(open(hamDirName + '\\%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    #存留交叉验证
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
         trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
         trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print("the error rate is: %f" % float(errorCount/len(testSet)))


if __name__ == "__main__":
    spamTest(r"spam",
             r"ham")
