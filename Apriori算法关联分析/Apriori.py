
from numpy import *

"""
该函数将构建大小为1的所有候选项集的集合
"""
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    #frozenset表示被冰冻的集合，用户只能读取不能修改，这里必须使用frozenset而不能用set
    #因为后面必须要将这些集合作为字典键值使用，使用frozenset可以实现这一点
    return map(frozenset, C1)
"""
该函数根据数据集D来对Ck进行筛选，选出符合最小支持度的所有候选项
参数:
D:数据集
Ck：候选项集列表
minSupport:最下支持度
"""
def scanD(D, Ck, minSupport):
    #用于存储每一个候选项在数据集中所出现的次数
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] +=1
    numItems = float(len(D))
    #存储符合最小支持度的候选项
    retList = []
    #存储所有候选项的支持度
    supportData = {}
    #计算所有的候选项的支持度
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

"""
该函数用来生成候选项集合Ck
参数：
Lk：经过筛选后，符合最小支持度的候选项list
k：表示所生成的候选项中拥有的元素个数
"""
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1， lenLk):
            L1 = list(Lk[i])[:, k-2]
            L2 = list(Lk[j])[:, k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                #Lk[i] | Lk[j] 集合的并操作
                retList,append(Lk[i] | Lk[j])
    return retList



def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, support = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


def generateResults(L, supportData, minConf = 0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if(i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

"""
对规则进行评估
"""
def calcConf(freqSet, H, supportData, br1, minConf = 0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            print("%d -->")
            print(conseq)
            print("conf:")
            print(conf)
            br1.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

"""
生成候选规则集合
"""
def rulesFromConseq(freqSet, H, supportData, br1, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m+1)):
        Hmpl = aprioriGen(H, m+1)
        Hmpl = calcConf(freqSet, Hmpl, supportData, br1, minConf)
        if(len(Hmpl) > 1):
            rulesFromConseq(freqSet, Hmpl, supportData, br1, minConf)

