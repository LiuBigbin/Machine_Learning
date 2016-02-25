
from numpy import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat
"""
计算两个向量的欧式距离
"""
def distEclud(vecA, VecB):
    return sqrt(sum(power(vecA-vecB, 2)))

"""
随机的生成k个质心
"""
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    #k个质心
    centroids = mat(zeros((k, n)))
    #为质心的每一个特征赋值
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k,1)
    return centroids

"""
K-均值聚类算法
"""
def kMeans(dataSet, k, distMeas = distEclud, createCent = randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    #如果簇没有改变，那么退出循环
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        #计算每一条数据与K个中心点的距离
        for i in range(m):
            #记录最小的距离
            minDist = inf
            #记录最小的距离所对应的中心点下标
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            #记录每一条数据应该所属的簇的下标，并存储到该簇中心点的距离
            clusterAssment[i, :] = minIndex, minDist**2
        print("centroids:")
        print(centroids)
        #跟新所有簇的质心的位置
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
            #axis = 0 表示沿矩阵的列方向进行均值计算
            centroids[cent,:] = mean(ptsInClust, axis = 0)

    return centroids, clusterAssment
    
"""
二分K-均值算法,该算法是为了克服K-均值算法收敛于局部最小值的问题而提出的。
该算法首先将所有的点作为一个簇，然后将该簇一分为二，之后选择其中的一个簇进行二分，至于选择哪一个
簇进行划分，取决于对其划分是否可以最大程度降低SSE的值，上述基于SSE的划分过程不断的重复，直到得
到用户指定的簇数目为止
"""
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    #用来存储数据集中每一条数据所属的簇，以及与该簇的质心的距离 
    clusterAssment = mat(zeros((m,2)))
    #将所有的数据节点看成一个簇，得到该簇的质心
    centroid0 = mean(dataSet, axis = 0).tolist()[0]
    centList = [centroid0]
    #计算每一个节点到质心的距离
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
        #循环，直到获得k个质心
    while(len(centList) < k):
        lowestSSE = inf
        #遍历所有的簇，找出最适合划分的簇
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzeros(clusterAssment[:,0].A == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            #划分的簇划分后的误差
            sseSplit = sum(splitClustAss[:,1])
            #划分前的误差
            sseNotSplit = sum(clusterAssment[nonzeros(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit, and notSplit: %f,  %f" % (sseSplit, sseNotSplit))
            if(sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[: ,0].A == 1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0],0] = bestCentToSplit
        print("the bestCentToSplit is : %d" % bestCentToSplit)
        print("the len of bestClustAss is: %d" % len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:]
        centList.append(bestNewCents[1,:])
        clusterAssment[nonzero(clustAssment[:,0].A == bestCentToSplit)[0], :] = bestClustAss
    return mat(centList), clusterAssment

                
        
        
        
