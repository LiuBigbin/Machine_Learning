#-*-coding:utf-8-*-
#用于绘制决策树
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle = "sawtooth", fc = "0.8")
leafNode = dict(boxstyle="round4", fc = "0.8")
arrow_args = dict(arrowstyle = "<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.axl.annotate(nodeTxt, xy = parentPt, xycoords = 'axes fraction',
                            xytext = centerPt, textcoords = "axes fraction",
                            va = "center", ha = "center", bbox = nodeType,
                            arrowprops = arrow_args)
"""
def createPlot():
    fig = plt.figure(1, facecolor = "white") #创建一个新图形
    fig.clf() #清空绘图区
    createPlot.axl = plt.subplot(111, frameon = False)
    plotNode('decision_making node',(0.5,0.1),(0.1,0.5),decisionNode) 
    plotNode('leafNode',(0.8,0.1),(0.3,0.8),leafNode) 
    plt.show()
"""

"""
获得叶节点的数目
"""
def getNumLeafs(myTree):
    numLeafs = 0
    #python 3.X改变了dict.keys,返回的是dict_keys对象,支持iterable 但不支持indexable
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

"""
获得树的层数
"""
def getTreeDepth(myTree):
    maxDepth = 0
    #python 3.X改变了dict.keys,返回的是dict_keys对象,支持iterable 但不支持indexable
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else :
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

"""
该函数在父子节点中填充文本信息
"""
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.axl.text(xMid, yMid, txtString)

"""
该函数用来画树
"""
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    #cntrPt用于存放绘制点的位置
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt,decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else :
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))

    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor = 'white')
    fig.clf()
    axprops = dict(xticks = [], yticks = [])
    createPlot.axl = plt.subplot(111, frameon = False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))  #存储树的宽度
    plotTree.totalD = float(getTreeDepth(inTree))  #存储树的深度
    #plotTree.xOff和plotTree.yOff用于追踪已经绘制的节点的位置以及放置下一个节点的位置
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
            
#createPlot()
