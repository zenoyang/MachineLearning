# _*_ coding: utf-8 _*_

import kNN
from numpy import *

#
# Author: yz
# Date: 2017-12-01
#

'''
约会对象分类
婚恋网站数据:datingTestSet2.txt
    每年的飞行里程
    玩游戏所花时间百分比
    每年吃几升冰激凌
    3 -> 喜欢  2 -> 一般  1 -> 不喜欢 
    
验证结果：
    前50%作为测试集，后50%作为训练集：
        错误的数量为33
        错误率为0.066
    
    前50%作为训练集，后50作为测试集：
        错误的数量为19
        错误率为0.038
'''

# 文件转换成矩阵
def file2matrix(filename):
    file = open(filename)
    numOfLines = len(file.readlines())  # 文件的行数
    returnMat = zeros((numOfLines, 3))    # 初始化要return的矩阵，numOfLines行，3列
    classLabelVector = []   # 初始化要return的标签向量
    index = 0
    file = open(filename)
    for line in file.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# 归一化
# 每列的range = 每列的最大值 - 每列的最小值
# 每个元素归一化后的值 = (原来的值 - 该列的最小值) / 该列的最大值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    #ranges = maxVals - minVals
    # normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]    # 行数
    normDataSet = (dataSet - tile(minVals, (m, 1))) / tile(maxVals, (m, 1))
    return normDataSet


# 用50%数据来做测试，统计分类结果错误率  总共1000行数据
def datingClassTest():
    ratio = 0.50    # 训练和测试的比例
    datingDataMat, datingLabels = file2matrix('data/datingTestSet2.txt')
    normMat = autoNorm(datingDataMat)
    m = normMat.shape[0]    # 行数
    numTestVecs = int(m * ratio)
    errorCount = 0
    for i in range(numTestVecs):
        classifyRes = kNN.classify(normMat[i, :], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
        print("kNN分类器分类结果为：{}, 真实的类别为：{}".format(classifyRes, datingLabels[i]) )
        if (classifyRes != datingLabels[i]): errorCount += 1
    print("错误的数量为%d" % errorCount)
    print("错误率为{}".format(str(errorCount/numTestVecs)))


# 图形化展现
def graphicalDisplay():
    import matplotlib.pyplot as plt
    datingDataMat, datingLabels = file2matrix('data/datingTestSet2.txt')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
    # ax.scatter(datingDataMat[:, 0], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    plt.show()


if __name__ == '__main__':
    # 约会对象分类效果测试
    datingClassTest()
    # 图形化展现
    graphicalDisplay()


