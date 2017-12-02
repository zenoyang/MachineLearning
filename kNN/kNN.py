# _*_ coding: utf-8 _*_

from numpy import *
import operator

#
# Author: yz
# Date: 2017-12-01
#

'''
kNN:k近邻
Input:      inX: 待分类向量 (1xN)
            dataSet: 先验数据集 (NxM)
            labels: 先验数据分类标签 (1xM vector)
            k: 参数：k个近邻 (should be an odd number)
Output:     分类标签
'''

# KNN分类器
def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # shape[0]表示行数
    ## step 1: 计算距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet # 按元素求差值 tile
    sqDiffMat = diffMat**2  # 将差值平方
    sqDistences = sqDiffMat.sum(axis=1) # 按行累加
    distences = sqDistences**0.5    #将差值平方和求开方，即得距离
    ## step 2: 对距离排序
    sortedDistIndicies = distences.argsort()  # 排序后的索引   argsort() 返回排序后的索引值
    ## step 3: 选择k个最近邻
    classCount = {}
    for i in range(k):
        voteLable = labels[sortedDistIndicies[i]]
        ## step 4: 计算k个最近邻中各类别出现的次数
        classCount[voteLable] = classCount.get(voteLable, 0) + 1
    ## step 5: 返回出现次数最多的类别标签
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    # 创建一个数据集，包含2个类别共4个样本
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    # kNN分类器测试
    res = classify([3,3], group, labels, 3)
    print(res)