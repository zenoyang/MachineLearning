# _*_ coding: utf-8 _*_

import kNN
from numpy import *
from os import listdir

#
# Author: yz
# Date: 2017-12-01
#

'''
利用分类器进行手写数字识别测试

识别结果：
    错误的数量为10
    错误率为0.010570824524312896
'''

def img2vector(filePath):
    returnVect = zeros((1, 1024))
    file = open(filePath)
    for i in range(32):
        line = file.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(line[j])
    return returnVect



def handwritingClassTest():
    trainingFilePath = "data/digits/trainingDigits/"
    testFilePath = "data/digits/testDigits/"
    hwLabels = []
    trainingFileList = listdir(trainingFilePath)
    m = len(trainingFileList)   # 1934
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]   # 0_10.txt
        fileName = fileNameStr.split(".")[0]    # 0_10
        classNum = int(fileName.split("_")[0])  # 0
        hwLabels.append(classNum)
        trainingMat[i, :] = img2vector(trainingFilePath + fileNameStr)
    testFileList = listdir(testFilePath)
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileName = fileNameStr.split(".")[0]
        classNum = int(fileName.split("_")[0])
        testVector = img2vector(testFilePath + fileNameStr)
        classifyRes = kNN.classify(testVector, trainingMat, hwLabels, 3)
        print("kNN分类器分类结果为：{}, 真实的数字为：{}".format(classifyRes, classNum))
        if (classifyRes != classNum): errorCount += 1
    print("错误的数量为%d" % errorCount)
    print("错误率为{}".format(str(errorCount / mTest)))


if __name__ == '__main__':
    handwritingClassTest()