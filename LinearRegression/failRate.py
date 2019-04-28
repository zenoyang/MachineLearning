import numpy as np
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
# 导入多项式回归模型
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def getEta(xArr):
    """
    返回参数 η：依塔 Eta
    :param xArr:
    :return:
    """
    eta = []
    for i, tn in enumerate(xArr):
        item = tn
        if i == 0:
            item += tn / len(xArr)
            eta.append(item)
        else:
            count = 0
            for j in range(i, -1, -1):
                if j < 1: continue
                if xArr[i] == xArr[j - 1]:
                    count += 1
                    continue
                else:
                    item += (tn - xArr[i - count - 1]) / ((count + 1) * len(xArr))
                    eta.append(item)
                    break
    return eta


def failRate(xArr):
    """
    故障率函数
    :param xArr:
    :return:
    """
    eta = getEta(xArr)
    t = np.array(xArr)
    xMat = t/eta
    x = xMat

    alpha, beta, lamda = 0.5384, 0.1967, 0.8499

    """概率密度函数"""
    # res = ((c*np.power(xMat,a-1))/(n*np.power(1-xMat,b+1)))*\
    #           ((b-a)*xMat+c)*np.exp(-1*((c*np.power(xMat, a))/np.power(1-xMat, b)))

    """故障率函数"""
    res = ((lamda*np.power(x, alpha-1))/(eta*np.power(1-x, beta+1)))*((beta-alpha)*x+alpha)
    return res

def distinct(xArr, yArr):
    X, Y = [], []
    for i in range(0, len(xArr)):
        x, y = xArr[i], yArr[i]
        for j in range(i+1, len(xArr)):
            if xArr[i] != xArr[j]:
                X.append(x)
                Y.append(yArr[j-1])
                break
            else:
                if j == len(xArr)-1:
                    X.append(x)
                    Y.append(yArr[j])
                break
    return X, Y

def quadraticFeaturizer(x, y):

    poly_reg = Pipeline([
        ('poly', PolynomialFeatures(degree=10)),
        ('std_scaler', StandardScaler()),
        ('lin_reg', LinearRegression())
    ])
    X = np.array(x)
    X = X.reshape(-1, 1)
    print(X)
    print(y)

    poly_reg.fit(X, y)
    y_predict = poly_reg.predict(X)
    plt.scatter(x, y)
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')
    plt.show()


if __name__ == '__main__':
    xArr = [0.1, 0.2, 1, 1, 1, 1, 1, 2, 3, 6, 7, 11, 12, 18, 18, 18, 18, 18, 21, 32, 36, 40, 45,
            46, 47, 50, 55, 60, 63, 63, 67, 67, 67, 67, 72, 75, 79, 82, 82, 83, 84, 84, 84, 85,
            85, 85, 85, 85, 86, 86]
    yArr = failRate(xArr)
    X, y = distinct(xArr, yArr)

    quadraticFeaturizer(X, y)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # t = np.arange(1, 51, 1)
    #
    # ax.scatter(X, Y)
    # # ax.plot(xArr, failRate(xArr))
    #
    # plt.show()
