import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# 导入多项式回归模型
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

poly_reg = Pipeline([
    ('poly',PolynomialFeatures(degree=2)),
    ('std_scaler',StandardScaler()),
    ('lin_reg',LinearRegression())
])

x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1,1)
y = 0.5 * x**2 + x + 2 + np.random.normal(0,1,100)

X = np.arange(1,11).reshape(-1,2)
poly = PolynomialFeatures(degree=2)
poly.fit(X)
X2 = poly.transform(X)

poly_reg.fit(X,y)
y_predict = poly_reg.predict(X)
plt.scatter(x,y)
plt.plot(np.sort(x),y_predict[np.argsort(x)],color='r')
plt.show()