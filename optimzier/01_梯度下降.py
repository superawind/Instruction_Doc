# # 1、梯度下降
# def f(x1, x2):
#     return x1 ** 2 + 2 * x2 ** 2 + x1

# def grad(x1, x2):
#     return 2 * x1 + 1, 4 * x2

# # 给定初始值
# x1, x2 = 0.3, 0.3
# # 设定学习率
# eta = 0.1
# # 开始迭代
# for step in range(20):
#     g1, g2 = grad(x1, x2)
#     x1 -= eta * g1
#     x2 -= eta * g2

#     print(x1, x2, f(x1, x2)) 

# 2、梯度下降案例
import numpy as np 
import matplotlib.pyplot as plt 

def model(x, w, b):
    """
        y = XW+B  # 预测值
    """
    return x @ w + b 

def grad(x, d, w, b):
    y = model(x, w, b)
    ymd = 2 * (y - d) / len(x)

    dw = x.T @ ymd
    db = np.sum(ymd, axis=0)
    return dw, db

x = np.random.normal(0, 1, [1000, 1])
d = x * 2 + 1 + np.random.normal(0, 0.3, [1000, 1]) # 真实值
w = np.random.normal(0, 1, [1, 1])
b = np.zeros([1])

eta = 0.1 # 学习率
# x = np.concatenate([x, x**2], axis=1)
for step in range(1000):
    gw, gb = grad(x, d, w, b)
    # 梯度下降
    w -= eta * gw
    b -= eta * gb
    print(step, w, b)

xplt = np.linspace(-3, 3, 1000).reshape(1000, 1)
# x = np.concatenate([xplt, xplt**2], axis=1)
yplt = model(x, w, b)
plt.scatter(x[:,0], d[:,0], c='black')
plt.plot(x[:,0], yplt[:,0], c='red')
plt.show()

