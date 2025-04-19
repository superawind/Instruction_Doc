# 2、随机梯度下降，随机的选择一些样本参与一次更新，随机在样本选择上
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
d = x ** 2 - 1 * x + 1 + np.random.normal(0, 0.3, [1000, 1]) # 真实值
w = np.random.normal(0, 1, [2, 1])
b = np.zeros([1])
eta = 0.1 # 学习率
x = np.concatenate([x, x**2], axis=1)

# 随机筛选一个batch大小的数据，实现随机梯度下降
batch_szie = 32

for step in range(50):
    idx = np.random.randint(0, 1000, batch_szie)
    xin = x[idx]; din = d[idx]
    # 通过随机选取的数据进行梯度下降计算，就是随机梯度下降
    gw, gb = grad(xin, din, w, b)
    w -= eta * gw
    b -= eta * gb
    # print(step, w, b)

xplt = np.linspace(-3, 3, 1000).reshape(1000, 1)
xplt = np.concatenate([x, x**2], axis=1)
yplt = model(x, w, b)
plt.scatter(x[:,0], d[:,0], c='black')
plt.scatter(xplt[:,0], yplt[:,0])
plt.show()

