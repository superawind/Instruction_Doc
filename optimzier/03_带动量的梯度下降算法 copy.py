# 2、带动量的梯度下降算法,迭代过程中，对梯度进行滑动平均
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

# 在随机梯度下降基础上，升级成为带动量的随机梯度下降
beta1 = 0.1
vwt = 0; vbt = 0
for step in range(50):
    idx = np.random.randint(0, 1000, batch_szie)
    xin = x[idx]; din = d[idx]
    # 通过随机选取的数据进行梯度下降计算，就是随机梯度下降
    gw, gb = grad(xin, din, w, b)
    vwt = beta1 * vwt + (1-beta1) * gw
    vbt = beta1 * vbt + (1-beta1) * gb
    # 初始化防止v过小，进行一些标准化处理
    vwthat = vwt / (1-beta1**(step+1))
    vbthat = vbt / (1-beta1**(step+1))
    w -= eta * vwthat
    b -= eta * vbthat
    # print(step, w, b)

xplt = np.linspace(-3, 3, 1000).reshape(1000, 1)
xplt = np.concatenate([x, x**2], axis=1)
yplt = model(x, w, b)
plt.scatter(x[:,0], d[:,0], c='black')
plt.scatter(xplt[:,0], yplt[:,0])
plt.show()

