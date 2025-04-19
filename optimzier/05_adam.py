# 2、adam是一个缝合怪，讲带动来给你的随机梯度下降和 均方根传递相结合即是
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
eta = 1e-3 # 学习率
x = np.concatenate([x, x**2], axis=1)

# 随机筛选一个batch大小的数据，实现随机梯度下降
batch_szie = 32

# 带动量的梯度下降
vwt = 0; vbt = 0

# 修改为均方根传递
beta1 = 0.9; beta2 = 0.999
uwt = 0; ubt = 0

# 修改为
for step in range(50):
    idx = np.random.randint(0, 1000, batch_szie)
    xin = x[idx]; din = d[idx]
    # 通过随机选取的数据进行梯度下降计算，就是随机梯度下降
    gw, gb = grad(xin, din, w, b)

    # 带动量的部分
    vwt = beta1 * vwt + (1-beta1) * gw
    vbt = beta1 * vbt + (1-beta1) * gb
    # 初始化防止v过小，进行一些标准化处理
    vwthat = vwt / (1-beta1**(step+1))
    vbthat = vbt / (1-beta1**(step+1))

    # 均方根部分
    uwt = beta1 * uwt + (1-beta2) * gw ** 2
    ubt = beta1 * ubt + (1-beta2) * gb ** 2
    # 初始化防止v过小，进行一些标准化处理
    uwthat = uwt / (1-beta2**(step+1))
    ubthat = ubt / (1-beta2**(step+1))

    w -= eta * vwthat / (np.sqrt(uwthat) + 1e-4)
    b -= eta * vbthat / (np.sqrt(ubthat) + 1e-4)
    # print(step, w, b)

xplt = np.linspace(-3, 3, 1000).reshape(1000, 1)
xplt = np.concatenate([x, x**2], axis=1)
yplt = model(x, w, b)
plt.scatter(x[:,0], d[:,0], c='black')
plt.scatter(xplt[:,0], yplt[:,0])
plt.show()

