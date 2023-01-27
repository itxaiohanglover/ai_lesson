import dataset
import matplotlib.pyplot as plt
import numpy as np

# 豆豆数量m
m = 100
xs, ys = dataset.get_beans(m)

# 配置图像
plt.title("Size-Toxicity Function", fontsize=12)
plt.xlabel("Bean Size")
plt.ylabel("Toxicity")
plt.scatter(xs, ys)

w = 0.1
b = 0.1
z = w * xs + b
a = 1 / (1 + np.exp(-z)) # 加入激活函数
plt.plot(xs, a)
plt.show()

# alpha为学习率
alpha = 0.01
# 训练5000次
for _ in range(5000):
    for i in range(100):
        x = xs[i]
        y = ys[i]
        # 三个函数
        z = w * x + b
        a = 1 / (1 + np.exp(-z))
        e = (y - a) ** 2
        # 对w和b求偏导
        deda = -2 * (y - a)
        dadz = a * (1 - a)
        dzdw = x
        dzdb = 1

        dedw = deda * dadz * dzdw
        dedb = deda * dadz * dzdb

        w = w - alpha * dedw
        b = b - alpha * dedb

    if _ % 100 == 0:
        # 绘制动态
        plt.clf()  ## 清空窗口
        plt.scatter(xs, ys)
        z = w * xs + b
        a = 1 / (1 + np.exp(-z))  # 加入激活函数
        plt.xlim(0, 1)
        plt.ylim(0, 1.2)
        plt.plot(xs, a)
        plt.pause(0.01)  # 暂停0.01秒

