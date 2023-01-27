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
y_pre = w * xs
plt.plot(xs, y_pre)
plt.show()

alpha = 0.01
for _ in range(100):
    # 抛物线代价函数
    # e = x0^2 * w^2 + (-2x0y0) * w + y0^2
    # 斜率k = 2aw + b（求导）
    k = 2 * np.sum(xs ** 2) * w + np.sum(-2 * xs* ys)
    k = k / m
    w = w - alpha * k
    y_pre = w * xs
    # 绘制动态
    plt.clf() ## 清空窗口
    plt.scatter(xs, ys)
    plt.xlim(0, 1)
    plt.ylim(0, 1.2)
    plt.plot(xs, y_pre)
    plt.pause(0.01) # 暂停0.01秒


