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

for _ in range(100):
    for i in range(100):
        x = xs[i]
        y = ys[i]
        # 抛物线代价函数
        # e = x0^2 * w^2 + (-2x0y0) * w + y0^2
        # 斜率k = 2aw + b（求导）
        k = 2 * (x**2) * w + (-2 * x * y)
        # alpha为学习率
        alpha = 0.1
        w = w - alpha * k
        # 绘制动态
        plt.clf() ## 清空窗口
        plt.scatter(xs, ys)
        y_pre = w * xs
        plt.xlim(0, 1)
        plt.ylim(0, 1.2)
        plt.plot(xs, y_pre)
        plt.pause(0.01) # 暂停0.01秒

# 重新绘制散点图和预测曲线
# plt.scatter(xs, ys)
# y_pre = w * xs
# plt.plot(xs, y_pre)
# plt.show()

