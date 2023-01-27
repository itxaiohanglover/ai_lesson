import dataset
import matplotlib.pyplot as plt

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
y_pre = w * xs + b
plt.plot(xs, y_pre)
plt.show()

# alpha为学习率
alpha = 0.01
# 训练500次
for _ in range(500):
    for i in range(100):
        x = xs[i]
        y = ys[i]
        # 抛物线代价函数
        # 斜率k（求导）
        dw = 2 * (x ** 2) * w + 2 * x * b - 2 * x * y
        db = 2 * b + 2 * x * w - 2 * y

        w = w - alpha * dw
        b = b - alpha * db
    # 训练一次后刷新
    # 绘制动态
    plt.clf()  ## 清空窗口
    plt.scatter(xs, ys)
    y_pre = w * xs + b
    plt.xlim(0, 1)
    plt.ylim(0, 1.2)
    plt.plot(xs, y_pre)
    plt.pause(0.01)  # 暂停0.01秒

