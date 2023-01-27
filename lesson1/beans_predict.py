import dataset
from matplotlib import pyplot as plt
# 拿到随机豆豆
xs, ys = dataset.get_beans(100)
# print(xs)
# print(ys)
# 配置图像
plt.title("Size-Toxicity Function", fontsize=12)
plt.xlabel("Bean Size")
plt.ylabel("Toxicity")
plt.scatter(xs, ys)

# y = 0.5 * w
w = 0.5
for m in range(100):
    for i in range(100):
        x = xs[i]
        y = ys[i]
        y_pre = w * x
        # 误差
        e = y - y_pre
        alpha = 0.05
        w = w + alpha * e * x

y_pre = w * xs
plt.plot(xs, y_pre)
plt.show()