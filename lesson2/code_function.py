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

# 绘制代价函数的图像（w和方差e的图像）
es = []
ws = np.arange(0, 3, 0.1)
for w in ws:
    y_pre = w * xs
    # 平方误差、求和、平均得到误差
    e = (1 / m) * np.sum((ys - y_pre) ** 2)
    es.append(e)

plt.title("Cost Function", fontsize=12)
plt.xlabel("w")
plt.ylabel("e")
plt.plot(ws, es)
plt.show()

# 用抛物线顶点坐标求解最低点的w
w_min = np.sum(xs * ys) / np.sum(xs ** 2)
print("最小点的w：" + str(w_min))

# 把最低点的W值带回预测函数中，看看效果
y_pre = w_min * xs
plt.title("Size-Toxicity Function", fontsize=12)
plt.xlabel("Bean Size")
plt.ylabel("Toxicity")
plt.scatter(xs, ys)
plt.plot(xs, y_pre)
plt.show()