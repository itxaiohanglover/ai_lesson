import numpy as np
import dataset
import plot_utils

m = 100
xs, ys = dataset.get_beans(m)
print(xs)
print(ys)

plot_utils.show_scatter(xs, ys)

w1 = 0.1
w2 = 0.2
b = 0.1

## [[a,b][c,d]]
## x1s[a,c]
## x2s[b,d]
## 逗号，区分的是维度，冒号：区分的是索引，省略号… 用来代替全索引长度
# 在所有的行上，把第0列切割下来形成一个新的数组
x1s = xs[:, 0]
x2s = xs[:, 1]

# 前端传播
def forward_propgation(x1s, x2s):
    z = w1 * x1s + w2 * x2s + b
    a = 1 / (1 + np.exp(-z))
    return a

plot_utils.show_scatter_surface(xs, ys, forward_propgation)

for _ in range(500):
    for i in range(m):
        x = xs[i] ## 豆豆特征
        y = ys[i] ## 豆豆是否有毒
        x1 = x[0]
        x2 = x[1]

        a = forward_propgation(x1, x2)

        e = (y - a) ** 2

        deda = -2 * (y - a)
        dadz = a * (1 - a)
        dzdw1 = x1
        dzdw2 = x2
        dzdb = 1

        dedw1 = deda * dadz * dzdw1
        dedw2 = deda * dadz * dzdw2
        dedb = deda * dadz * dzdb

        alpha = 0.01
        w1 = w1 - alpha * dedw1
        w2 = w2 - alpha * dedw2
        b = b - alpha * dedb

plot_utils.show_scatter_surface(xs, ys, forward_propgation)