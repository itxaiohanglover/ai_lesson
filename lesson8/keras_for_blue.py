import dataset
import plot_utils
from keras.models import Sequential
from keras.layers import Dense

m = 100
X, Y = dataset.get_beans1(m)
plot_utils.show_scatter(X, Y)

model = Sequential()
# 当前层神经元的数量为1，激活函数类型：sigmoid，输入数据特征维度：1
model.add(Dense(units=1, activation='sigmoid', input_dim=1))
# loss（损失函数、代价函数）：mean_squared_error均方误差；
# optimizer（优化器）：sgd（随机梯度下降算法）；
# metrics（评估标准）：accuracy（准确度）；
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# epochs：回合数（全部样本完成一次训练）、batch_size：批数量（一次训练使用多少个样本）
model.fit(X, Y, epochs=5000, batch_size=10)

pres = model.predict(X)

plot_utils.show_scatter_curve(X, Y, pres)