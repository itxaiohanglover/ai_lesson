# 导入数据集
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
# one-hot编码转化
from keras.utils import to_categorical

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# 查看样本数据的类型：60000, 28, 28
print("X_train.shape:" + str(X_train.shape))
print("X_test.shape:" + str(X_test.shape))
print("Y_train.shape:" + str(Y_train.shape))
print("Y_test.shape:" + str(Y_test.shape))
# # 打印标签值
# print(Y_train[0])
# # 训练集的第一个样本数据，绘图模式：灰度图
# plt.imshow(X_train[0], cmap="gray")
# plt.show()

# 28 * 28 = 784 二维变一维
X_train = X_train.reshape(60000, 784) / 255.0 # 减少差距，加快梯度下降
X_test = X_test.reshape(10000, 784) / 255.0

Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

model = Sequential()

model.add(Dense(units=256, activation='relu', input_dim=784))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
# 使用多分类交叉熵代价函数
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.05), metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=100, batch_size=256)

loss, accuracy = model.evaluate(X_test, Y_test)
print("loss" + str(loss))
print("accuracy" + str(accuracy))
