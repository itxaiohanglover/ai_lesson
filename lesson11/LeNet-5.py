# 导入数据集
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
# One-Hot编码转化
from keras.utils import to_categorical
# 2D卷积层
from keras.layers import Conv2D
# 二维平均池化层
from keras.layers import AveragePooling2D
# 数组平铺
from keras.layers import Flatten

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# 减少差距，加快梯度下降，归一化操作
X_train = X_train.reshape(60000, 28, 28, 1) / 255.0
X_test = X_test.reshape(10000, 28, 28, 1) / 255.0

Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

model = Sequential()
# 卷积层部分
model.add(
    Conv2D(
        filters=6,  # 卷积核/过滤器数量
        kernel_size=(5, 5),  # 卷积核尺寸
        strides=(1, 1),  # 步长
        input_shape=(28, 28, 1),  # 输入形状
        padding='valid',  # 填充模式（越卷越小）
        activation='relu'  # 激活函数
    )
)
# 池化窗口大小为 2*2
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(
    Conv2D(
        filters=16,  # 卷积核/过滤器数量
        kernel_size=(5, 5),  # 卷积核尺寸
        strides=(1, 1),  # 步长
        padding='valid',  # 填充模式（越卷越小）
        activation='relu'  # 激活函数
    )
)
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Flatten())
# 全连接层部分
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# 送入训练
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.05), metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=100, batch_size=256)

# 评估测试集
loss, accuracy = model.evaluate(X_test, Y_test)
print("loss" + str(loss))
print("accuracy" + str(accuracy))