import shopping_data
# 数据对齐
from keras.utils import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import Flatten

x_train, y_train, x_test, y_test = shopping_data.load_data()
# 打印数据集
# print('x_train.shape：', x_train.shape)
# print('y_train.shape：', y_train.shape)
# print('x_test.shape：', x_test.shape)
# print('y_test.shape：', y_test.shape)
# print(x_train[0])
# print(y_train[0])

vocalen, word_index = shopping_data.createWordIndex(x_train, x_test)
# print(word_index)
# print('词典总词数：', vocalen)

# 转化为索引向量
x_train_index = shopping_data.word2Index(x_train, word_index)
x_test_index = shopping_data.word2Index(x_test, word_index)
# 每一句话的索引向量个数不一样，我们需要把序列按照maxlen对齐
maxlen = 25
x_train_index = pad_sequences(x_train_index, maxlen=maxlen)
x_test_index = pad_sequences(x_test_index, maxlen=maxlen)

# 神经网络模型
model = Sequential()
model.add(
    Embedding(
        trainable=True, # 是否可训练：是否让这一层在训练的时候更新参数
        input_dim=vocalen, # 输入维度
        output_dim=300, # 输出维度
        input_length=maxlen # 序列长度
    )
)
model.add(Flatten()) # 数据平铺
# 三个隐藏层
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
# 二分类问题，使用sigmoid激活函数
model.add(Dense(1, activation='sigmoid'))
model.compile(
    loss='binary_crossentropy', # 适用于二分类问题的交叉熵代价函数
    optimizer='adam', # adam是一种使用动量的自适应优化器，比普通的sgd优化器更快
    metrics=['accuracy']
)
# 训练
model.fit(x_train_index, y_train, batch_size=512, epochs=200)
score, acc = model.evaluate(x_test_index, y_test)
# 评估
print('Test score:', score)
print('Test accuracy:', acc)


