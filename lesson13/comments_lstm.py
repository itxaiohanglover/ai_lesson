import shopping_data
# 数据对齐
from keras.utils import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, Embedding
# 导入LSTM
from keras.layers import LSTM
# 读取中文词向量工具
import chinese_vec
import numpy as np

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

# 自行构造词嵌入矩阵
word_vecs = chinese_vec.load_word_vecs()
embedding_matrix = np.zeros((vocalen, 300))

for word, i in word_index.items():
    embedding_vector = word_vecs.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# 神经网络模型
model = Sequential()
model.add(
    Embedding(
        trainable=False, # 冻结这一层，不要让它预训练
        weights=[embedding_matrix],
        input_dim=vocalen, # 输入维度
        output_dim=300, # 输出维度
        input_length=maxlen # 序列长度
    )
)
model.add(LSTM(
        128, # 输出数据的维度
        return_sequences=True # 每一个都输出结果
))
model.add(LSTM(128))

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


