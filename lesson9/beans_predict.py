import dataset
import plot_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

m = 100
X, Y = dataset.get_beans(m)
plot_utils.show_scatter(X, Y)

model = Sequential()

model.add(Dense(units=8, activation='relu', input_dim=2))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.05), metrics=['accuracy'])
model.fit(X, Y, epochs=5000, batch_size=10)

pres = model.predict(X)

plot_utils.show_scatter_surface(X, Y, model)
