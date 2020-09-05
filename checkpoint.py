from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import numpy as np

# Load dữ liệu
dataset = np.loadtxt("pima-indians-diabetes.data.txt", delimiter=",")

# Chia ra input X và output y
X = dataset[:,0:8]
Y = dataset[:,8]

# Tạo model
model = Sequential()
model.add(Dense(32, input_dim=8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Tạo callback
filepath="checkpoint.hdf5"
callback = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')

# Train model
model.fit(X, Y, validation_split=0.2, epochs=100, batch_size=8, callbacks=[callback])