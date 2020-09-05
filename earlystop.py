import tensorflow as tf
import numpy as np

# Thiết lập hàm call back Early Stopping
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)

# Tạo lập model
model = tf.keras.models.Sequential([tf.keras.layers.Dense(512)])
model.compile(tf.keras.optimizers.SGD(), loss='mse')

# Sinh dữ liệu fake để train thử
X = np.arange(100).reshape(5, 20).astype(float)
y = np.zeros(5)

# Train model 10 epochs, batch_sizes = 1
history = model.fit( X, y , epochs=10, batch_size=1, callbacks=[callback])

# In history ra xem train mấy epochs
print("Số epoch đã train = ", len(history.history['loss']))