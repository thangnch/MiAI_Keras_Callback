import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

# Định nghĩa hàm trả về LR
def scheduler(epoch, lr):
    # Nếu dưới 10 epoch
    if epoch < 5:
        # Trả về lr
        return float(lr)
    else:
        # Còn không thì trả về
        return float(lr * tf.math.exp(-0.1))


# Định nghĩa model
model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
model.compile(tf.keras.optimizers.SGD(), loss='mse')

print("Learning rate ban đầu = ", round(model.optimizer.lr.numpy(), 5))

# Train model và xem learning rate
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
X = np.arange(100).reshape(5, 20).astype(float)
y = np.zeros(5)
history = model.fit( X, y , epochs=8, callbacks=[callback], verbose=1)

print("Learning rate sau khi train xong 15 epochs = ", round(model.optimizer.lr.numpy(), 5))
