import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


(x_train,y_train),(x_test,y_test) = ds.cifar10.load_data()


x_train = x_train.reshape(50000,3072)
x_test = x_test.reshape(10000,3072)

x_train = x_train.astype(np.float32)/255.0
x_test = x_test.astype(np.float32)/255.0

y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)

model = Sequential()


model.add(Dense(units=1024,activation='relu',input_shape=(3072,)))
model.add(Dense(units=512,activation='relu'))
model.add(Dense(units=512,activation='relu'))
model.add(Dense(units=10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001),metrics=['accuracy'])

history = model.fit(x_train,y_train,batch_size=128,epochs=50,validation_data=(x_test,y_test),verbose=2)

res = model.evaluate(x_test,y_test,verbose=0)[1]*100

print("정확률 = ",res)

model.save("./model_trained.h5")

import matplotlib.pyplot as plt


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy Graph")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(['train','test'])
plt.grid()
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss Graph")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.legend(['train','test'])
plt.grid()
plt.show()









