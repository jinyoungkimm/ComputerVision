import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Dropout,Dense
from tensorflow.keras.optimizers import Adam


(x_train,y_train),(x_test,y_test) = ds.cifar10.load_data()

x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)

model_cnn = Sequential()
model_cnn.add(Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model_cnn.add(Conv2D(32,(3,3),activation='relu'))
model_cnn.add(MaxPool2D(pool_size=(2,2)))

model_cnn.add(Dropout(0.25))

model_cnn.add(Conv2D(64,(3,3),activation='relu'))
model_cnn.add(Conv2D(64,(3,3),activation='relu'))
model_cnn.add(MaxPool2D(pool_size=(2,2)))

model_cnn.add(Dropout(0.25))
model_cnn.add(Flatten())

model_cnn.add(Dense(units=512,activation='relu'))
model_cnn.add(Dropout(0.5))
model_cnn.add(Dense(units=10,activation='softmax'))

model_cnn.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])
history = model_cnn.fit(x_train,y_train,batch_size=128,epochs=100,validation_data=(x_test,y_test),verbose=1)

result = model_cnn.evaluate(x_test,y_test)[1]*100
print("Accuracy : ", result)


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy Graph')
plt.ylabel('accuracy')
plt.xlabel('eopch')
plt.legend(['Train','Validation'])
plt.grid()
plt.show()



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Graph')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.grid()
plt.show()




