import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


(x_train,y_train),(x_test,y_test) = ds.mnist.load_data()


x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)

x_train = x_train.astype(np.float32)/255.0
x_test = x_test.astype(np.float32)/255.0

y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)

model_SGD = Sequential()

model_SGD.add(Dense(units=512,activation='tanh',input_shape=(784,)))
model_SGD.add(Dense(units=10,activation='softmax'))

model_SGD.compile(loss='mse',optimizer=SGD(learning_rate=0.01),metrics=['accuracy'])

history_sgd = model_SGD.fit(x_train,y_train,batch_size=128,epochs=50,validation_data=(x_test,y_test),verbose=1)

res = model_SGD.evaluate(x_test,y_test,verbose=0)
print("SGE 정확률=",res[1]*100)

from tensorflow.keras.optimizers import Adam

model_Adam = Sequential()
model_Adam.add(Dense(units=512, activation='tanh',input_shape=(784,)))
model_Adam.add(Dense(units=10,activation='softmax'))

model_Adam.compile(loss='MSE',optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])
history_adam = model_Adam.fit(x_train,y_train,batch_size=128,epochs=50,validation_data=(x_test,y_test), verbose=2)
res = model_Adam.evaluate(x_test,y_test,verbose=0)[1]*100
print("Adam 정확률=",res)


import matplotlib.pyplot as plt

plt.plot(history_sgd.history['accuracy'],'r--') # 각 Epoch에 대응하는 Accuracy 값을 Plot(여기서는 [점선] 그래프)형태로 포현
plt.plot(history_sgd.history['val_accuracy'],'r') #각 Epoch에 대응하는 Val_Accuracy 값을 Plot(여기서는 점 그래프) 형태로 표현
plt.plot(history_adam.history['accuracy'],'b--')
plt.plot(history_adam.history['val_accuracy'],'b')
plt.title('sgd vs adam')
plt.ylim((0.7,1.0)) # ylim : y limitation
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train_sgd','val_sgd','train_adam','val_adam']) # plt.legend() 함수는 그래프의 범례(legend)를 추가하는 함수입니다. 
                                                             # [범례]는 [각 선 또는 점에 대한 설명을 제공]하여 그래프를 이해하고 해석하는 데 도움이 됩니다. 
                                                             # 여기서 "legend"는 [설명]을 의미하는 단어입니다.
                                                             # plt.legend() 함수는 [그래프에 어떤 데이터가 표시되었는지] 설명하는 텍스트를 추가합니다.
#plt.grid()
plt.show()





