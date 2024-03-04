import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Dropout,Dense
from tensorflow.keras.optimizers import Adam


(x_train,y_train),(x_test,y_test) = ds.mnist.load_data()
x_train = x_train.reshape(60000,28,28,1)
x_train = x_train.astype(np.float32) / 255.0

x_test = x_test.reshape(10000,28,28,1)
x_test = x_test.astype(np.float32) / 255.0

y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)


model_cnn = Sequential()
model_cnn.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model_cnn.add(Conv2D(32,(3,3),activation='relu'))
model_cnn.add(MaxPool2D(pool_size=(2,2)))

model_cnn.add(Dropout(0.25))

model_cnn.add(Conv2D(64,(3,3),activation='relu'))
model_cnn.add(Conv2D(64,(3,3),activation='relu'))
model_cnn.add(MaxPool2D(pool_size=(2,2)))

model_cnn.add(Dropout(0.25))

model_cnn.add(Flatten())

model_cnn.add(Dense(units=512,activation='relu'))
model_cnn.add(Dropout(0.25))
model_cnn.add(Dense(units=10,activation = 'softmax'))

model_cnn.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])
history = model_cnn.fit(x_train,y_train,batch_size=128,epochs=100,validation_data=(x_test,y_test),verbose=1)

model_cnn.save("./model_cnn.h5")

result = model_cnn.evaluate(x_test,y_test)[1] * 100
print("정확률 = ", result)


import cv2 as cv
import matplotlib.pyplot as plt
import winsound

model = tf.keras.models.load_model('./model_cnn.h5') # PG 7-5에서 생성한 Model 로드. 

print("model",model)

def reset():
    global img

    img = np.ones((200,520,3),dtype=np.uint8)*255
    for i in range(5):
        cv.rectangle(img,(10+i*100,50),(10+(i+1)*100,150),(0,0,255))
    cv.putText(img,'e:erase, s:show, r:recognition, q:quit',(10,40),cv.FONT_HERSHEY_SIMPLEX, 0.8,(255,0,0),1)


def grab_numrals():
    numerals = []
    for i in range(5):
        roi = img[51:149,11+i*100:9+(i+1)*100,0]
        roi = 255 - cv.resize(roi,(28,28),interpolation=cv.INTER_CUBIC) # 점 연산 : 반전 그림
        numerals.append(roi)
    numerals = np.array(numerals)
    return numerals

def show():
    numerals = grab_numrals()
    plt.figure(figsize=(25,5))
    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.imshow(numerals[i], cmap = 'gray')
        plt.xticks([]);plt.yticks([])
    plt.show()


def recognition():

    numerals = grab_numrals()
    numerals = numerals.reshape(5,28,28,1)
    numerals = numerals.astype(np.float32) / 255.0

    res = model.predict(numerals)

    print("res",res)

    class_id = np.argmax(res,axis=1)

    print("class_id",class_id)

    for i in range(5):
        cv.putText(img,str(class_id[i]),(50+i*100,180),cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),1)

    winsound.Beep(1000,500)

BrushSiz = 4
LColor = (0,0,0)

def writing(event,x,y,flags,param):

    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img,(x,y),BrushSiz,LColor,-1)

    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        cv.circle(img,(x,y),BrushSiz,LColor,-1)

reset()
cv.namedWindow('Writing')
cv.setMouseCallback('Writing',writing)

while(True):

    cv.imshow("Writing",img)
    key = cv.waitKey(1)
    if   key == ord('e'):
        reset()

    elif key == ord('s'):
        show()

    elif key == ord('r'):
        recognition()

    elif key == ord('q'):
        break

cv.destroyAllWindows()





