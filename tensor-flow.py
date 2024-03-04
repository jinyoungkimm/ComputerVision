import tensorflow as tf
import tensorflow.keras.datasets as ds
import matplotlib.pyplot as plt

(x_train,y_train) , (x_test,y_test) = ds.mnist.load_data()

print(f"[x_train : {x_train.shape} y_train : {y_train.shape}], [x_test : {x_test.shape}, y_test : {y_test.shape}]")

plt.figure(figsize=(10,3))
plt.suptitle('MNIST',fontsize=30)
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(x_train[i],cmap='gray') # cmap : Color MAP
    plt.xticks([]);plt.yticks([])
    plt.title(str(y_train[i]),fontsize=30)
plt.show()

(x_train,y_train),(x_test,y_test) = ds.cifar10.load_data()
print(f"[x_train : {x_train.shape} y_train : {y_train.shape}], [x_test : {x_test.shape}, y_test : {y_test.shape}]")

class_names = ['airplane','car','bird','cat','deer','dog','frog','horse','ship','truck']

plt.figure(figsize=(24,3))
plt.suptitle("CIFAR-10",fontsize=30)
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(x_train[i])
    plt.xticks([]);plt.yticks([])
    plt.title(class_names[i])
plt.show()


