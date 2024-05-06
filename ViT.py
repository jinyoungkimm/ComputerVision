import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import numpy as np

(x_train,y_train),(x_test,y_test) = keras.datasets.cifar10.load_data()

n_class = 10
img_siz = (32,32,3)

patch_siz = 4
p2 = (img_siz[0] // patch_siz) ** 2
print("p2111111111111111111111111111111",p2)
d_model = 64
h=8
N=6

# 32 x 32 x 3 => 16 x (8 x 8 x 3) => 

class Patches:#(layers.Layer):

    def __init__(self,patch_size):
        #super(Patches,self).__init__()
        self.p_siz = patch_size

    def call3232(self,img):


        batch_size = tf.shape(img)[0] # 첫번째 차원의 크기!
        print("batch size",batch_size)

        patches = tf.image.extract_patches(images = img, sizes=[1,self.p_siz, self.p_siz,1],strides=[1,self.p_siz,self.p_siz,1],rates=[1,1,1,1], padding="VALID")
        print("patches",patches.shape)


        patch_dims = patches.shape[-1]
        print("patch_dims",patch_dims)


        patches = tf.reshape(patches,[batch_size,-1,patch_dims])
        print("patches",patches.shape)    

        return patches


# x_train : (50000,32,32,3)
p = Patches(patch_siz)
p.call3232(x_train)


"""""""""
class PatchEncoder(layers.Layer):

    def __init__(self,p2,d_model):
    
        super(PatchEncoder,self).__init__()
        self.p2 = p2
        self.projection = layers.Dense(units=d_model)
        self.position_embedding = layers.Embedding(input_dim=p2, output_dim=d_model)

    def call(self,patch):
        positions = tf.range(start=0, limit=self.p2, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
def create_vit_classifier():
        
        input = layers.Input(shape=(img_siz))
        nor = layers.experimental.preprocessing.Normalization()(input)

        patches = Patches(patch_siz)(nor)
        x = PatchEncoder(p2,d_model)(patches)

        for _ in range(N):
            x1 = layers.LayerNormalization(epsilon = 1e-6)(x)
            x2 = layers.MultiHeadAttention(num_heads=h, key_dim=d_model//h, dropout=0.1)(x1,x1)
            x3 = layers.Add()([x2,x])
            x4 = layers.LayerNormalization(epsilon=1e-6)(x3)
            x5 = layers.Dense(d_model*2, activation=tf.nn.gelu)(x4)
            x6 = layers.Dropout(0.1)(x5)
            x7 = layers.Dense(d_model,activation=tf.nn.gelu)(x6)
            x8 = layers.Dropout(0.1)(x7)
            x = layers.Add()([x8,x3])

        

        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(2048,activation=tf.nn.gelu)(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1024,activation = tf.nn.gelu)(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(n_class, activation = 'softmax')(x)

        model = keras.Model(inputs = input, outputs = output)
        return model
    

print("tensroflow",tf.__version__)

model = create_vit_classifier()
model.layers[1].adapt(x_train)

model.compile(optimizer = Adam(), loss=SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

hist = model.fit(x_train,y_train,batch_size=128, epochs=100, validation_data = (x_test,y_test), verbose=1)

res = model.evaluate(x_test,y_test,verbose=0)
print("정확률",res[1]*100)

import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.grid()
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history(['val_loss']))
plt.title('Loss Graph')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.grid()
plt.show()
"""""""""
